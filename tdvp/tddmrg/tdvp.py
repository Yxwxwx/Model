"""One-site and two-site TDVP based on quimb MPS/MPO objects.

The implementation deliberately mirrors ``tddmrg.dmrg`` and ``tddmrg.tebd``:

* the moving MPO environments and effective Hamiltonians come from DMRG;
* local time propagation uses ``expm_multiply`` like a Krylov exponential;
* ``order=1`` selects one-site TDVP;
* ``order=2`` selects two-site TDVP.

Only finite open-boundary MPS/MPOs are supported.
"""

from math import prod, sqrt
from uuid import uuid4

import numpy as np
from quimb.linalg.base_linalg import expm_multiply
from quimb.tensor import Tensor
from quimb.tensor.tensor_core import TNLinearOperator
from tqdm.auto import tqdm

try:
    # Normal package import: tddmrg.tdvp
    from .dmrg import MovingEnvironment, parse_two_site_indices
except ImportError:  # pragma: no cover - convenient for running this file directly
    from dmrg import MovingEnvironment, parse_two_site_indices


class TDVP:
    r"""Projector-splitting time-dependent variational principle for MPS.

    Parameters
    ----------
    p0 : MatrixProductState
        Initial MPS. It is copied and canonicalized with its center at site 0.
    H : MatrixProductOperator
        Hamiltonian MPO. Only open boundary conditions are supported.
    dt : float, optional
        Default physical time step.
    order : {1, 2}, optional
        Select the TDVP variant:

        * ``1``: one-site TDVP. The bond dimensions remain fixed and there is
          no SVD truncation.
        * ``2``: two-site TDVP. The bond dimensions can grow and are controlled
          by ``split_opts``.

        Here ``order`` selects the number of active sites; it is not the order
        of a Suzuki--Trotter formula. Both variants use a symmetric forward and
        backward projector-splitting sweep.
    t0 : float, optional
        Initial time.
    split_opts : dict, optional
        Options supplied to ``Tensor.split`` in two-site TDVP. Useful keys are
        ``max_bond``, ``cutoff``, ``cutoff_mode`` and ``method``.
    imag : bool, optional
        If true, use imaginary-time rather than real-time evolution.
    dense_threshold : int, optional
        Form a dense local effective Hamiltonian when its dimension is smaller
        than this value; otherwise use ``TNLinearOperator``.
    expm_backend : str, optional
        Backend passed to ``quimb.linalg.base_linalg.expm_multiply``.
    expm_opts : dict, optional
        Additional options passed to ``expm_multiply``.

    Notes
    -----
    The local projector-splitting steps are

    one-site TDVP::

        one-site forward evolution
        -> exact QR/LQ split
        -> zero-site (bond) backward evolution

    two-site TDVP::

        two-site forward evolution
        -> SVD split/truncation
        -> one-site backward evolution

    A full physical time step consists of a left-to-right half sweep and a
    right-to-left half sweep. The local problem at the turning point is evolved
    for a full time step.
    """

    TARGET_TOL = 1.0e-13

    def __init__(
        self,
        p0,
        H,
        dt=None,
        order=1,
        t0=0.0,
        split_opts=None,
        imag=False,
        *,
        dense_threshold=800,
        expm_backend="AUTO",
        expm_opts=None,
    ):
        if getattr(H, "cyclic", False):
            raise ValueError("TDVP only supports open-boundary MPOs.")
        if p0.L != H.L:
            raise ValueError("The MPS and MPO must have the same length.")

        self.order = self._validate_order(order)
        self.dt = dt
        self.t0 = self.t = float(t0)
        self.imag = bool(imag)
        self.dense_threshold = dense_threshold
        self.expm_backend = expm_backend
        self.expm_opts = {} if expm_opts is None else dict(expm_opts)

        default_split_opts = {
            "cutoff": 1.0e-10,
            "cutoff_mode": "sum2",
            "method": "svd",
        }
        if split_opts is not None:
            default_split_opts.update(split_opts)
        self.split_opts = default_split_opts

        # Keep exactly the same virtual bra--MPO--ket organization as DMRG.
        self._pt = p0.canonicalize(0)
        self.L = self._pt.L
        self._bra = self._pt.H
        self.H = H.copy()

        self._pt.add_tag("_KET")
        self._bra.add_tag("_BRA")
        self.H.add_tag("_HAM")
        self._pt.align_(self.H, self._bra)
        self.energy_tn = self._bra | self.H | self._pt

        # Two-site truncation diagnostics. One-site TDVP leaves these at zero.
        self._trunc_err = 0.0
        self._last_trunc_err = 0.0
        self._step_trunc_err_sq = 0.0

    # ------------------------------------------------------------------
    # Public state and diagnostics
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_order(order):
        order = int(order)
        if order not in (1, 2):
            raise ValueError("order must be 1 (1TDVP) or 2 (2TDVP).")
        return order

    @property
    def pt(self):
        """Return a copy of the current MPS without the internal ket tag."""
        psi = self._pt.copy()
        psi.drop_tags("_KET")
        return psi

    @property
    def state(self):
        """Alias of :attr:`pt`, matching the DMRG interface."""
        return self.pt

    @property
    def trunc_err(self):
        """Root-sum-square of the reported two-site split errors."""
        return self._trunc_err

    @property
    def last_trunc_err(self):
        """Root-sum-square truncation error from the latest physical step."""
        return self._last_trunc_err

    def compute_energy(self):
        """Contract ``<psi|H|psi>`` for the current, connected MPS."""
        return self.energy_tn ^ all

    # ------------------------------------------------------------------
    # Generic local effective Hamiltonian and exponential action
    # ------------------------------------------------------------------

    def _effective_hamiltonian(self, eff_tn, dims, bra_inds, ket_inds):
        """Construct a dense or lazy effective Hamiltonian from ``eff_tn``."""
        if prod(dims) < self.dense_threshold:
            return (eff_tn ^ "_HAM")["_HAM"].to_dense(bra_inds, ket_inds)

        return TNLinearOperator(
            eff_tn["_HAM"],
            ldims=dims,
            rdims=dims,
            left_inds=bra_inds,
            right_inds=ket_inds,
        )

    def _evolution_coefficient(self, dt, *, backward=False):
        # Real time:      exp(-i H dt), backward projector: exp(+i H dt)
        # Imaginary time: exp(-  H dt), backward projector: exp(+  H dt)
        coefficient = (-1.0 if self.imag else -1.0j) * dt
        return -coefficient if backward else coefficient

    def _expm_action(self, heff, vector, dt, *, backward=False):
        """Apply the local exponential to a flattened tensor."""
        coefficient = self._evolution_coefficient(dt, backward=backward)
        vector = np.asarray(vector).reshape(-1)

        result = expm_multiply(
            coefficient * heff,
            vector,
            backend=self.expm_backend,
            **self.expm_opts,
        )

        if hasattr(result, "toarray"):
            result = result.toarray()
        return np.asarray(result).reshape(-1)

    def _evolve_one_site(self, i, eff_tn, dt, *, backward=False):
        ket_inds = self._pt[i].inds
        bra_inds = self._bra[i].inds
        dims = self._pt[i].shape

        heff = self._effective_hamiltonian(eff_tn, dims, bra_inds, ket_inds)
        evolved = self._expm_action(
            heff,
            self._pt[i].data,
            dt,
            backward=backward,
        ).reshape(dims)

        self._pt[i].modify(data=evolved)
        self._bra[i].modify(data=evolved.conj())

    def _evolve_bond_tensor(self, C, bra_inds, eff_tn, dt):
        """Perform the backward zero-site evolution of a bond tensor."""
        heff = self._effective_hamiltonian(
            eff_tn,
            C.shape,
            bra_inds,
            C.inds,
        )
        evolved = self._expm_action(
            heff,
            C.data,
            dt,
            backward=True,
        ).reshape(C.shape)
        return Tensor(evolved, inds=C.inds, tags=C.tags)

    # ------------------------------------------------------------------
    # Index and tensor replacement helpers
    # ------------------------------------------------------------------

    def _ket_to_bra_index_map(self, *sites):
        mapping = {}
        for i in sites:
            ket = self._pt[i]
            bra = self._bra[i]
            if ket.shape != bra.shape:
                raise ValueError("Ket and bra site tensors have incompatible shapes.")
            mapping.update(zip(ket.inds, bra.inds))
        return mapping

    @staticmethod
    def _new_bra_bond():
        return f"_tdvp_bra_bond_{uuid4().hex}"

    @staticmethod
    def _shared_bond(left, right):
        shared = tuple(set(left.inds).intersection(right.inds))
        if len(shared) != 1:
            raise ValueError("A split tensor pair must share exactly one bond.")
        return shared[0]

    @staticmethod
    def _map_inds(inds, ket_to_bra):
        try:
            return tuple(ket_to_bra[ix] for ix in inds)
        except KeyError as exc:
            raise KeyError(
                f"Missing ket-to-bra mapping for index {exc.args[0]!r}."
            ) from exc

    def _replace_site_tensor(self, i, tensor, ket_to_bra):
        bra_inds = self._map_inds(tensor.inds, ket_to_bra)
        self._pt[i].modify(data=tensor.data, inds=tensor.inds)
        self._bra[i].modify(data=tensor.data.conj(), inds=bra_inds)

    def _absorb_bond_right(self, C, i, ket_to_bra):
        """Absorb ``C`` into site ``i`` from its left."""
        theta = C.contract(self._pt[i])
        self._replace_site_tensor(i, theta, ket_to_bra)

    def _absorb_bond_left(self, C, i, ket_to_bra):
        """Absorb ``C`` into site ``i`` from its right."""
        theta = self._pt[i].contract(C)
        self._replace_site_tensor(i, theta, ket_to_bra)

    # ------------------------------------------------------------------
    # One-site TDVP: exact split and zero-site backward evolution
    # ------------------------------------------------------------------

    def _split_one_site_right(self, i):
        """Write ``A_C[i] = A_L[i] C[i]`` without truncation."""
        old_bond = self._pt.bond(i, i + 1)
        ket_to_bra = self._ket_to_bra_index_map(i, i + 1)

        left_inds = tuple(ix for ix in self._pt[i].inds if ix != old_bond)
        A, C = self._pt[i].split(
            left_inds=left_inds,
            right_inds=(old_bond,),
            method="qr",
            absorb="right",
            get="tensors",
        )

        new_bond = self._shared_bond(A, C)
        ket_to_bra[new_bond] = self._new_bra_bond()
        self._replace_site_tensor(i, A, ket_to_bra)
        return C, ket_to_bra

    def _split_one_site_left(self, i):
        """Write ``A_C[i] = C[i-1] A_R[i]`` without truncation."""
        old_bond = self._pt.bond(i - 1, i)
        ket_to_bra = self._ket_to_bra_index_map(i - 1, i)

        right_inds = tuple(ix for ix in self._pt[i].inds if ix != old_bond)
        C, B = self._pt[i].split(
            left_inds=(old_bond,),
            right_inds=right_inds,
            method="lq",
            absorb="left",
            get="tensors",
        )

        new_bond = self._shared_bond(C, B)
        ket_to_bra[new_bond] = self._new_bra_bond()
        self._replace_site_tensor(i, B, ket_to_bra)
        return C, ket_to_bra

    @staticmethod
    def _reduce_environment(eff_tn, tags):
        """Contract selected active tensors into one environment block."""
        reduced = eff_tn.copy(virtual=True)
        reduced ^= tags
        return reduced

    def _step_1tdvp(self, dt):
        if self.L == 1:
            env = MovingEnvironment(self.energy_tn, begin="left", bsz=1)
            self._evolve_one_site(0, env(), dt)
            return

        half_dt = 0.5 * dt

        # ----------------------- left -> right -----------------------
        env = MovingEnvironment(self.energy_tn, begin="left", bsz=1)

        for i in range(self.L - 1):
            env.move_to(i)

            # + P_1: one-site forward evolution for half a step.
            self._evolve_one_site(i, env(), half_dt)

            # Move the center right: A_C[i] = A_L[i] C[i].
            C, ket_to_bra = self._split_one_site_right(i)

            # - P_0: zero-site backward evolution for half a step.
            zero_site_tn = self._reduce_environment(
                env(),
                ("_LEFT", env.site_tag(i)),
            )
            C_bra_inds = self._map_inds(C.inds, ket_to_bra)
            C = self._evolve_bond_tensor(C, C_bra_inds, zero_site_tn, half_dt)

            # Form the next center tensor A_C[i+1].
            self._absorb_bond_right(C, i + 1, ket_to_bra)

        # Turning point: this site represents the two half steps together.
        env.move_to(self.L - 1)
        self._evolve_one_site(self.L - 1, env(), dt)

        # ----------------------- right -> left -----------------------
        # Rebuild environments in the reverse direction. The minimal DMRG
        # MovingEnvironment is directional and intentionally does not maintain
        # both updated sides during a round trip.
        env = MovingEnvironment(self.energy_tn, begin="right", bsz=1)

        for i in range(self.L - 1, 0, -1):
            env.move_to(i)

            # The turning site was already evolved for a full step.
            if i != self.L - 1:
                self._evolve_one_site(i, env(), half_dt)

            # Move the center left: A_C[i] = C[i-1] A_R[i].
            C, ket_to_bra = self._split_one_site_left(i)

            zero_site_tn = self._reduce_environment(
                env(),
                ("_RIGHT", env.site_tag(i)),
            )
            C_bra_inds = self._map_inds(C.inds, ket_to_bra)
            C = self._evolve_bond_tensor(C, C_bra_inds, zero_site_tn, half_dt)

            self._absorb_bond_left(C, i - 1, ket_to_bra)

        # Complete the symmetric step at the left boundary.
        env.move_to(0)
        self._evolve_one_site(0, env(), half_dt)

    # ------------------------------------------------------------------
    # Two-site TDVP: two-site forward and one-site backward evolution
    # ------------------------------------------------------------------

    def _split_two_site(self, i, theta, metadata, *, absorb):
        (
            _dims,
            bra_inds_left,
            bra_inds_right,
            _bra_inds,
            ket_inds_left,
            ket_inds_right,
            _ket_inds,
            _bra_bond,
            _ket_bond,
        ) = metadata

        ket_to_bra = self._ket_to_bra_index_map(i, i + 1)
        info = {"error": None}

        split_opts = dict(self.split_opts)
        # These are controlled by the TDVP sweep itself.
        for key in ("absorb", "get", "left_inds", "right_inds", "info"):
            split_opts.pop(key, None)

        left, right = theta.split(
            left_inds=ket_inds_left,
            right_inds=ket_inds_right,
            absorb=absorb,
            get="tensors",
            info=info,
            **split_opts,
        )

        new_bond = self._shared_bond(left, right)
        ket_to_bra[new_bond] = self._new_bra_bond()

        # Add explicit mappings for the outer indices. They are normally
        # already present, but spelling them out makes the intended ordering
        # and bra/ket relation clear.
        ket_to_bra.update(zip(ket_inds_left, bra_inds_left))
        ket_to_bra.update(zip(ket_inds_right, bra_inds_right))

        self._replace_site_tensor(i, left, ket_to_bra)
        self._replace_site_tensor(i + 1, right, ket_to_bra)

        error = info.get("error")
        if error is not None:
            self._step_trunc_err_sq += float(abs(error)) ** 2

    def _evolve_two_site(self, i, eff_tn, dt, *, absorb):
        metadata = parse_two_site_indices(self._pt, self._bra, i)
        dims = metadata[0]
        bra_inds = metadata[3]
        ket_inds = metadata[6]

        heff = self._effective_hamiltonian(eff_tn, dims, bra_inds, ket_inds)
        theta0 = self._pt[i].contract(self._pt[i + 1]).to_dense(ket_inds)
        evolved = self._expm_action(heff, theta0, dt).reshape(dims)
        theta = Tensor(evolved, inds=ket_inds)

        self._split_two_site(i, theta, metadata, absorb=absorb)

    def _one_site_environment_from_two_site(self, env, i, *, move):
        """Reduce a two-site effective network to the overlap one-site problem."""
        if move == "right":
            # Contract the newly left-canonical site i into the left block,
            # leaving site i+1 active.
            return self._reduce_environment(
                env(),
                ("_LEFT", env.site_tag(i)),
            )

        # Contract the newly right-canonical site i+1 into the right block,
        # leaving site i active.
        return self._reduce_environment(
            env(),
            ("_RIGHT", env.site_tag(i + 1)),
        )

    def _step_2tdvp(self, dt):
        if self.L < 2:
            raise ValueError("2TDVP requires an MPS with at least two sites.")

        half_dt = 0.5 * dt

        # ----------------------- left -> right -----------------------
        env = MovingEnvironment(self.energy_tn, begin="left", bsz=2)

        # Non-turning two-site blocks: forward half step, then compensate the
        # overlapping site by a backward half step.
        for i in range(0, self.L - 2):
            env.move_to(i)
            self._evolve_two_site(i, env(), half_dt, absorb="right")

            one_site_tn = self._one_site_environment_from_two_site(
                env,
                i,
                move="right",
            )
            self._evolve_one_site(i + 1, one_site_tn, half_dt, backward=True)

        # Turning pair: a full forward step. For L > 2, move the center left
        # with the corresponding one-site backward half step.
        turning = self.L - 2
        env.move_to(turning)
        self._evolve_two_site(turning, env(), dt, absorb="left")

        if turning > 0:
            one_site_tn = self._one_site_environment_from_two_site(
                env,
                turning,
                move="left",
            )
            self._evolve_one_site(turning, one_site_tn, half_dt, backward=True)

        # For L == 2, the turning pair is also the final pair and the state is
        # already centered at site 0.
        if self.L == 2:
            return

        # ----------------------- right -> left -----------------------
        env = MovingEnvironment(self.energy_tn, begin="right", bsz=2)

        # Pairs between the turning pair and the final left boundary pair.
        for i in range(self.L - 3, 0, -1):
            env.move_to(i)
            self._evolve_two_site(i, env(), half_dt, absorb="left")

            one_site_tn = self._one_site_environment_from_two_site(
                env,
                i,
                move="left",
            )
            self._evolve_one_site(i, one_site_tn, half_dt, backward=True)

        # Final pair: forward half step only, leaving the center at site 0.
        env.move_to(0)
        self._evolve_two_site(0, env(), half_dt, absorb="left")

    # ------------------------------------------------------------------
    # Time-evolution driver and progress reporting
    # ------------------------------------------------------------------

    def _normalize_imaginary_time_state(self):
        if not self.imag:
            return

        # Both algorithms end with the orthogonality center at site 0.
        factor = self._pt[0].norm()
        if factor == 0:
            raise FloatingPointError("The imaginary-time state has zero norm.")
        self._pt[0].modify(data=self._pt[0].data / factor)
        self._bra[0].modify(data=self._pt[0].data.conj())

    def _resolve_dt(self, dt):
        dt = self.dt if dt is None else dt
        if dt is None:
            raise ValueError(
                "A time step must be supplied either at construction or call time."
            )
        dt = float(dt)
        if dt <= 0.0:
            raise ValueError("dt must be positive.")
        return dt

    def _make_progress_bar(self, T, order, *, leave=True):
        return tqdm(
            total=T - self.t,
            desc=f"{order}TDVP",
            unit="t",
            dynamic_ncols=True,
            smoothing=0.1,
            mininterval=0.1,
            leave=leave,
            bar_format=(
                "{desc}: {percentage:6.2f}%|{bar}| "
                "{n:.3f}/{total:.3f} "
                "[{elapsed}<{remaining}, {rate_fmt}]"
                "{postfix}"
            ),
        )

    def step(self, order=None, dt=None, pbar=None):
        """Perform one complete symmetric TDVP time step."""
        order = self.order if order is None else self._validate_order(order)
        dt = self._resolve_dt(dt)

        self._step_trunc_err_sq = 0.0

        if order == 1:
            self._step_1tdvp(dt)
        else:
            self._step_2tdvp(dt)

        self._normalize_imaginary_time_state()

        self.t += dt
        self._last_trunc_err = sqrt(self._step_trunc_err_sq)
        self._trunc_err = sqrt(self._trunc_err**2 + self._step_trunc_err_sq)

        if pbar is not None:
            postfix = f"t={self.t:.3f}  dt={dt:.3g}  chi={self._pt.max_bond()}"
            if order == 2:
                postfix += f"  trunc={self._last_trunc_err:.2e}"

            # Set the status first and let update() perform the only refresh.
            # This avoids rendering the progress bar twice per TDVP step.
            pbar.set_postfix_str(postfix, refresh=False)

            # Clamp the final update against floating-point roundoff so the
            # displayed progress ends at exactly 100%.
            remaining = max(0.0, pbar.total - pbar.n)
            pbar.update(min(dt, remaining))

    def update_to(self, T, dt=None, order=None, pbar=None):
        """Evolve the state forward to physical time ``T``."""
        T = float(T)
        if T < self.t - self.TARGET_TOL:
            raise NotImplementedError(
                "Backward evolution in physical time is not supported."
            )

        step_dt = self._resolve_dt(dt)

        while self.t < T - step_dt:
            self.step(order=order, dt=step_dt, pbar=pbar)

        final_dt = T - self.t
        if final_dt > self.TARGET_TOL:
            self.step(order=order, dt=final_dt, pbar=pbar)

    def at_times(self, ts, dt=None, order=None, progress=True, leave=True):
        """Yield a copied MPS at each requested physical time.

        Parameters
        ----------
        leave : bool, optional
            Whether the completed progress bar remains in the terminal.
        """
        ts = sorted(float(t) for t in ts)
        if not ts:
            return

        if ts[0] < self.t - self.TARGET_TOL:
            raise ValueError("Requested times must not precede the current TDVP time.")

        order = self.order if order is None else self._validate_order(order)
        step_dt = self._resolve_dt(dt)
        T = ts[-1]
        pbar = self._make_progress_bar(T, order, leave=leave) if progress else None

        try:
            for t in ts:
                self.update_to(t, dt=step_dt, order=order, pbar=pbar)
                yield self.pt
        finally:
            if pbar is not None:
                pbar.close()
