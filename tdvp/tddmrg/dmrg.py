import itertools
from math import prod

from quimb.linalg.base_linalg import eigh
from quimb.tensor import Tensor
from quimb.tensor.tensor_core import TNLinearOperator
from tqdm import tqdm


class MovingEnvironment:
    r"""Helper class for efficiently moving the effective 'environment' of a
    few sites in a 1D tensor network. E.g. for ``begin='left', bsz=2``, this
    initializes the right environments like so::

        n - 1: ●─●─●─     ─●─●─●
               │ │ │       │ │ │
               H─H─H─ ... ─H─H─H
               │ │ │       │ │ │
               ●─●─●─     ─●─●─●

        n - 2: ●─●─●─     ─●─●─╮
               │ │ │       │ │ ●
               H─H─H─ ... ─H─H─H
               │ │ │       │ │ ●
               ●─●─●─     ─●─●─╯

        n - 3: ●─●─●─     ─●─╮
               │ │ │       │ ●●
               H─H─H─ ... ─H─HH
               │ │ │       │ ●●
               ●─●─●─     ─●─╯

        ...

        0    : ●─●─╮
               │ │ ●●   ●●●
               H─H─HH...HHH
               │ │ ●●   ●●●
               ●─●─╯

    which can then be used to efficiently generate the left environments as
    each site is updated. For example if ``bsz=2`` and the environements have
    been shifted many sites into the middle, then ``MovingEnvironment()``
    returns something like::

             <---> bsz sites
            ╭─●─●─╮
        ●●●●● │ │ ●●●●●●●
        HHHHH─H─H─HHHHHHH
        ●●●●● │ │ ●●●●●●●
            ╰─●─●─╯
        0 ... i i+1 ... n-1

    Parameters
    ----------
    tn
        Closed tensor network, e.g. <psi|H|psi>.
    begin : {"left", "right"}
        Side from which the sweep starts.
    bsz : int
        Number of active sites. For two-site DMRG, bsz=2.

    Notes
    -----
    This is deliberately OBC-only. There is no segmentation, transfer-matrix
    compression, pseudo-orthogonalization, or effective norm operator.
    """

    def __init__(self, tn, begin, bsz):
        self.tn = tn.copy(virtual=True)
        self.L = tn.L
        self.bsz = bsz
        self._site_tag_id = tn.site_tag_id

        # Valid active-block starting positions:
        # bsz=1 -> 0, ..., L-1
        # bsz=2 -> 0, ..., L-2
        self.first = 0
        self.last = self.L - self.bsz

        if self.last < self.first:
            raise ValueError("Active block is larger than the system.")

        self._initialize(begin)

    def site_tag(self, i):
        return self._site_tag_id.format(i)

    def _initialize(self, begin):
        self.tnc = self.tn.copy(virtual=True)

        # Dummy OBC end pieces. They make the environment update logic uniform.
        self.tnc |= Tensor(tags="_LEFT").astype(self.tn.dtype)
        self.tnc |= Tensor(tags="_RIGHT").astype(self.tn.dtype)

        if begin == "left":
            i = self.last
            initial_tags = ["_RIGHT"] + [self.site_tag(i + b) for b in range(self.bsz)]
            self.envs = {i: self.tnc.select_any(initial_tags)}

            for j in reversed(range(self.first, self.last)):
                self.envs[j] = self.envs[j + 1].copy(virtual=True)
                self.envs[j] |= self.tnc.select(j)
                self.envs[j] ^= ("_RIGHT", self.site_tag(j + self.bsz))

            self.envs[self.first] |= self.tnc["_LEFT"]
            self.pos = self.first

        elif begin == "right":
            i = self.first
            initial_tags = ["_LEFT"] + [self.site_tag(i + b) for b in range(self.bsz)]
            self.envs = {i: self.tnc.select_any(initial_tags)}

            for j in range(self.first + 1, self.last + 1):
                self.envs[j] = self.envs[j - 1].copy(virtual=True)
                self.envs[j] |= self.tnc.select(j + self.bsz - 1)
                self.envs[j] ^= ("_LEFT", self.site_tag(j - 1))

            self.envs[self.last] |= self.tnc["_RIGHT"]
            self.pos = self.last

        else:
            raise ValueError("begin must be 'left' or 'right'.")

    def move_right(self):
        if self.pos >= self.last:
            raise ValueError("Environment is already at the right boundary.")

        i = self.pos + 1
        self.pos = i

        new_left = self.envs[i - 1].select(["_LEFT", self.site_tag(i - 1)], which="any")
        self.envs[i] |= new_left ^ all

    def move_left(self):
        if self.pos <= self.first:
            raise ValueError("Environment is already at the left boundary.")

        i = self.pos - 1
        self.pos = i

        new_right = self.envs[i + 1].select(
            ["_RIGHT", self.site_tag(i + self.bsz)], which="any"
        )
        self.envs[i] |= new_right ^ all

    def move_to(self, i):
        if not (self.first <= i <= self.last):
            raise ValueError(
                f"Active block start must satisfy {self.first} <= i <= {self.last}."
            )

        while self.pos < i:
            self.move_right()
        while self.pos > i:
            self.move_left()

    def __call__(self):
        return self.envs[self.pos]


def parse_two_site_indices(ket, bra, i):
    """Collect dimensions and index groups for sites i and i+1."""
    ket_bond = ket.bond(i, i + 1)

    dims_left, ket_inds_left = zip(
        *((d, ix) for d, ix in zip(ket[i].shape, ket[i].inds) if ix != ket_bond)
    )
    dims_right, ket_inds_right = zip(
        *((d, ix) for d, ix in zip(ket[i + 1].shape, ket[i + 1].inds) if ix != ket_bond)
    )

    bra_bond = bra.bond(i, i + 1)
    bra_inds_left = tuple(ix for ix in bra[i].inds if ix != bra_bond)
    bra_inds_right = tuple(ix for ix in bra[i + 1].inds if ix != bra_bond)

    dims = dims_left + dims_right
    bra_inds = bra_inds_left + bra_inds_right
    ket_inds = ket_inds_left + ket_inds_right

    return (
        dims,
        bra_inds_left,
        bra_inds_right,
        bra_inds,
        ket_inds_left,
        ket_inds_right,
        ket_inds,
        bra_bond,
        ket_bond,
    )


class DMRG:
    """Minimal open-boundary one-/two-site DMRG built on quimb.

    This is deliberately organized as preparation for 1TDVP and 2TDVP:
    both variants share moving environments and effective Hamiltonians,
    while the local update differs between one-site and two-site blocks.
    """

    def __init__(
        self,
        ham,
        bond_dims,
        cutoffs=1e-9,
        bsz=2,
        p0=None,
        *,
        eig_tol=1e-3,
        eig_ncv=4,
        eig_maxiter=None,
        dense_threshold=800,
    ):
        if getattr(ham, "cyclic", False):
            raise ValueError("DMRG only supports open-boundary MPOs.")
        if bsz not in (1, 2):
            raise ValueError("bsz must be 1 or 2.")

        self.L = ham.L
        self.bsz = bsz

        self.eig_tol = eig_tol
        self.eig_ncv = eig_ncv
        self.eig_maxiter = eig_maxiter
        self.dense_threshold = dense_threshold

        self._set_bond_dim_schedule(bond_dims)
        self._set_cutoff_schedule(cutoffs)

        if p0 is None:
            self._ket = ham.rand_state(self._bond_dim0)
        else:
            self._ket = p0.copy()

        self._bra = self._ket.H
        self.ham = ham.copy()

        self._ket.add_tag("_KET")
        self._bra.add_tag("_BRA")
        self.ham.add_tag("_HAM")

        # Align physical indices and create a virtual combined network.
        self._ket.align_(self.ham, self._bra)
        self.energy_tn = self._bra | self.ham | self._ket

        self.energies = []

    @staticmethod
    def _schedule(values):
        if isinstance(values, (int, float)):
            values = (values,)
        else:
            values = tuple(values)

        if not values:
            raise ValueError("Schedule cannot be empty.")

        return itertools.chain(values, itertools.repeat(values[-1]))

    def _set_bond_dim_schedule(self, bond_dims):
        if isinstance(bond_dims, int):
            values = (bond_dims,)
        else:
            values = tuple(bond_dims)

        self._bond_dim0 = values[0]
        self._bond_dims = self._schedule(values)

    def _set_cutoff_schedule(self, cutoffs):
        self._cutoffs = self._schedule(cutoffs)

    @property
    def energy(self):
        return self.energies[-1]

    @property
    def state(self):
        psi = self._ket.copy()
        psi.drop_tags("_KET")
        return psi

    def _solve_local_ground_state(self, heff, v0):
        """Solve the lowest local eigenpair."""
        return eigh(
            heff,
            k=1,
            which="SA",
            v0=v0,
            ncv=self.eig_ncv,
            tol=self.eig_tol,
            maxiter=self.eig_maxiter,
            fallback_to_scipy=True,
        )

    def _effective_hamiltonian(self, dims, bra_inds, ket_inds):
        """Build the effective Hamiltonian for the current active block."""
        self._eff_tn = self.moving_env()

        if prod(dims) < self.dense_threshold:
            return (self._eff_tn ^ "_HAM")["_HAM"].to_dense(bra_inds, ket_inds)

        return TNLinearOperator(
            self._eff_tn["_HAM"],
            ldims=dims,
            rdims=dims,
            left_inds=bra_inds,
            right_inds=ket_inds,
        )

    def _canonize_after_one_site_update(self, i, direction):
        """Move the orthogonality center after a one-site update."""
        if direction == "right" and i < self.L - 1:
            self._ket.left_canonize_site(i, bra=self._bra)
        elif direction == "left" and i > 0:
            self._ket.right_canonize_site(i, bra=self._bra)

    def _update_one_site(self, i, direction, **_):
        r"""Find the single site effective tensor groundstate of::

            >->->->->-/|\-<-<-<-<-<-<-<-<          /|\       <-- uix
            | | | | |  |  | | | | | | | |         / | \
            H-H-H-H-H--H--H-H-H-H-H-H-H-H   =    L--H--R
            | | | | | i|  | | | | | | | |         \i| /
            >->->->->-\|/-<-<-<-<-<-<-<-<          \|/       <-- lix

        And insert it back into the states ``k`` and ``b``, and thus
        ``TN_energy``.
        """
        ket_inds = self._ket[i].inds
        bra_inds = self._bra[i].inds
        dims = self._ket[i].shape

        heff = self._effective_hamiltonian(dims, bra_inds, ket_inds)
        theta0 = self._ket[i].data.ravel()

        local_energy, theta = self._solve_local_ground_state(heff, theta0)
        theta = theta.toarray().reshape(dims)

        self._ket[i].modify(data=theta)
        self._bra[i].modify(data=theta.conj())

        # With OBC + canonical MPS the effective norm is the identity.
        # Canonicalizing here moves the center to the next site in the sweep.
        self._canonize_after_one_site_update(i, direction)

        return local_energy.item()

    def _update_two_sites(
        self,
        i,
        direction,
        *,
        max_bond,
        cutoff,
        cutoff_mode="sum2",
        method="svd",
    ):
        r"""Find the 2-site effective tensor groundstate of::

            >->->->->-/| |\-<-<-<-<-<-<-<-<          /| |\
            | | | | |  | |  | | | | | | | |         / | | \
            H-H-H-H-H--H-H--H-H-H-H-H-H-H-H   =    L--H-H--R
            | | | | |  i i+1| | | | | | | |         \ | | /
            >->->->->-\| |/-<-<-<-<-<-<-<-<          \| |/
                                                 i i+1

        And insert it back into the states ``k`` and ``b``, and thus
        ``TN_energy``.
        """
        (
            dims,
            bra_inds_left,
            bra_inds_right,
            bra_inds,
            ket_inds_left,
            ket_inds_right,
            ket_inds,
            bra_bond,
            ket_bond,
        ) = parse_two_site_indices(self._ket, self._bra, i)

        heff = self._effective_hamiltonian(dims, bra_inds, ket_inds)

        # Current two-site tensor is an excellent initial guess for the local solve.
        theta0 = self._ket[i].contract(self._ket[i + 1]).to_dense(ket_inds)

        local_energy, theta = self._solve_local_ground_state(heff, theta0)

        # Split and truncate. `absorb=direction` also moves the orthogonality center.
        theta = Tensor(theta.toarray().reshape(dims), ket_inds)
        left, right = theta.split(
            left_inds=ket_inds_left,
            right_inds=ket_inds_right,
            get="arrays",
            absorb=direction,
            max_bond=max_bond,
            cutoff=cutoff,
            cutoff_mode=cutoff_mode,
            method=method,
        )

        # Update ket and bra tensors in place so the virtual environment network
        # continues to see the current state.
        self._ket[i].modify(data=left, inds=(*ket_inds_left, ket_bond))
        self._bra[i].modify(data=left.conj(), inds=(*bra_inds_left, bra_bond))

        self._ket[i + 1].modify(data=right, inds=(ket_bond, *ket_inds_right))
        self._bra[i + 1].modify(data=right.conj(), inds=(bra_bond, *bra_inds_right))

        return local_energy.item()

    def _update_local(self, i, direction, **update_opts):
        self.moving_env.move_to(i)

        if self.bsz == 1:
            return self._update_one_site(i, direction, **update_opts)
        return self._update_two_sites(i, direction, **update_opts)

    def sweep(
        self,
        direction,
        *,
        max_bond,
        cutoff,
        canonize=True,
        cutoff_mode="sum2",
        method="svd",
        verbosity=0,
    ):
        r"""Perform a sweep of optimizations, either rightwards::

              optimize -->
                ...
            >->-o-<-<-<-<-<-<-<-<-<-<-<-<-<
            | | | | | | | | | | | | | | | |
            H-H-H-H-H-H-H-H-H-H-H-H-H-H-H-H
            | | | | | | | | | | | | | | | |
            >->-o-<-<-<-<-<-<-<-<-<-<-<-<-<

        or leftwards (`direction='L'`)::

                            <-- optimize
                                      ...
            >->->->->->->->->->->->->-o-<-<
            | | | | | | | | | | | | | | | |
            H-H-H-H-H-H-H-H-H-H-H-H-H-H-H-H
            | | | | | | | | | | | | | | | |
            >->->->->->->->->->->->->-o-<-<

        After the sweep the state is left or right canonized respectively.

        Parameters
        ----------
        direction : {'R', 'L'}
            Sweep from left to right (->) or right to left (<-) respectively.
        canonize : bool, optional
            Canonize the state first, not needed if doing alternate sweeps.
        verbosity : {0, 1, 2}, optional
            Show a progress bar for the sweep.
        update_opts :
            Supplied to ``self._update_local``.
        """
        if direction not in {"R", "L"}:
            raise ValueError("direction must be 'R' or 'L'.")

        if canonize:
            if direction == "R":
                self._ket.right_canonize(bra=self._bra)
            else:
                self._ket.left_canonize(bra=self._bra)

        if direction == "R":
            begin = "left"
            sites = range(0, self.L - self.bsz + 1)
            move = "right"
        else:
            begin = "right"
            sites = range(self.L - self.bsz, -1, -1)
            move = "left"

        self.moving_env = MovingEnvironment(
            self.energy_tn,
            begin=begin,
            bsz=self.bsz,
        )

        if verbosity >= 2:
            sites = tqdm(
                sites,
                ascii=True,
                ncols=80,
                desc=f"{direction} sweep",
            )

        energy = None
        for i in sites:
            energy = self._update_local(
                i,
                move,
                max_bond=max_bond,
                cutoff=cutoff,
                cutoff_mode=cutoff_mode,
                method=method,
            )

            if verbosity >= 3:
                print(
                    f"  site={i:4d}  "
                    f"E_local={energy:.12f}  "
                    f"max_bond={self._ket.max_bond()}",
                    flush=True,
                )

        return energy

    def solve(
        self,
        *,
        tol=1e-8,
        max_sweeps=10,
        sweep_sequence="RL",
        verbosity=2,
    ):
        """Run DMRG sweeps until the sweep energy converges.

        verbosity
            0: silent
            1: one summary line per sweep
            2: summary + site progress bar
            3: summary + local energy after every site update
        """
        directions = itertools.cycle(sweep_sequence)
        previous_direction = None

        for sweep_no in range(1, max_sweeps + 1):
            direction = next(directions)
            max_bond = next(self._bond_dims)
            cutoff = next(self._cutoffs)

            # If the previous sweep ended at the correct boundary, the MPS is
            # already in the canonical form required by the opposite sweep.
            canonize = not ((previous_direction, direction) in {("R", "L"), ("L", "R")})

            if verbosity >= 1:
                print(
                    f"Sweep {sweep_no:3d}  "
                    f"direction={direction}  "
                    f"bsz={self.bsz}  "
                    f"bond={self._ket.max_bond()}/{max_bond}  "
                    f"cutoff={cutoff:.1e}",
                    flush=True,
                )

            if self.bsz == 1:
                self._ket.expand_bond_dimension(
                    max_bond,
                    bra=self._bra,
                    rand_strength=1e-6,
                )

            energy = self.sweep(
                direction,
                max_bond=max_bond,
                cutoff=cutoff,
                canonize=canonize,
                verbosity=verbosity,
            )
            self.energies.append(energy)

            delta_e = None
            if len(self.energies) >= 2:
                delta_e = abs(self.energies[-1] - self.energies[-2])

            converged = (delta_e is not None) and (delta_e < tol)

            if verbosity >= 1:
                de_text = "--------" if delta_e is None else f"{delta_e:.3e}"
                status = "  converged" if converged else ""
                print(
                    f"  E = {energy:.12f}  "
                    f"dE = {de_text}  "
                    f"max_bond = {self._ket.max_bond()}"
                    f"{status}",
                    flush=True,
                )

            if converged:
                return True

            previous_direction = direction

        if verbosity >= 1:
            print(
                f"Not converged after {max_sweeps} sweeps.",
                flush=True,
            )

        return False
