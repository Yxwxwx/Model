from tqdm.auto import tqdm


class TEBD:
    """Time Evolving Block Decimation (TEBD).

    Parameters
    ----------
    p0 : MatrixProductState
        Initial state.
    H : LocalHam1D
        Nearest-neighbor Hamiltonian.
    dt : float, optional
        Default time step. Cannot be set together with ``tol``.
    tol : float, optional
        Target Trotter error. Cannot be set together with ``dt``.
    t0 : float, optional
        Initial time.
    split_opts : dict, optional
        Compression options used after applying a two-site gate.
    imag : bool, optional
        Whether to perform imaginary-time evolution.
    """

    def __init__(
        self,
        p0,
        H,
        dt=None,
        tol=None,
        t0=0.0,
        split_opts=None,
        imag=False,
    ):
        # initial state
        self._pt = p0.canonicalize(0)
        self.L = self._pt.L

        # Hamiltonian
        self.H = H
        self.cyclic = H.cyclic
        self._ham_norm = H.mean_norm()

        # estimated accumulated Trotter error
        self._err = 0.0

        # time
        self.t0 = self.t = t0

        if dt is not None and tol is not None:
            raise ValueError("Can't set both ``dt`` and ``tol``.")

        self.dt = self._dt = dt
        self.tol = tol
        self.imag = imag

        # SVD/compression options
        self.split_opts = {} if split_opts is None else dict(split_opts)

    @property
    def pt(self):
        return self._pt.copy()

    @property
    def err(self):
        return self._err

    def _make_progress_bar(self, T):
        return tqdm(
            total=T - self.t,
            desc="TEBD",
            unit="t",
            dynamic_ncols=True,
            smoothing=0.1,
            bar_format=(
                "{desc}: {percentage:6.2f}%|{bar}| "
                "{n:.3f}/{total:.3f} "
                "[{elapsed}<{remaining}, {rate_fmt}] "
                "{postfix}"
            ),
        )

    def choose_time_step(self, tol, T, order):
        """Choose dt from a target Trotter error.

        Global Trotter error scales approximately as

            (T / dt) * dt**(order + 1)
            = T * dt**order.
        """
        return (tol / (T * self._ham_norm)) ** (1 / order)

    def _get_gate_from_ham(self, dt_frac, sites):
        """Get the unitary (exponentiated) gate for fraction of timestep
        ``dt_frac`` and sites ``sites``, cached.
        """
        imag_factor = 1.0 if self.imag else 1.0j
        return self.H.get_gate_expm(sites, -imag_factor * self._dt * dt_frac)

    def sweep(self, direction, dt_frac, dt=None, queue=False):
        """Perform a single sweep of gates and compression. This shifts the
        orthonognality centre along with the gates as they are applied and
        split.

        Parameters
        ----------
        direction : {'right', 'left'}
            Which direction to sweep. Right is even bonds, left is odd.
        dt_frac : float
            What fraction of dt substep to take.
        dt : float, optional
            Overide the current ``dt`` with a custom value.
        """

        # if custom dt set, scale the dt fraction
        if dt is not None:
            dt_frac *= dt / self._dt

        # ------ automatically combine consecutive sweeps of same time ------ #

        if not hasattr(self, "_queued_sweep"):
            self._queued_sweep = None

        if queue:
            # check for queued sweep
            if self._queued_sweep:
                # if matches, combine and continue
                if direction == self._queued_sweep[0]:
                    self._queued_sweep[1] += dt_frac
                    return
                # else perform the old, queue the new
                else:
                    new_queued_sweep = [direction, dt_frac]
                    direction, dt_frac = self._queued_sweep
                    self._queued_sweep = new_queued_sweep

            # just queue the new sweep
            else:
                self._queued_sweep = [direction, dt_frac]
                return

        # check if need to drain the queue first
        elif self._queued_sweep:
            queued_direction, queued_dt_frac = self._queued_sweep
            self._queued_sweep = None
            self.sweep(queued_direction, queued_dt_frac, queue=False)

        # ------------------------------------------------------------------- #

        if direction == "right":
            start_site_ind = 0
            final_site_ind = self.L - 1
            # Apply even gates:
            #
            #     o-<-<-<-<-<-<-<-<-<-   -<-<
            #     | | | | | | | | | |     | |       >~>~>~>~>~>~>~>~>~>~>~o
            #     UUU UUU UUU UUU UUU ... UUU  -->  | | | | | | | | | | | |
            #     | | | | | | | | | |     | |
            #      1   2   3   4   5  ==>
            #
            for i in range(start_site_ind, final_site_ind, 2):
                sites = (i, (i + 1) % self.L)
                U = self._get_gate_from_ham(dt_frac, sites)
                self._pt.left_canonize(start=max(0, i - 1), stop=i)
                self._pt.gate_split_(U, where=sites, absorb="right", **self.split_opts)

            if self.L % 2 == 1:
                self._pt.left_canonize_site(self.L - 2)
                if self.cyclic:
                    sites = (self.L - 1, 0)
                    U = self._get_gate_from_ham(dt_frac, sites)
                    self._pt.right_canonize_site(1)
                    self._pt.gate_split_(
                        U, where=sites, absorb="left", **self.split_opts
                    )

        elif direction == "left":
            if self.cyclic and (self.L % 2 == 0):
                sites = (self.L - 1, 0)
                U = self._get_gate_from_ham(dt_frac, sites)
                self._pt.right_canonize_site(1)
                self._pt.gate_split_(U, where=sites, absorb="left", **self.split_opts)

            final_site_ind = 1
            # Apply odd gates:
            #
            #     >->->-   ->->->->->->->->-o
            #     | | |     | | | | | | | | |       o~<~<~<~<~<~<~<~<~<~<~<
            #     | UUU ... UUU UUU UUU UUU |  -->  | | | | | | | | | | | |
            #     | | |     | | | | | | | | |
            #           <==  4   3   2   1
            #
            for i in reversed(range(final_site_ind, self.L - 1, 2)):
                sites = (i, (i + 1) % self.L)
                U = self._get_gate_from_ham(dt_frac, sites)
                self._pt.right_canonize(start=min(self.L - 1, i + 2), stop=i + 1)
                self._pt.gate_split_(U, where=sites, absorb="left", **self.split_opts)

            # one extra canonicalization not included in last split
            self._pt.right_canonize_site(1)

        # Renormalise after imaginary time evolution
        if self.imag:
            factor = self._pt[final_site_ind].norm()
            self._pt[final_site_ind] /= factor

    def _step_order2(self, tau=1, **sweep_opts):
        """Perform a single, second order step."""
        self.sweep("right", tau / 2, **sweep_opts)
        self.sweep("left", tau, **sweep_opts)
        self.sweep("right", tau / 2, **sweep_opts)

    def _step_order4(self, **sweep_opts):
        """Perform a single, fourth order step."""
        tau1 = tau2 = 1 / (4 * 4 ** (1 / 3))
        tau3 = 1 - 2 * tau1 - 2 * tau2
        self._step_order2(tau1, **sweep_opts)
        self._step_order2(tau2, **sweep_opts)
        self._step_order2(tau3, **sweep_opts)
        self._step_order2(tau2, **sweep_opts)
        self._step_order2(tau1, **sweep_opts)

    def step(self, order=2, dt=None, pbar=None, **sweep_opts):
        """Perform one TEBD time step."""

        {
            2: self._step_order2,
            4: self._step_order4,
        }[order](dt=dt, **sweep_opts)

        dt = self._dt if dt is None else dt

        self.t += dt

        # rough accumulated Trotter error estimate
        self._err += self._ham_norm * abs(dt) ** (order + 1)

        if pbar is not None:
            pbar.update(dt)

            pbar.set_postfix_str(
                f"t={self.t:.3f}  "
                f"dt={dt:.3g}  "
                f"chi={self._pt.max_bond()}  "
                f"err~{self._err:.2e}"
            )

    def _compute_sweep_dt_tol(self, T, dt, tol, order):
        """Determine the time step from either dt or target tolerance."""

        if dt is None:
            dt = self.dt

        if tol is None:
            tol = self.tol

        if not (dt or tol):
            raise ValueError("Must set one of ``dt`` and ``tol``.")
        if dt and tol:
            raise ValueError("Can't set both ``dt`` and ``tol``.")

        if dt is None:
            self._dt = self.choose_time_step(
                tol,
                T - self.t,
                order,
            )
        else:
            self._dt = dt

        return self._dt

    TARGET_TOL = 1e-13  # tolerance to have 'reached' target time

    def update_to(
        self,
        T,
        dt=None,
        tol=None,
        order=4,
        pbar=None,
    ):
        """Evolve the state to time ``T``."""

        if T < self.t - self.TARGET_TOL:
            raise NotImplementedError("Backward time evolution is not supported.")

        self._compute_sweep_dt_tol(
            T,
            dt,
            tol,
            order,
        )

        while self.t < T - self._dt:
            self.step(
                order=order,
                dt=None,
                queue=True,
                pbar=pbar,
            )

        final_dt = T - self.t

        if final_dt > self.TARGET_TOL:
            self.step(
                order=order,
                dt=final_dt,
                queue=False,
                pbar=pbar,
            )

    def at_times(
        self,
        ts,
        dt=None,
        tol=None,
        order=4,
        progress=True,
    ):
        """Yield the evolved state at each requested time."""

        ts = sorted(ts)
        T = ts[-1]

        # Determine dt once so tol refers to whole evolution.
        dt = self._compute_sweep_dt_tol(
            T,
            dt,
            tol,
            order,
        )

        pbar = self._make_progress_bar(T) if progress else None

        try:
            for t in ts:
                self.update_to(
                    t,
                    dt=dt,
                    tol=False,
                    order=order,
                    pbar=pbar,
                )

                yield self.pt

        finally:
            if pbar is not None:
                pbar.close()
