import numpy as np
import scipy.linalg
import numpy.typing as npt
import psi4


class CQED_RHF:
    def __init__(
        self, molecule_string, lambda_vec: npt.NDArray, psi4_options_dict: dict
    ):
        """Initialize RHF calculator with a PySCF Mole object."""
        self.mol_ = psi4.geometry(molecule_string)
        self.lambda_vec_ = lambda_vec
        psi4.set_options(psi4_options_dict)
        self.psi4_rhf_energy_, self.wfn_ = psi4.energy("scf", return_wfn=True)
        self.mints_ = psi4.core.MintsHelper(self.wfn_.basisset())

        # Basic parameters
        self.ndocc_ = self.wfn_.nalpha()
        self.nao_ = np.asarray(self.wfn_.Ca()).shape[0]

        # Initialize integral matrices
        self.S_ = np.asarray(self.mints_.ao_overlap())
        self.H_ = np.asarray(self.mints_.ao_kinetic()) + np.asarray(
            self.mints_.ao_potential()
        )

        self.eri_ = np.asarray(self.mints_.ao_eri())  # Electron repulsion integrals

        self.Qij_ = np.zeros((self.nao_, self.nao_))  # Quadrupole integrals
        self.ao_dipole_ = np.zeros((3, self.nao_, self.nao_))  # Dipole integrals

        # Nuclear repulsion energy
        self.E_nn_ = self.mol_.nuclear_repulsion_energy()
        self.mu_nuc_val_ = np.array(
            [
                self.mol_.nuclear_dipole()[0],
                self.mol_.nuclear_dipole()[1],
                self.mol_.nuclear_dipole()[2],
            ],
            dtype=float,
        )

    def _compute_all_integrals(self):
        """Precompute all necessary CQED integrals before starting SCF."""
        Q = np.asarray(self.mints_.ao_quadrupole())
        Q_ao_xx = Q[0]
        Q_ao_xy = Q[1]
        Q_ao_xz = Q[2]
        Q_ao_yy = Q[3]
        Q_ao_yz = Q[4]
        Q_ao_zz = Q[5]

        # -1/2 * \sum_{\zeta, \zeta'} \lambda^{\zeta} * \lambda^{\zeta'} * Q_{ij}^{\zeta\zeta'}
        self.Qij_ = -0.5 * (
            self.lambda_vec_[0] ** 2 * Q_ao_xx
            + self.lambda_vec_[1] ** 2 * Q_ao_yy
            + self.lambda_vec_[2] ** 2 * Q_ao_zz
            + 2.0 * self.lambda_vec_[0] * self.lambda_vec_[1] * Q_ao_xy
            + 2.0 * self.lambda_vec_[0] * self.lambda_vec_[2] * Q_ao_xz
            + 2.0 * self.lambda_vec_[1] * self.lambda_vec_[2] * Q_ao_yz
        )

        self.ao_dipole_ = np.asarray(self.mints_.ao_dipole())
        self.l_dot_mu_nuc_ = np.einsum(
            "x,x->", self.lambda_vec_, self.mu_nuc_val_, optimize=True
        )
        self.l_dot_ao_dipole_ = np.einsum(
            "x, xij->ij", self.lambda_vec_, self.ao_dipole_, optimize=True
        )

    def compute_mu_exp(self, dm: npt.NDArray):
        r"""
        # <\mu> = 2.0 * \sum_i^N_{occ} <i|\mu|i> + \mu_{nuc}
        """
        return self.mu_nuc_val_ + 2.0 * np.einsum(
            "xij,ji->x", self.ao_dipole_, dm, optimize=True
        )

    def cqed_dij(self, dm: npt.NDArray, mu_exp_val: float = None):
        r"""
        # \sum_{\zeta} \lambda_{\zeta} * \mu_{ij}^{\zeta} * (\lambda \dot \mu_{nuc} - \lambda \dot <\mu>)
        """
        if mu_exp_val is None:
            mu_exp_val = self.compute_mu_exp(dm)

        dij = (
            self.l_dot_mu_nuc_
            - np.einsum("x,x->", self.lambda_vec_, mu_exp_val, optimize=True)
        ) * self.l_dot_ao_dipole_
        return dij

    def cqed_h1e(self, dm: npt.NDArray, mu_exp_val: float = None):
        r"""
        # H_{ij} = H_{ij}
        #          -1/2 * \sum_{\zeta, \zeta'} \lambda^{\zeta} * \lambda^{\zeta'} * Q_{ij}^{\zeta\zeta'}
        #          + \sum_{\zeta} \lambda_{\zeta} * \mu_{ij}^{\zeta} * (\lambda \dot \mu_{nuc} - \lambda \dot <\mu>)
        """
        if mu_exp_val is None:
            mu_exp_val = self.compute_mu_exp(dm)

        dij = self.cqed_dij(dm, mu_exp_val)
        return self.H_ + self.Qij_ + dij

    def scf_veff(self, dm: npt.NDArray):
        r"""
        # V_{ij} = \sum_{kl} (2(ij|kl) - (ik|jl)) * D_{kl}
        """
        return 2.0 * np.einsum("ijkl,kl->ij", self.eri_, dm, optimize=True) - np.einsum(
            "ikjl,kl->ij", self.eri_, dm, optimize=True
        )

    def dipole_veff(self, dm: npt.NDArray):
        r"""
        # \sum_{\zeta, \zeta'}\lambda^{\zeta} * \lambda^{\zeta'} *
        # (2\mu_{ij}^{\zeta}\mu_{kl}^{\zeta'} - \mu_{ik}^{\zeta}\mu_{jl}^{\zeta'}) * D_{kl}
        """
        return 2.0 * np.einsum(
            "ij, kl, kl->ij",
            self.l_dot_ao_dipole_,
            self.l_dot_ao_dipole_,
            dm,
            optimize=True,
        ) - np.einsum(
            "ik, jl, kl->ij",
            self.l_dot_ao_dipole_,
            self.l_dot_ao_dipole_,
            dm,
            optimize=True,
        )

    def cqed_veff(self, dm: npt.NDArray):
        return self.scf_veff(dm) + self.dipole_veff(dm)

    def get_fock(
        self, dm: npt.NDArray, h1e: npt.NDArray = None, mu_exp_val: float = None
    ) -> npt.NDArray:
        r"""
        # F = H + G
        """
        if mu_exp_val is None:
            mu_exp_val = self.compute_mu_exp(dm)
        if h1e is None:
            h1e = self.cqed_h1e(dm, mu_exp_val)
        return h1e + self.cqed_veff(dm)

    def make_density(self, fock: npt.NDArray) -> npt.NDArray:
        """Create new density matrix and calculate electronic energy."""
        # Solve eigenvalue problem
        _, C = scipy.linalg.eigh(fock, self.S_)
        # Form density matrix
        C_occ = C[:, : self.ndocc_]
        return np.einsum("pi,qi->pq", C_occ, C_occ, optimize=True)

    def get_energy_elec(
        self, h1e: npt.NDArray, F: npt.NDArray, dm: npt.NDArray
    ) -> float:
        return np.einsum("pq,pq->", (h1e + F), dm, optimize=True)

    def get_energy_dc(self, dm: npt.NDArray, mu_exp_val: float = None) -> float:
        r"""
        d_c = 0.5 * (\lambda \dot \mu_{nuc}) ** 2 - (\lambda \dot <\mu>)(\lambda \dot \mu_{nuc}) + 0.5 * (\lambda \dot <\mu>)**2
        """
        if mu_exp_val is None:
            mu_exp_val = self.compute_mu_exp(dm)
        l_dot_mu_exp = np.einsum("x,x->", self.lambda_vec_, mu_exp_val, optimize=True)
        return (
            0.5 * self.l_dot_mu_nuc_**2
            - l_dot_mu_exp * self.l_dot_mu_nuc_
            + 0.5 * l_dot_mu_exp**2
        )

    def get_energy_tot(
        self,
        h1e: npt.NDArray,
        F: npt.NDArray,
        dm: npt.NDArray,
        mu_exp_val: float = None,
    ) -> float:
        if mu_exp_val is None:
            mu_exp_val = self.compute_mu_exp(dm)
        if h1e is None:
            h1e = self.cqed_h1e(dm, mu_exp_val)
        return (
            self.get_energy_elec(h1e, F, dm)
            + self.E_nn_
            + self.get_energy_dc(dm, mu_exp_val)
        )

    def kernel(self, max_iter: int = 100, conv_tol: float = 1e-7) -> float:
        """Run the SCF procedure with precomputed integrals."""
        # Precompute all integrals before starting SCF
        self._compute_all_integrals()

        print("Starting SCF calculation...")
        # Initial guess using core Hamiltonian
        D = np.einsum(
            "pi,qi->pq",
            np.asarray(self.wfn_.Ca())[:, : self.ndocc_],
            np.asarray(self.wfn_.Ca())[:, : self.ndocc_],
        )
        E_old = self.psi4_rhf_energy_

        for iter_num in range(max_iter):
            # Build Fock matrix
            mu_exp_val = self.compute_mu_exp(D)
            h1e = self.cqed_h1e(D, mu_exp_val)
            F = self.get_fock(D, h1e, mu_exp_val)
            E_total = self.get_energy_tot(h1e, F, D, mu_exp_val)

            # Get new density matrix and energy
            D_new = self.make_density(F)
            # Check convergence
            E_diff = E_total - E_old
            D_diff = np.mean((D_new - D) ** 2) ** 0.5

            print(
                f"Iter {iter_num:3d}: E = {E_total:.10f}, "
                f"dE = {E_diff:.3e}, dD = {D_diff:.3e}"
            )

            if E_diff < conv_tol and D_diff < conv_tol:
                scf_1e_e = np.einsum("pq,pq->", 2.0 * self.H_, D, optimize=True)
                scf_2e_e = np.einsum("pq,pq->", self.scf_veff(D), D, optimize=True)
                cqed_2e_e = np.einsum("pq,pq->", self.dipole_veff(D), D, optimize=True)
                cqed_d_e = np.einsum(
                    "pq,pq->",
                    self.cqed_dij(D, mu_exp_val),
                    D,
                    optimize=True,
                )
                cqed_dc_e = self.get_energy_dc(D, mu_exp_val)
                cqed_q_e = np.einsum("pq,pq->", 2.0 * self.Qij_, D, optimize=True)

                assert np.isclose(
                    scf_1e_e + scf_2e_e + cqed_2e_e + cqed_d_e + cqed_dc_e + cqed_q_e,
                    E_total - self.E_nn_,
                    atol=1e-10,
                )
                print(f"1E ENERGY: {scf_1e_e:.10f}")
                print(f"2E ENERGY: {scf_2e_e:.10f}")
                print(f"DIPOLE ENERGY: {cqed_2e_e:.10f}")
                print(f"DIPOLE CORRECTION ENERGY: {cqed_d_e:.10f}")
                print(f"DIPOLE CORRECTION ENERGY: {cqed_dc_e:.10f}")
                print(f"Q ENERGY: {cqed_q_e:.10f}")
                print("\nSCF Converged!")
                return E_total

            D = D_new
            E_old = E_total

        raise RuntimeError("SCF did not converge within maximum iterations")


if __name__ == "__main__":
    import psi4

    psi4.set_memory("2 GB")

    h2o_options_dict = {
        "basis": "cc-pVDZ",
        "save_jk": True,
        "scf_type": "pk",
        "e_convergence": 1e-12,
        "d_convergence": 1e-12,
    }
    h2o_string = """

    0 1
        O      0.000000000000   0.000000000000  -0.068516219320
        H      0.000000000000  -0.790689573744   0.543701060715
        H      0.000000000000   0.790689573744   0.543701060715
    no_reorient
    symmetry c1
    """

    lam_h2o = np.array([0.0, 0.0, 0.05])

    cqed_rhf = CQED_RHF(h2o_string, lam_h2o, h2o_options_dict)
    cqed_e = cqed_rhf.kernel()
    ref_e = -76.016355284146

    print(f"Final SCF energy (Psi4): {cqed_rhf.psi4_rhf_energy_:.10f}")
    print(f"Final CQED_RHF energy: {cqed_e:.10f}")
    print(f"Reference energy: {ref_e:.10f}")
    print(f"Energy difference: {cqed_e - ref_e:.10f}")
