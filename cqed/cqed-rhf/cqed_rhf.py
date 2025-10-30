import numpy as np
import scipy.linalg
from pyscf import gto, scf
import numpy.typing as npt


class CQED_RHF:
    def __init__(self, mf: scf.hf.RHF, lambda_vec: npt.NDArray):
        """Initialize RHF calculator with a PySCF Mole object."""
        self.mf_ = mf
        self.lambda_vec_ = lambda_vec
        # Basic parameters
        self.nao_ = self.mf_.mol.nao_nr()
        self.nelec_ = self.mf_.mol.nelec
        self.ndocc_ = min(self.nelec_)

        # Initialize integral matrices
        self.S_ = np.array(self.mf_.get_ovlp())  # Overlap matrix
        self.H_ = np.array(self.mf_.get_hcore())  # Core Hamiltonian
        self.eri_ = self.eri_ = np.array(
            self.mf_.mol.intor("int2e_sph", aosym="s1")
        )  # Electron repulsion integrals

        self.Qij_ = np.zeros((self.nao_, self.nao_))  # Quadrupole integrals
        self.ao_dipole_ = np.zeros((3, self.nao_, self.nao_))  # Dipole integrals

        # Diis parameter
        self.DIIS = False
        self.diis_space_ = 12
        self.diis_start_ = 2
        self.A_ = scipy.linalg.fractional_matrix_power(
            self.S_, -0.5
        )  # Overlap orthogonalization matrix
        self.F_list_ = []
        self.DIIS_list_ = []

        # Nuclear repulsion energy
        self.E_nn_ = self.mf_.energy_nuc()
        self.mu_nuc_val_ = self._compute_mu_nuc()

    def _compute_mu_nuc(self):
        r"""
        #\mu_{nuc} = \sum_A Z_A*R_A
        """
        charges = self.mf_.mol.atom_charges()
        coords = self.mf_.mol.atom_coords()
        return np.einsum("i,ij->j", charges, coords, optimize=True)

    def _compute_all_integrals(self):
        """Precompute all necessary CQED integrals before starting SCF."""
        Q = self.mf_.mol.intor("int1e_rr_sph").reshape(
            3, 3, self.nao_, self.nao_
        )  # FIXME: Maybe wrong order

        # -1/2 * \sum_{\zeta, \zeta'} \lambda^{\zeta} * \lambda^{\zeta'} * Q_{ij}^{\zeta\zeta'}
        self.Qij_ = -0.5 * (
            self.lambda_vec_[0] ** 2 * Q[0, 0]
            + self.lambda_vec_[1] ** 2 * Q[1, 1]
            + self.lambda_vec_[2] ** 2 * Q[2, 2]
            + 2.0 * self.lambda_vec_[0] * self.lambda_vec_[1] * Q[0, 1]
            + 2.0 * self.lambda_vec_[0] * self.lambda_vec_[2] * Q[0, 2]
            + 2.0 * self.lambda_vec_[1] * self.lambda_vec_[2] * Q[1, 2]
        )

        self.ao_dipole_ = self.mf_.mol.intor("int1e_r_sph", comp=3)
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

    def cqed_h1e(self, dm: npt.NDArray, mu_exp_val: float = None):
        r"""
        # H_{ij} = H_{ij}
        #          -1/2 * \sum_{\zeta, \zeta'} \lambda^{\zeta} * \lambda^{\zeta'} * Q_{ij}^{\zeta\zeta'}
        #          + \sum_{\zeta} \lambda_{\zeta} * \mu_{ij}^{\zeta} * (\lambda \dot \mu_{nuc} - \lambda \dot <\mu>)
        """
        if mu_exp_val is None:
            mu_exp_val = self.compute_mu_exp(dm)

        dij = (
            self.l_dot_mu_nuc_
            - np.einsum("x,x->", self.lambda_vec_, mu_exp_val, optimize=True)
        ) * self.l_dot_ao_dipole_
        return self.H_ + self.Qij_ + dij

    def cqed_veff(self, dm: npt.NDArray):
        r"""
        # G_{ij} = \sum_{kl} (2(ij|kl) - (ik|jl)) * D_{kl} +
        # \sum_{\zeta, \zeta'}\lambda^{\zeta} * \lambda^{\zeta'} *
        # (2\mu_{ij}^{\zeta}\mu_{kl}^{\zeta'} - \mu_{ik}^{\zeta}\mu_{jl}^{\zeta'}) * D_{kl}
        """
        return (
            2.0 * np.einsum("ijkl,kl->ij", self.eri_, dm, optimize=True)
            - np.einsum("ikjl,kl->ij", self.eri_, dm, optimize=True)
            + 2.0
            * np.einsum(
                "ij, kl, kl->ij",
                self.l_dot_ao_dipole_,
                self.l_dot_ao_dipole_,
                dm,
                optimize=True,
            )
            - np.einsum(
                "ik, jl, kl->ij",
                self.l_dot_ao_dipole_,
                self.l_dot_ao_dipole_,
                dm,
                optimize=True,
            )
        )

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
        l_dot_mu_exp = np.einsum("x,x", self.lambda_vec_, mu_exp_val, optimize=True)
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

    def _compute_diis_res(self, F: npt.NDArray, D: npt.NDArray) -> npt.NDArray:
        return self.A_ @ (F @ D @ self.S_ - self.S_ @ D @ F) @ self.A_

    def apply_diis(self, F_list: list, DIIS_list: list) -> npt.NDArray:
        """Apply DIIS to update the Fock matrix."""
        B_dim = len(F_list) + 1
        B = np.empty((B_dim, B_dim))
        B[-1, :] = -1
        B[:, -1] = -1
        B[-1, -1] = 0

        for i in range(len(F_list)):
            for j in range(len(F_list)):
                # Compute the inner product of residuals
                B[i, j] = np.einsum(
                    "ij,ij->", DIIS_list[i], DIIS_list[j], optimize=True
                )

        rhs = np.zeros((B_dim))
        rhs[-1] = -1
        coeff = np.linalg.solve(B, rhs)

        # Update the Fock matrix as a linear combination of previous Fock matrices
        F_new = np.einsum("i,ikl->kl", coeff[:-1], F_list)

        return F_new

    def kernel(self, max_iter: int = 100, conv_tol: float = 1e-7) -> float:
        """Run the SCF procedure with precomputed integrals."""
        # Precompute all integrals before starting SCF
        self._compute_all_integrals()

        print("Starting SCF calculation...")
        # Initial guess using core Hamiltonian
        D = self.mf_.make_rdm1() / 2.0  # Initial density matrix
        E_old = self.mf_.energy_tot()

        for iter_num in range(max_iter):
            # Build Fock matrix
            mu_exp_val = self.compute_mu_exp(D)
            h1e = self.cqed_h1e(D, mu_exp_val)
            F = self.get_fock(D, h1e, mu_exp_val)
            E_total = self.get_energy_tot(h1e, F, D, mu_exp_val)

            if self.DIIS:
                diis_res = self._compute_diis_res(F, D)
                self.F_list_.append(F)
                self.DIIS_list_.append(diis_res)

                if len(self.F_list_) > self.diis_space_:
                    self.F_list_.pop(0)
                    self.DIIS_list_.pop(0)

                if iter_num > self.diis_start_:
                    F = self.apply_diis(self.F_list_, self.DIIS_list_)

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
                print("\nSCF Converged!")
                print(f"Final SCF energy: {E_total:.10f}")
                return E_total

            D = D_new
            E_old = E_total

        raise RuntimeError("SCF did not converge within maximum iterations")


def main():
    # Example usage
    mol = gto.M(
        atom="""
    O      0.000000000000   0.000000000000  -0.068516219320
    H      0.000000000000  -0.790689573744   0.543701060715
    H      0.000000000000   0.790689573744   0.543701060715
        """,
        basis="ccpvdz",
        max_memory=10000,
    )

    # Compare with PySCF
    mf_pyscf = scf.RHF(mol)
    E_pyscf = mf_pyscf.kernel()
    # print(f"\nPySCF energy: {E_pyscf:.10f}")

    # Our implementation
    mf = CQED_RHF(mf_pyscf, lambda_vec=np.array([0.0, 0.0, 0.05]))
    E_our = mf.kernel()
    print(f"Our energy:   {E_our:.10f}")
    print(f"reference energy (PySCF): {-76.016355284146}")

    print(f"Difference:   {abs(-76.016355284146 - E_our):.10f}")


if __name__ == "__main__":
    main()
