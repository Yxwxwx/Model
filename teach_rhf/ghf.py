import numpy as np
import scipy.linalg
from pyscf import gto, scf
import numpy.typing as npt


class GHF:
    def __init__(self, mol: gto.Mole):
        """Initialize RHF calculator with a PySCF Mole object."""
        self.mol = mol
        # Basic parameters
        self.atm = mol._atm
        self.bas = mol._bas
        self.env = mol._env
        self.nao = mol.nao_nr()
        self.nso = 2 * self.nao
        self.nshls = len(self.bas)
        self.natm = len(self.atm)
        self.nalpha = mol.nelec[0]
        self.nbeta = mol.nelec[1]
        self.nelec = self.nalpha + self.nbeta
        self.ndocc = min(self.nalpha, self.nbeta)
        self.nsocc = abs(self.nalpha - self.nbeta)

        # Initialize integral matrices
        self.S = None  # Overlap matrix
        self.T = None  # Kinetic matrix
        self.V = None  # Nuclear attraction matrix
        self.H = None  # Core Hamiltonian
        self.eri = None  # Electron repulsion integrals

        # Diis parameter
        self.DIIS = True
        self.diis_space = 12
        self.diis_start = 2
        self.A = None  # Overlap orthogonalization matrix
        self.F_list = []
        self.DIIS_list = []

        # Nuclear repulsion energy
        self.E_nn = self._compute_nuclear_repulsion()

    def _compute_nuclear_repulsion(self) -> float:
        """Calculate nuclear repulsion energy."""
        coords = self.mol.atom_coords()
        charges = self.mol.atom_charges()
        natm = len(charges)
        E_nn = 0.0
        for i in range(natm):
            for j in range(i + 1, natm):
                r_ij = np.linalg.norm(coords[i] - coords[j])
                E_nn += charges[i] * charges[j] / r_ij
        return E_nn

    def _compute_all_integrals(self):
        """Precompute all necessary integrals."""
        print("Precomputing integrals...")

        ao_loc = self.mol.ao_loc_nr()
        # Initialize matrices
        self.S = np.zeros((self.nao, self.nao))
        self.T = np.zeros((self.nao, self.nao))
        self.V = np.zeros((self.nao, self.nao))
        self.eri = np.zeros((self.nao,) * 4)

        # Compute one-electron integrals
        for i in range(self.nshls):
            i0, i1 = ao_loc[i], ao_loc[i + 1]
            for j in range(i, self.nshls):
                j0, j1 = ao_loc[j], ao_loc[j + 1]
                # Compute integrals
                buf_s = self.mol.intor(
                    "int1e_ovlp_sph", shls_slice=(i, i + 1, j, j + 1)
                )
                buf_t = self.mol.intor("int1e_kin_sph", shls_slice=(i, i + 1, j, j + 1))
                buf_v = self.mol.intor("int1e_nuc_sph", shls_slice=(i, i + 1, j, j + 1))
                # Store results
                self.S[i0:i1, j0:j1] = buf_s
                self.T[i0:i1, j0:j1] = buf_t
                self.V[i0:i1, j0:j1] = buf_v

                self.S[j0:j1, i0:i1] = buf_s.T
                self.T[j0:j1, i0:i1] = buf_t.T
                self.V[j0:j1, i0:i1] = buf_v.T

        # Compute two-electron integrals
        print("Computing ERI integrals...")
        for i in range(self.nshls):
            i0, i1 = ao_loc[i], ao_loc[i + 1]
            for j in range(i, self.nshls):
                j0, j1 = ao_loc[j], ao_loc[j + 1]
                for k in range(self.nshls):
                    k0, k1 = ao_loc[k], ao_loc[k + 1]
                    for l in range(k, self.nshls):  # noqa: E741
                        l0, l1 = ao_loc[l], ao_loc[l + 1]

                        buf = self.mol.intor(
                            "int2e_sph",
                            shls_slice=(i, i + 1, j, j + 1, k, k + 1, l, l + 1),
                        )
                        self.eri[i0:i1, j0:j1, k0:k1, l0:l1] = buf.transpose(0, 1, 2, 3)
                        self.eri[j0:j1, i0:i1, k0:k1, l0:l1] = buf.transpose(1, 0, 2, 3)
                        self.eri[i0:i1, j0:j1, l0:l1, k0:k1] = buf.transpose(0, 1, 3, 2)
                        self.eri[j0:j1, i0:i1, l0:l1, k0:k1] = buf.transpose(1, 0, 3, 2)
                        self.eri[k0:k1, l0:l1, i0:i1, j0:j1] = buf.transpose(2, 3, 0, 1)
                        self.eri[l0:l1, k0:k1, i0:i1, j0:j1] = buf.transpose(3, 2, 0, 1)
                        self.eri[k0:k1, l0:l1, j0:j1, i0:i1] = buf.transpose(2, 3, 1, 0)
                        self.eri[l0:l1, k0:k1, j0:j1, i0:i1] = buf.transpose(3, 2, 1, 0)

        # Compute core Hamiltonian and orthogonalization matrix
        self.S = scipy.linalg.block_diag(self.S, self.S)
        self.H = self.T + self.V
        self.H = scipy.linalg.block_diag(self.H, self.H)
        self.A = scipy.linalg.fractional_matrix_power(self.S, -0.5)
        print("Integral computation completed.")

    def build_init_guess(self) -> npt.NDArray:
        _, C = scipy.linalg.eigh(self.H, self.S)
        C_occ = C[:, : self.nelec]
        return np.einsum("pi,qi->pq", C_occ, C_occ, optimize=True) + 0.05

    def make_density(self, fock: npt.NDArray) -> npt.NDArray:
        # Solve eigenvalue problem
        _, C = scipy.linalg.eigh(fock, self.S)
        C_occ = C[:, : self.nelec]
        return np.einsum("pi,qi->pq", C_occ, C_occ, optimize=True)

    def get_fock(self, D: npt.NDArray) -> npt.NDArray:
        nao = self.nao
        dmaa = D[:nao, :nao]
        dmab = D[:nao, nao:]
        dmbb = D[nao:, nao:]

        J = np.einsum("rs, pqrs -> pq", dmaa, self.eri, optimize=True) + np.einsum(
            "rs, pqrs -> pq", dmbb, self.eri, optimize=True
        )
        Kaa = np.einsum("rs, prqs -> pq", dmaa, self.eri, optimize=True)
        Kbb = np.einsum("rs, prqs -> pq", dmbb, self.eri, optimize=True)
        Kab = np.einsum("rs, prqs -> pq", dmab, self.eri, optimize=True)
        return np.block([[J - Kaa, -Kab], [-Kab, J - Kbb]]) + self.H

    def get_energy_elec(self, F: npt.NDArray, D: npt.NDArray) -> float:
        return np.einsum("pq,pq->", (self.H + F), D, optimize=True) / 2

    def get_energy_tot(self, F: npt.NDArray, D: npt.NDArray) -> float:
        return self.get_energy_elec(F, D) + self.E_nn

    def _compute_diis_res(self, F: npt.NDArray, D: npt.NDArray) -> npt.NDArray:
        return self.A @ (F @ D @ self.S - self.S @ D @ F) @ self.A

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

    def kernel(self, max_iter: int = 1000, conv_tol: float = 1e-6):
        """Run the SCF procedure with precomputed integrals."""
        # Precompute all integrals before starting SCF
        self._compute_all_integrals()

        print("Starting SCF calculation...")
        # Initial guess using core Hamiltonian
        D = self.build_init_guess()
        E_old = 0.0

        for iter_num in range(max_iter):
            # Build Fock matrix
            F = self.get_fock(D)
            E_total = self.get_energy_tot(F, D)

            if self.DIIS:
                diis_res = self._compute_diis_res(F, D)
                self.F_list.append(F)
                self.DIIS_list.append(diis_res)

                if len(self.F_list) > self.diis_space:
                    self.F_list.pop(0)
                    self.DIIS_list.pop(0)

                if iter_num > self.diis_start:
                    F = self.apply_diis(self.F_list, self.DIIS_list)

            # Get new density matrix and energy
            D_new = self.make_density(F)

            # Check convergence
            E_diff = abs(E_total - E_old)
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

        print("\nSCF Converged!")
        print(f"Final SCF energy: {E_old:.10f}")
        raise RuntimeError("SCF did not converge within maximum iterations")


def main():
    # Example usage
    mol = gto.M(atom="O 0 0 0; O 0 0 1.2", basis="ccpvdz", spin=2)

    # Compare with PySCF
    mf_pyscf = scf.GHF(mol)
    E_pyscf = mf_pyscf.kernel()
    print(f"\nPySCF energy: {E_pyscf:.10f}")

    # Our implementation
    mf = GHF(mol)
    E_our = mf.kernel()
    print(f"Our energy:   {E_our:.10f}")
    print(f"Difference:   {abs(E_pyscf - E_our):.10f}")


if __name__ == "__main__":
    main()
