import numpy as np
import scipy.linalg
from pyscf import gto, scf
import numpy.typing as npt


class UHF:
    def __init__(self, mol: gto.Mole):
        """Initialize RHF calculator with a PySCF Mole object."""
        self.mol = mol
        # Basic parameters
        self.atm = mol._atm
        self.bas = mol._bas
        self.env = mol._env
        self.nao = mol.nao_nr()
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
        self.H = self.T + self.V
        self.A = scipy.linalg.fractional_matrix_power(self.S, -0.5)
        print("Integral computation completed.")

    def get_fock(
        self, D: tuple[npt.NDArray, npt.NDArray]
    ) -> tuple[npt.NDArray, npt.NDArray]:
        """Build Fock matrices from density matrices using precomputed integrals."""
        Da, Db = D
        Fa = (
            self.H
            + np.einsum("pqrs, rs -> pq", self.eri, Da)
            + np.einsum("pqrs, rs -> pq", self.eri, Db)
            - np.einsum("prqs, rs -> pq", self.eri, Da)
        )
        Fb = (
            self.H
            + np.einsum("pqrs, rs -> pq", self.eri, Da)
            + np.einsum("pqrs, rs -> pq", self.eri, Db)
            - np.einsum("prqs, rs -> pq", self.eri, Db)
        )
        return Fa, Fb

    def make_density(
        self, fock: tuple[npt.NDArray, npt.NDArray]
    ) -> tuple[npt.NDArray, npt.NDArray]:
        """Create new density matrices for alpha and beta components."""
        Fa, Fb = fock

        # Solve eigenvalue problems separately for alpha and beta
        _, Ca = scipy.linalg.eigh(Fa, self.S)
        _, Cb = scipy.linalg.eigh(Fb, self.S)

        # Form density matrices
        Ca_occ = Ca[:, : self.nalpha]
        Cb_occ = Cb[:, : self.nbeta]

        Da = np.einsum("pi,qi->pq", Ca_occ, Ca_occ, optimize=True)
        Db = np.einsum("pi,qi->pq", Cb_occ, Cb_occ, optimize=True)

        return Da, Db

    def get_energy_elec(
        self, F: tuple[npt.NDArray, npt.NDArray], D: tuple[npt.NDArray, npt.NDArray]
    ) -> float:
        """Calculate electronic energy."""
        Fa, Fb = F
        Da, Db = D
        return 0.5 * (
            np.einsum("pq, pq ->", (Da + Db), self.H, optimize=True)
            + np.einsum("pq, pq ->", Da, Fa, optimize=True)
            + np.einsum("pq, pq ->", Db, Fb, optimize=True)
        )

    def get_energy_tot(
        self, F: tuple[npt.NDArray, npt.NDArray], D: tuple[npt.NDArray, npt.NDArray]
    ) -> float:
        """Calculate total energy."""
        return self.get_energy_elec(F, D) + self.E_nn

    def _compute_diis_res(
        self, F: tuple[npt.NDArray, npt.NDArray], D: tuple[npt.NDArray, npt.NDArray]
    ) -> tuple[npt.NDArray, npt.NDArray]:
        Fa, Fb = F
        Da, Db = D
        res_a = self.A @ (Fa @ Da @ self.S - self.S @ Da @ Fa) @ self.A
        res_b = self.A @ (Fb @ Db @ self.S - self.S @ Db @ Fb) @ self.A
        return res_a, res_b

    def apply_diis(
        self,
        F_list: list[tuple[npt.NDArray, npt.NDArray]],
        DIIS_list: list[tuple[npt.NDArray, npt.NDArray]],
    ) -> tuple[npt.NDArray, npt.NDArray]:
        """Apply DIIS to update the Fock matrix."""
        B_dim = len(F_list) + 1
        Ba = np.zeros((B_dim, B_dim))
        Bb = np.zeros((B_dim, B_dim))
        Ba[-1, :], Bb[-1, :] = -1, -1
        Ba[:, -1], Bb[:, -1] = -1, -1
        Ba[-1, -1], Bb[-1, -1] = 0, 0

        for i in range(len(F_list)):
            for j in range(len(F_list)):
                # Compute the inner product of residuals
                Ba[i, j] = np.einsum(
                    "ij,ij->", DIIS_list[i][0], DIIS_list[j][0], optimize=True
                )
                Bb[i, j] = np.einsum(
                    "ij,ij->", DIIS_list[i][1], DIIS_list[j][1], optimize=True
                )

        rhs = np.zeros((B_dim))
        rhs[-1] = -1
        coeff_a = np.linalg.solve(Ba, rhs)
        coeff_b = np.linalg.solve(Bb, rhs)

        # Update the Fock matrix as a linear combination of previous Fock matrices
        Fa_new = np.einsum("i,ikl->kl", coeff_a[:-1], [f[0] for f in F_list])
        Fb_new = np.einsum("i,ikl->kl", coeff_b[:-1], [f[1] for f in F_list])

        return Fa_new, Fb_new

    def kernel(self, max_iter: int = 100, conv_tol: float = 1e-6) -> float:
        """Run the SCF procedure with precomputed integrals."""
        # Precompute all integrals before starting SCF
        self._compute_all_integrals()

        print("Starting SCF calculation...")
        # Initial guess using core Hamiltonian
        D = self.make_density((self.H, self.H))
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
            D_diff = (
                np.mean((D_new[0] - D[0]) ** 2) ** 0.5
                + np.mean((D_new[1] - D[1]) ** 2) ** 0.5
            )

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
    mol = gto.M(atom="O 0 0 0; O 0 0 1.2", basis="ccpvdz", spin=2)
    # mol = gto.M(
    #    atom="Co 0 0 0; Cl 2.2 0 0; Cl -2.2 0 0; Cl 0 2.2 0; Cl 0 -2.2 0",
    #    charge=-2,
    #    basis="ccpvdz",
    #    spin=3,
    # )
    #
    # Compare with PySCF
    mf_pyscf = scf.UHF(mol)
    mf_pyscf.init_guess = "1e"
    E_pyscf = mf_pyscf.kernel()
    print(f"\nPySCF energy: {E_pyscf:.10f}")

    # Our implementation
    mf = UHF(mol)
    mf.DIIS = False
    E_our = mf.kernel()
    print(f"Our energy:   {E_our:.10f}")
    print(f"Difference:   {abs(E_pyscf - E_our):.10f}")


if __name__ == "__main__":
    main()
