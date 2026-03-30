import numpy as np
import scipy
from pyscf import gto, scf
import numpy.typing as npt
import scipy.linalg


LIGHT_SPEED = 137.03599967994


class DHF:
    def __init__(self, mol: gto.Mole):
        """Initialize RHF calculator with a PySCF Mole object."""
        self.mol = mol
        # Basic parameters
        self.atm = mol._atm
        self.bas = mol._bas
        self.env = mol._env
        self.nao = mol.nao_nr()
        self.n2c = mol.nao_2c()
        self.n4c = self.n2c * 2
        self.nshls = len(self.bas)
        self.natm = len(self.atm)
        self.nelec = sum(mol.nelec)
        self.ndocc = min(mol.nelec)

        # Initialize integral matrices
        self.S = None  # Overlap matrix
        self.H = None  # Core Hamiltonian
        # Dirac Coulumb integral
        self._coulomb_level = "LLLL"
        self.LLLL = None
        self.SSLL = None
        self.with_ssss = True
        self.SSSS = None
        # Gaunt Breit integral

        self.with_gaunt = False
        self.with_breit = False
        self.LSLS = None
        self.SLSL = None
        self.LSSL = None
        self.SLLS = None

        self.B_LSSL = None
        self.B_LSLS = None

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

        c = LIGHT_SPEED
        n2c = self.n2c
        n4c = self.n4c
        ao_loc = self.mol.ao_loc_2c()

        t = np.zeros((n2c, n2c), np.complex128)
        vn = np.zeros((n2c, n2c), np.complex128)
        wn = np.zeros((n2c, n2c), np.complex128)
        s = np.zeros((n2c, n2c), np.complex128)

        # Compute one-electron integrals
        for i in range(self.nshls):
            i0, i1 = ao_loc[i], ao_loc[i + 1]
            for j in range(i, self.nshls):
                j0, j1 = ao_loc[j], ao_loc[j + 1]

                # Compute integrals
                buf_s = self.mol.intor(
                    "int1e_ovlp_spinor", shls_slice=(i, i + 1, j, j + 1)
                )
                buf_vn = self.mol.intor(
                    "int1e_nuc_spinor", shls_slice=(i, i + 1, j, j + 1)
                )
                buf_wn = self.mol.intor(
                    "int1e_spnucsp_spinor", shls_slice=(i, i + 1, j, j + 1)
                )
                buf_t = self.mol.intor(
                    "int1e_spsp_spinor", shls_slice=(i, i + 1, j, j + 1)
                )
                # Store results
                t[i0:i1, j0:j1] = buf_t
                vn[i0:i1, j0:j1] = buf_vn
                wn[i0:i1, j0:j1] = buf_wn
                s[i0:i1, j0:j1] = buf_s

                t[j0:j1, i0:i1] = buf_t.conj().T
                vn[j0:j1, i0:i1] = buf_vn.conj().T
                wn[j0:j1, i0:i1] = buf_wn.conj().T
                s[j0:j1, i0:i1] = buf_s.conj().T
        self.H = np.zeros((n4c, n4c), np.complex128)
        self.H[:n2c, :n2c] = vn
        self.H[n2c:, :n2c] = t * 0.5
        self.H[:n2c, n2c:] = t * 0.5
        self.H[n2c:, n2c:] = wn * (0.25 / c**2) - t * 0.5

        self.S = np.zeros((n4c, n4c), np.complex128)
        self.S[:n2c, :n2c] = s
        self.S[n2c:, n2c:] = t * (0.25 / c**2)

        # Compute two-electron integrals
        print("Computing ERI integrals...")
        self.LLLL = self.mol.intor("int2e_spinor").reshape(n2c, n2c, n2c, n2c)
        self.SSLL = self.mol.intor("int2e_spsp1_spinor").reshape(n2c, n2c, n2c, n2c)

        if self.with_ssss is True:
            self.SSSS = self.mol.intor("int2e_spsp1spsp2_spinor").reshape(
                n2c, n2c, n2c, n2c
            )
        if self.with_gaunt is True and self.with_breit is False:
            self.LSLS = self.mol.intor("int2e_ssp1ssp2_spinor").reshape(
                n2c, n2c, n2c, n2c
            )
            self.LSSL = self.mol.intor("int2e_ssp1sps2_spinor").reshape(
                n2c, n2c, n2c, n2c
            )
        if self.with_breit is True:
            self.B_LSLS = self.mol.intor("int2e_breit_ssp1ssp2_spinor", comp=1).reshape(
                n2c, n2c, n2c, n2c
            )
            self.B_LSSL = self.mol.intor("int2e_breit_ssp1sps2_spinor", comp=1).reshape(
                n2c, n2c, n2c, n2c
            )
        print("Integral computation completed.")

    def build_init_guess(self):
        return self.make_density(self.H)

    def _call_veff_LLLL(self, D: npt.NDArray):
        n2c = self.n2c
        vj = np.zeros((n2c * 2, n2c * 2), dtype=np.complex128)
        vk = np.zeros((n2c * 2, n2c * 2), dtype=np.complex128)

        dms = D[:n2c, :n2c].copy()
        J = np.einsum("ijkl,lk->ij", self.LLLL, dms, optimize=True)
        K = np.einsum("ilkj,lk->ij", self.LLLL, dms, optimize=True)

        vj[:n2c, :n2c] = J
        vk[:n2c, :n2c] = K

        return vj, vk

    def _call_veff_SSLL(self, D: npt.NDArray):
        n2c = self.n2c
        c1 = 0.5 / LIGHT_SPEED
        vj = np.zeros((n2c * 2, n2c * 2), dtype=np.complex128)
        vk = np.zeros((n2c * 2, n2c * 2), dtype=np.complex128)

        dm = D.copy()
        dmll = dm[:n2c, :n2c].copy()
        dmss = dm[n2c:, n2c:].copy()
        dmsl = dm[n2c:, :n2c].copy()
        dmls = dm[:n2c, n2c:].copy()

        J1 = np.einsum("ijkl,lk->ij", self.SSLL, dmll, optimize=True) * c1**2
        J2 = np.einsum("klij,lk->ij", self.SSLL, dmss, optimize=True) * c1**2
        K1 = np.einsum("ilkj,lk->ij", self.SSLL, dmsl, optimize=True) * c1**2
        # K2 = np.einsum("kjil,lk->ij", self.SSLL, dmls, optimize=True) * c1**2
        K2 = K1.transpose().conj()

        vj[n2c:, n2c:] = J1
        vj[:n2c, :n2c] = J2
        vk[n2c:, :n2c] = K1
        vk[:n2c, n2c:] = K2

        return vj, vk

    def _call_veff_SSSS(self, D: npt.NDArray):
        n2c = self.n2c
        c1 = 0.5 / LIGHT_SPEED

        vj = np.zeros((n2c * 2, n2c * 2), dtype=np.complex128)
        vk = np.zeros((n2c * 2, n2c * 2), dtype=np.complex128)

        dms = D[n2c:, n2c:].copy()
        J = np.einsum("ijkl,lk->ij", self.SSSS, dms, optimize=True) * c1**4
        K = np.einsum("ilkj,lk->ij", self.SSSS, dms, optimize=True) * c1**4

        vj[n2c:, n2c:] += J
        vk[n2c:, n2c:] += K

        return vj, vk

    def _call_veff_gaunt_breit(self, D: npt.NDArray):
        n2c = self.n2c
        c1 = 0.5 / LIGHT_SPEED
        vj = np.zeros((n2c * 2, n2c * 2), dtype=np.complex128)
        vk = np.zeros((n2c * 2, n2c * 2), dtype=np.complex128)

        dm = D.copy()
        dmll = dm[:n2c, :n2c].copy()
        dmss = dm[n2c:, n2c:].copy()
        dmsl = dm[n2c:, :n2c].copy()
        dmls = dm[:n2c, n2c:].copy()

        if self.with_breit:
            Kll = np.einsum("ilkj,lk->ij", self.B_LSSL, dmss, optimize=True) * c1**2
            Kss = np.einsum("kjil,lk->ij", self.B_LSSL, dmll, optimize=True) * c1**2
            Kls = np.einsum("ilkj,lk->ij", self.B_LSLS, dmsl, optimize=True) * c1**2
            Ksl = Kls.transpose().conj()

            Jls = (
                np.einsum("ijkl,lk->ij", self.B_LSSL, dmls, optimize=True)
                + np.einsum("ijkl,lk->ij", self.B_LSLS, dmsl, optimize=True)
            ) * c1**2
            Jsl = Jls.transpose().conj()

            vk[:n2c, :n2c] = Kll
            vk[n2c:, n2c:] = Kss
            vk[:n2c, n2c:] = Kls
            vk[n2c:, :n2c] = Ksl

            vj[:n2c, n2c:] = Jls
            vj[n2c:, :n2c] = Jsl

            return -vj, -vk
        else:
            Kll = np.einsum("ilkj,lk->ij", self.LSSL, dmss, optimize=True) * c1**2
            Kss = np.einsum("kjil,lk->ij", self.LSSL, dmll, optimize=True) * c1**2
            Kls = np.einsum("ilkj,lk->ij", self.LSLS, dmsl, optimize=True) * c1**2
            Ksl = Kls.transpose().conj()

            Jls = (
                np.einsum("ijkl,lk->ij", self.LSSL, dmls, optimize=True)
                + np.einsum("ijkl,lk->ij", self.LSLS, dmsl, optimize=True)
            ) * c1**2
            Jsl = Jls.transpose().conj()

            vk[:n2c, :n2c] = Kll
            vk[n2c:, n2c:] = Kss
            vk[:n2c, n2c:] = Kls
            vk[n2c:, :n2c] = Ksl

            vj[:n2c, n2c:] = Jls
            vj[n2c:, :n2c] = Jsl

            return vj, vk

    def get_vhf(self, D: npt.NDArray) -> npt.NDArray:
        """Build Fock matrix from density matrix using precomputed integrals."""

        coulomb_level = self._coulomb_level

        if coulomb_level.upper() == "LLLL":
            vj, vk = self._call_veff_LLLL(D)

        elif coulomb_level.upper() == "SSLL" or coulomb_level.upper() == "LLSS":
            vj, vk = self._call_veff_SSLL(D)
            J, K = self._call_veff_LLLL(D)
            vj += J
            vk += K

            if self.with_gaunt or self.with_breit:
                J, K = self._call_veff_gaunt_breit(D)
                vj -= J
                vk -= K

        else:  # SSSS
            vj, vk = self._call_veff_SSLL(D)
            J, K = self._call_veff_LLLL(D)
            vj += J
            vk += K

            J, K = self._call_veff_SSSS(D)

            vj += J
            vk += K

            if self.with_gaunt or self.with_breit:
                J, K = self._call_veff_gaunt_breit(D)
                vj -= J
                vk -= K

        return vj - vk

    def make_density(self, fock: npt.NDArray) -> npt.NDArray:
        """Create new density matrix and calculate electronic energy."""
        # Solve eigenvalue problem
        mo_energy, mo_coeff = scipy.linalg.eigh(fock, self.S)

        n4c = self.n4c
        n2c = self.n2c

        mo_occ = np.zeros(n4c)
        c = LIGHT_SPEED
        collapse_threshold = -1.999 * c**2

        if mo_energy[n2c] > collapse_threshold:
            # Normal case - fill lowest energy orbitals
            mo_occ[n2c : n2c + self.nelec] = 1
        else:
            # Handle variational collapse
            valid_energies = mo_energy > collapse_threshold
            lumo = mo_energy[valid_energies][self.nelec]
            mo_occ[valid_energies] = 1
            mo_occ[mo_energy >= lumo] = 0

        # Form density matrix
        C_occ = mo_coeff[:, mo_occ > 0]

        return (C_occ * mo_occ[mo_occ > 0]).dot(C_occ.conj().T)

    def get_energy_elec(self, V: npt.NDArray, D: npt.NDArray) -> float:
        e1 = np.einsum("pq,qp->", self.H, D, optimize=True).real
        e_col = np.einsum("pq,qp->", V, D, optimize=True).real * 0.5
        return e1 + e_col

    def get_energy_tot(self, V: npt.NDArray, D: npt.NDArray) -> float:
        return self.get_energy_elec(V, D) + self.E_nn

    def scf(
        self, max_iter: int = 100, conv_tol: float = 1e-6, D: npt.NDArray = None
    ) -> npt.NDArray:
        if D is None:
            D = self.build_init_guess()

        E_old = 0.0
        print(f"Starting {self._coulomb_level}")

        for iter_num in range(max_iter):
            vhf = self.get_vhf(D)
            E_total = self.get_energy_tot(vhf, D)

            D_new = self.make_density(self.H + vhf)

            E_diff = E_total - E_old
            D_diff = np.mean((D_new - D).real ** 2) ** 0.5

            print(
                f"Iter {iter_num:3d}: E = {E_total:.10f}, "
                f"dE = {E_diff:.3e}, dD = {D_diff:.3e}"
            )

            if abs(E_diff) < conv_tol and D_diff < conv_tol:
                print("\nSCF Converged!")
                print(f"Final SCF energy: {E_total:.10f}")
                return D_new, E_total
            D = D_new
            E_old = E_total

        raise RuntimeError("SCF did not converge within maximum iterations")

    def kernel(self, max_iter: int = 100, conv_tol: float = 1e-6) -> float:
        """Run the SCF procedure with precomputed integrals."""
        # Precompute all integrals before starting SCF
        self._compute_all_integrals()
        self._coulomb_level = "LLLL"
        dm, e = self.scf(max_iter=100, conv_tol=1e-3, D=None)
        if self.with_ssss is True:
            self._coulomb_level = "SSLL"
            dm, e = self.scf(max_iter=100, conv_tol=1e-4, D=dm)

            self._coulomb_level = "SSSS"
            dm, e = self.scf(max_iter=100, conv_tol=conv_tol, D=dm)
        else:
            self._coulomb_level = "SSLL"
            dm, e = self.scf(max_iter=100, conv_tol=conv_tol, D=dm)
        return e


def main():
    # Example usage
    mol = gto.M(atom="O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587", basis="sto-3g")
    """
    mol.basis = {
        "H": [
            [0, 0, [8.29687389e01, 1.0]],
            [0, 0, [1.24571508e01, 1.0]],
            [0, 0, [2.83382422e00, 1.0]],
            [0, 0, [8.00164139e-01, 1.0]],
            [0, 0, [2.58629441e-01, 1.0]],
            [0, 0, [8.99766770e-02, 1.0]],
            [1, 0, [5.02448897e-01, 1.0]],
        ],
        "O": [
            [0, 0, [1.59925081e04, 1.0]],
            [0, 0, [2.35199842e03, 1.0]],
            [0, 0, [5.29060079e02, 1.0]],
            [0, 0, [1.48462548e02, 1.0]],
            [0, 0, [4.78417313e01, 1.0]],
            [0, 0, [1.68559662e01, 1.0]],
            [0, 0, [6.23711537e00, 1.0]],
            [0, 0, [1.75808802e00, 1.0]],
            [0, 0, [6.90688830e-01, 1.0]],
            [0, 0, [2.39078229e-01, 1.0]],
            [1, 0, [6.33409745e01, 1.0]],
            [1, 0, [1.45830479e01, 1.0]],
            [1, 0, [4.42797374e00, 1.0]],
            [1, 0, [1.51782600e00, 1.0]],
            [1, 0, [5.24175730e-01, 1.0]],
            [1, 0, [1.72133105e-01, 1.0]],
            [2, 0, [1.17766132e00, 1.0]],
        ],
    }
    """

    mf = DHF(mol)
    mf.with_gaunt = True
    mf.with_breit = True
    E_our = mf.kernel()

    # ref = mol.intor("int2e_breit_ssp1ssp2_spinor").reshape(
    #     mf.n2c, mf.n2c, mf.n2c, mf.n2c
    # )
    # print(ref)
    # print(np.abs(ref - mf.B2_LSSL).max())
    # Compare with PySCF
    mf_pyscf = scf.DHF(mol)

    # mf_pyscf.verbose = 5
    mf_pyscf.init_guess = "1e"
    mf_pyscf.with_ssss = True
    mf_pyscf.with_breit = True
    E_pyscf = mf_pyscf.kernel()

    # print("E(Dirac-Coulomb) = %.15g" % mf_pyscf.kernel())

    # mf_pyscf.with_gaunt = True
    # print("E(Dirac-Coulomb-Gaunt) = %.15g" % mf_pyscf.kernel())

    #
    # mf_pyscf.with_breit = True
    # print("E(Dirac-Coulomb-Breit) = %.15g" % mf_pyscf.kernel())

    #
    # Our implementation
    # E_our = mf.kernel()
    print(f"\n\nOur energy:   {E_our:.10f}")
    print(f"PySCF energy: {E_pyscf:.10f}")
    print(f"Difference:   {abs(E_pyscf - E_our):.10f}")


if __name__ == "__main__":
    main()
