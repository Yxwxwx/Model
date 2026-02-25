import numpy as np
import scipy.linalg
import numpy.typing as npt
from pyscf import gto, scf


class CQED_RHF:
    def __init__(self, molecule: gto.Mole, lambda_vec: npt.NDArray):
        """Initialize RHF calculator with a PySCF Mole object."""
        self.mol_ = molecule
        self.lambda_vec_ = lambda_vec
        mf = scf.RHF(self.mol_).run()
        self.pyscf_rhf_energy_, self.wfn_ = mf.e_tot, mf.mo_coeff

        # Basic parameters
        self.ndocc_ = self.mol_.nelectron // 2
        self.nao_ = self.wfn_.shape[0]

        # Initialize integral matrices
        self.S_ = self.mol_.intor("int1e_ovlp")
        self.H_ = self.mol_.intor("int1e_kin") + self.mol_.intor("int1e_nuc")

        self.eri_ = self.mol_.intor("int2e", aosym="s1")  # Electron repulsion integrals

        self.Qij_ = np.zeros((self.nao_, self.nao_))  # Quadrupole integrals
        self.ao_dipole_ = np.zeros((3, self.nao_, self.nao_))  # Dipole integrals

        # Nuclear repulsion energy
        self.E_nn_ = self.mol_.energy_nuc()
        self.mu_nuc_val_ = np.einsum(
            "i,ix->x", self.mol_.atom_charges(), self.mol_.atom_coords(), optimize=True
        )
        print(f"Nuclear dipole moment: {self.mu_nuc_val_}")

    def _compute_all_integrals(self):
        """Precompute all necessary CQED integrals before starting SCF."""
        Q_ao = -1.0 * self.mol_.intor("int1e_rr").reshape(3, 3, self.nao_, self.nao_)

        Q_ao_xx = Q_ao[0, 0]
        Q_ao_xy = Q_ao[0, 1]
        Q_ao_xz = Q_ao[0, 2]
        Q_ao_yy = Q_ao[1, 1]
        Q_ao_yz = Q_ao[1, 2]
        Q_ao_zz = Q_ao[2, 2]

        # -1/2 * \sum_{\zeta, \zeta'} \lambda^{\zeta} * \lambda^{\zeta'} * Q_{ij}^{\zeta\zeta'}
        self.Qij_ = -0.5 * (
            self.lambda_vec_[0] ** 2 * Q_ao_xx
            + self.lambda_vec_[1] ** 2 * Q_ao_yy
            + self.lambda_vec_[2] ** 2 * Q_ao_zz
            + 2.0 * self.lambda_vec_[0] * self.lambda_vec_[1] * Q_ao_xy
            + 2.0 * self.lambda_vec_[0] * self.lambda_vec_[2] * Q_ao_xz
            + 2.0 * self.lambda_vec_[1] * self.lambda_vec_[2] * Q_ao_yz
        )

        self.ao_dipole_ = -1.0 * self.mol_.intor("int1e_r", comp=3)
        self.l_dot_mu_nuc_ = np.einsum(
            "x,x->", self.lambda_vec_, self.mu_nuc_val_, optimize=True
        )
        self.l_dot_ao_dipole_ = np.einsum(
            "x, xij->ij", self.lambda_vec_, self.ao_dipole_, optimize=True
        )

    def compute_mu_e_exp(self, dm: npt.NDArray):
        r"""
        # <\mu> = 2.0 * \sum_i^N_{occ} <i|\mu|i>
        """
        return 2.0 * np.einsum("xij,ji->x", self.ao_dipole_, dm, optimize=True)

    def compute_mu_exp(self, dm: npt.NDArray):
        r"""
        # <\mu> = 2.0 * \sum_i^N_{occ} <i|\mu|i> + \mu_{nuc}
        """
        return self.mu_nuc_val_ + self.compute_mu_e_exp(dm)

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

    def get_coeffs(self, fock: npt.NDArray) -> npt.NDArray:
        """Solve eigenvalue problem to get molecular orbital coefficients."""
        _, C = scipy.linalg.eigh(fock, self.S_)
        return C

    def make_density(self, mo_coeffs: npt.NDArray) -> npt.NDArray:
        """Create new density matrix and calculate electronic energy."""
        # Form density matrix
        C_occ = mo_coeffs[:, : self.ndocc_]
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
        h1e: npt.NDArray = None,
        F: npt.NDArray = None,
        dm: npt.NDArray = None,
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

    def kernel(
        self, max_iter: int = 100, conv_tol: float = 1e-7
    ) -> tuple[float, npt.NDArray]:
        """Run the SCF procedure with precomputed integrals."""
        # Precompute all integrals before starting SCF
        self._compute_all_integrals()

        print("Starting SCF calculation...")
        # Initial guess using core Hamiltonian
        D = np.einsum(
            "pi,qi->pq",
            self.wfn_[:, : self.ndocc_],
            self.wfn_[:, : self.ndocc_],
        )
        E_old = self.pyscf_rhf_energy_

        for iter_num in range(max_iter):
            # Build Fock matrix
            mu_exp_val = self.compute_mu_exp(D)
            h1e = self.cqed_h1e(D, mu_exp_val)
            F = self.get_fock(D, h1e, mu_exp_val)
            E_total = self.get_energy_tot(h1e, F, D, mu_exp_val)

            # Get new density matrix and energy\
            C = self.get_coeffs(F)
            D_new = self.make_density(C)
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
                    2.0 * self.cqed_dij(D, mu_exp_val),
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
                print(f"DIPOLE CORRECTION ENERGY: {cqed_dc_e:.10f}")
                print(f"Q ENERGY: {cqed_q_e:.10f}")
                print("\nSCF Converged!")
                return E_total, C

            D = D_new
            E_old = E_total

        raise RuntimeWarning("SCF did not converge within maximum iterations")

    def get_mo_integral(self, mo: npt.NDArray, ncore=0, ncas=None):
        from pyscf import ao2mo

        if ncas is None:
            ncas = mo.shape[1] - ncore

        orb_sym = [0] * ncas
        ecore = self.E_nn_ + self.get_energy_dc(self.make_density(mo))
        mo_core = mo[:, :ncore]
        mo_cas = mo[:, ncore : ncore + ncas]

        mu_eff_val = self.compute_mu_exp(self.make_density(mo))

        hveff_ao = 0

        if ncore != 0:
            core_dm = self.make_density(mo_core)
            hveff_ao = self.cqed_veff(core_dm)
            h1e_ao = self.cqed_h1e(None, mu_eff_val)
            ecore += np.einsum(
                "ij,ji->", core_dm, 2.0 * h1e_ao + hveff_ao, optimize=True
            )

        # first term in h1e
        hcore = np.einsum(
            "ip, ij, jq -> pq", mo_cas, self.H_ + hveff_ao, mo_cas, optimize=True
        )

        # second term in h1e
        d = np.einsum(
            "ip, xij, jq -> xpq",
            mo_cas,
            self.ao_dipole_,
            mo_cas,
            optimize=True,
        )
        dpq = np.einsum(
            "x, xpq -> pq",
            self.lambda_vec_,
            d,
            optimize=True,
        )
        de = self.l_dot_mu_nuc_ - np.einsum(
            "x,x->",
            self.lambda_vec_,
            self.compute_mu_exp(self.make_density(mo)),
            optimize=True,
        )

        de_dpq = de * dpq

        # third term in h1e
        q = np.einsum(
            "ip, xyij, jq->xypq",
            mo_cas,
            -1.0 * self.mol_.intor("int1e_rr").reshape(3, 3, self.nao_, self.nao_),
            mo_cas,
            optimize=True,
        )

        qpq = (
            self.lambda_vec_[0] ** 2 * q[0, 0]
            + self.lambda_vec_[1] ** 2 * q[1, 1]
            + self.lambda_vec_[2] ** 2 * q[2, 2]
            + 2.0 * self.lambda_vec_[0] * self.lambda_vec_[1] * q[0, 1]
            + 2.0 * self.lambda_vec_[0] * self.lambda_vec_[2] * q[0, 2]
            + 2.0 * self.lambda_vec_[1] * self.lambda_vec_[2] * q[1, 2]
        )

        # Ignore nuc dipole in J. Chem. Theory Comput. 2024, 20, 9424−9434
        # But for simple case, we don't
        h1e = hcore + de_dpq - 0.5 * qpq

        # first term in g2e
        eri_mo = ao2mo.kernel(self.eri_, mo_cas)
        # second term in g2e
        dd = np.einsum("pq, rs->pqrs", dpq, dpq, optimize=True)

        g2e = eri_mo + dd

        # ecore += self.get_energy_dc(dm)

        n_elec = self.mol_.nelectron - 2 * ncore
        spin = self.mol_.spin
        return ncas, n_elec, spin, ecore, h1e, g2e, dpq, de, orb_sym


if __name__ == "__main__":
    from pyscf import gto, scf

    mol = gto.M(
        atom="""
        Li     0.0, 0.0, 0.0
        H      0.0, 0.0, 1.4
                """,
        basis="sto-3g",
    )

    mol.set_common_orig((0.0, 0.0, 0.0))
    lam = np.array([0.0, 0.0, 0.05])

    cqed_rhf = CQED_RHF(mol, lam)
    cqed_e, C = cqed_rhf.kernel()

    print(f"Final SCF energy (PySCF): {cqed_rhf.pyscf_rhf_energy_:.10f}")
    print(f"Final CQED_RHF energy: {cqed_e:.10f}")

    ncas, n_elec, spin, ecore, h1e, g2e, dpq, de, orb_sym = cqed_rhf.get_mo_integral(
        C, ncore=0, ncas=None
    )
    print("ncas:", ncas, "n_elec:", n_elec, "spin:", spin, "ecore:", ecore)

    # import h5py

    # with h5py.File("cqed_integrals.h5", "w") as f:
    #     f["norb"] = ncas
    #     f["nelec"] = n_elec
    #     f["spin"] = spin
    #     f["ecore"] = ecore

    #     f.create_dataset("h1e", data=h1e)
    #     f.create_dataset("h2e", data=g2e)

    #     f.create_dataset("dpq", data=dpq)
    #     f["de"] = de
    #     f.create_dataset("orb_irreps", data=orb_sym)
