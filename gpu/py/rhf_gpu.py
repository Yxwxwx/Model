import cupy as cp
from scipy.linalg import fractional_matrix_power

def eigh(fock, X):
    fock_p = cp.dot(cp.dot(X,fock),X)
    e, C_p = cp.linalg.eigh(fock_p)
    C = cp.dot(X,C_p)
    return e, C

def get_ovlp(mol):
    return cp.asarray(mol.intor("int1e_ovlp"))

def get_hcore(mol):
    h = mol.intor("int1e_kin") + mol.intor("int1e_nuc")
    return cp.asarray(h)

def get_eri(mol):
    eri = mol.intor("int2e_sph", aosym="s1")
    return cp.asarray(eri)

def get_occ(mol, mo_energy):
    e_idx = cp.argsort(mo_energy)
    e_sort = mo_energy[e_idx]
    nmo = len(mo_energy)
    mo_occ = cp.zeros(nmo)
    nocc = mol.nelectron // 2
    mo_occ[e_idx[:nocc]] = 2
    print(f" HOMO {e_sort[nocc - 1]:.15g}, LUMO {e_sort[nocc]:.15g}")
    return mo_occ

def make_rdm1(mo_coeff, mo_occ):
    mocc = mo_coeff[:, mo_occ > 0]
    # dm = (mocc * mo_occ[mo_occ > 0]).dot(mocc.T.conj())
    dm = cp.einsum(
        "pi,i,iq->pq", mocc, mo_occ[mo_occ > 0], mocc.T.conj(), optimize=True
    )
    return dm


def get_core_guess(mol, hcore, ovlp, X):
    e, C = eigh(hcore, X)
    occ = get_occ(mol, e)
    return make_rdm1(C, occ)

def get_jk(eri, dm):
    J = cp.einsum("pqrs,rs->pq", eri, dm, optimize=True)
    K = cp.einsum("prqs,rs->pq", eri, dm, optimize=True)
    return J, K

def get_veff(eri, dm):
    J, K = get_jk(eri, dm)
    return J - 0.5 * K

def get_fock(hcore, eri, dm):
    return hcore + get_veff(eri, dm)

def energy_elec(hcore, fock, dm):
    return 0.5 * cp.einsum("pq,pq->", hcore + fock, dm).item()

def energy_tot(mol, hcore, fock, dm):
    e_nuc = mol.energy_nuc()
    e_elec = energy_elec(hcore, fock, dm)
    return e_nuc + e_elec


def scf(mol, max_iter=100, conv_tol=1e-6):
    H = get_hcore(mol)
    S = get_ovlp(mol)
    I = get_eri(mol)
    A = cp.asarray(fractional_matrix_power(mol.intor("int1e_ovlp"), -0.5))

    D = get_core_guess(mol, H, S, A)
    hf_energy_old = 0.0

    for it in range(max_iter):
        F = get_fock(H, I, D)
        hf_energy = energy_tot(mol, H, F, D)

        e, C = eigh(F, A)
        occ = get_occ(mol, e)
        D_new = make_rdm1(C, occ)

        E_diff = hf_energy - hf_energy_old
        D_diff = cp.linalg.norm(D_new - D).item()

        print(f"SCF Iter {it}: HF Energy = {hf_energy:.12f}, "
              f"E_diff = {E_diff:.2e}, D_diff = {D_diff:.2e}")

        if abs(E_diff) < conv_tol and D_diff < conv_tol:
            print(f"SCF converged in {it} iterations.")
            print(f"HF Energy = {hf_energy:.12f}")
            return hf_energy

        D = D_new
        hf_energy_old = hf_energy

    raise RuntimeWarning("SCF did not converge.")


if __name__ == "__main__":
    from pyscf import gto

    mol = gto.M(
        atom = """
        O 0 0 0;
        H 0 0 1;
        H 0 1 0;
        """,
        basis = "cc-pvdz",
        charge = 0,
        spin = 0,
    )

    e = scf(mol)
    
    from pyscf import scf
    
    mf = scf.RHF(mol)
    
    e_ref = mf.kernel()
    
    assert cp.allclose(e, e_ref)
