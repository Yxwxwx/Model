from pyscf import gto, scf, ao2mo
import numpy as np
import scipy

from pyscf.data.nist import MP_ME

elec = gto.M(
    atom="""
    C     0.000000    0.000000    0.000000
    O     0.000000    0.000000    1.220000
    X-H     0.935307    0.000000   -0.540000
    X-H    -0.935307    0.000000   -0.540000
        """,
    basis="cc-pvdz",
    verbose=0,
    spin=0,
    cart=False,
)
protonic = gto.M(
    atom="""
    C     0.000000    0.000000    0.000000
    O     0.000000    0.000000    1.220000
    X-H     0.935307    0.000000   -0.540000
    X-H    -0.935307    0.000000   -0.540000
        """,
    basis={
        "X-H": [
            [0, (4.0, 1.0)],
            [1, (4.0, 1.0)],
        ],
    },
    cart=False,
    spin=0,
    verbose=0,
)
neo = gto.M(
    atom="""
    C     0.000000    0.000000    0.000000
    O     0.000000    0.000000    1.220000
    H     0.935307    0.000000   -0.540000
    H    -0.935307    0.000000   -0.540000
    X-H     0.935307    0.000000   -0.540000
    X-H    -0.935307    0.000000   -0.540000
    """,
    basis={
        "default": "cc-pvdz",
        "X-H": [
            [0, (4.0, 1.0)],
            [1, (4.0, 1.0)],
        ],
    },
    cart=False,
    verbose=4,
)

mol = gto.M(
    atom="""
   C     0.000000    0.000000    0.000000
   O     0.000000    0.000000    1.220000
   H     0.935307    0.000000   -0.540000
   H    -0.935307    0.000000   -0.540000
    """,
    basis="cc-pvdz",
    cart=False,
    verbose=4,
)

mf = scf.RHF(mol).run()


Mp = MP_ME
nao_e = elec.nao_nr()
nao_p = protonic.nao_nr()

nuc_e = elec.intor("int1e_nuc")
nuc_p = protonic.intor("int1e_nuc")


ovlp_all = neo.intor("int1e_ovlp")
ovlp_e = ovlp_all[:nao_e, :nao_e]
ovlp_p = ovlp_all[nao_e:, nao_e:]


kin_all = neo.intor("int1e_kin")
kin_e = kin_all[:nao_e, :nao_e]
kin_p = kin_all[nao_e:, nao_e:] / Mp

he = kin_e + nuc_e
hp = kin_p - nuc_p


I_all = neo.intor("int2e", aosym="s1")
I_ee = I_all[:nao_e, :nao_e, :nao_e, :nao_e]
I_pp = I_all[nao_e:, nao_e:, nao_e:, nao_e:]
I_ep = I_all[:nao_e, :nao_e, nao_e:, nao_e:]


docc = mol.nelectron // 2
print("electron docc:", docc)
Np = 2


def e_coeffs(fock):
    _, C = scipy.linalg.eigh(fock, ovlp_e)
    return C


def p_coeffs(fock):
    _, C = scipy.linalg.eigh(fock, ovlp_p)
    return C


def make_De(coeffs):
    C_occ = coeffs[:, :docc]
    return np.einsum("pi,qi->pq", C_occ, C_occ)


def make_Dp(coeffs):
    C_occ = coeffs[:, :Np]
    return np.einsum("pi,qi->pq", C_occ, C_occ)


def make_Fe(dm_e, dm_p):
    return (
        he
        + 2.0 * np.einsum("ijkl,lk->ij", I_ee, dm_e, optimize=True)
        - np.einsum("ilkj,lk->ij", I_ee, dm_e, optimize=True)
    ) - np.einsum("ijkl, lk->ij", I_ep, dm_p, optimize=True)


def make_Fp(dm_e, dm_p):
    return (
        hp
        + np.einsum("ijkl,lk->ij", I_pp, dm_p, optimize=True)
        - np.einsum("ilkj,lk->ij", I_pp, dm_p, optimize=True)
    ) - 2.0 * np.einsum("ijkl, ji->kl", I_ep, dm_e, optimize=True)


def energy(fock_e, dm_e, fock_p, dm_p):
    return (
        np.einsum("ij,ij->", he + fock_e, dm_e, optimize=True)
        + 0.5 * np.einsum("ij,ij->", hp + fock_p, dm_p, optimize=True)
        + elec.energy_nuc()
    )


def get_mo_integrals(Ce, Cp, ncore_e=0, ncas_e=None, ncore_p=0, ncas_p=None):
    from pyscf import ao2mo

    assert ncore_p == 0, "Currently only support no frozen core for protonic part"

    if ncas_e is None:
        ncas_e = Ce.shape[1] - ncore_e
    if ncas_p is None:
        ncas_p = Cp.shape[1] - ncore_p

    ecore = elec.energy_nuc()
    mo_e_core = Ce[:, :ncore_e]
    mo_e_cas = Ce[:, ncore_e : ncore_e + ncas_e]
    mo_p_core = Cp[:, :ncore_p]
    mo_p_cas = Cp[:, ncore_p : ncore_p + ncas_p]

    hveff_e_ao = 0

    if ncore_e != 0:
        core_e_dm = make_De(mo_e_core)
        hveff_e_ao = 2.0 * np.einsum(
            "ijkl,lk->ij", I_ee, core_e_dm, optimize=True
        ) - np.einsum("ilkj,lk->ij", I_ee, core_e_dm, optimize=True)
        ecore += np.einsum("ij,ij->", 2.0 * he + hveff_e_ao, core_e_dm, optimize=True)

    hcore_e = np.einsum(
        "ip, ij, jq -> pq", mo_e_cas, he + hveff_e_ao, mo_e_cas, optimize=True
    )
    hcore_p = np.einsum("ip, ij, jq -> pq", mo_p_cas, hp, mo_p_cas, optimize=True)

    eri_ee = ao2mo.full(I_ee, mo_e_cas)
    eri_pp = ao2mo.full(I_pp, mo_p_cas)
    tmp = np.einsum("ijpq,qQ->ijpQ", I_ep, mo_p_cas, optimize=True)
    tmp = np.einsum("ijpQ,pP->ijPQ", tmp, mo_p_cas, optimize=True)
    tmp = np.einsum("ijPQ,jJ->iJPQ", tmp, mo_e_cas, optimize=True)
    eri_ep = np.einsum("iJPQ,iI->IJPQ", tmp, mo_e_cas, optimize=True)

    n_elec = mol.nelectron - 2 * ncore_e
    n_proton = Np - ncore_p
    spin = mol.spin

    return (
        ncas_e,
        ncas_p,
        n_elec,
        n_proton,
        spin,
        ecore,
        hcore_e,
        hcore_p,
        eri_ee,
        eri_pp,
        eri_ep,
    )


De = mf.make_rdm1() * 0.5
Dp = make_Dp(hp)
Ce = np.zeros_like(De)
Cp = np.zeros_like(Dp)
De_old = De
Do_old = Dp
E_old = mf.energy_tot()
max_iter = 100

for out_iter in range(max_iter):
    Fe = make_Fe(De, Dp)
    for p_iter in range(50):
        Fp = make_Fp(De, Dp)
        E_new = energy(Fe, De, Fp, Dp)
        Cp = p_coeffs(Fp)
        Dp_new = make_Dp(Cp)
        if np.linalg.norm(Dp_new - Dp) < 1e-5:
            break
        Dp = Dp_new

    Fp = make_Fp(De, Dp)
    for e_iter in range(50):
        Fe = make_Fe(De, Dp)
        E_new = energy(Fe, De, Fp, Dp)
        Ce = e_coeffs(Fe)
        De_new = make_De(Ce)
        if np.linalg.norm(De_new - De) < 1e-5:
            break
        De = De_new

    E_new = energy(Fe, De, Fp, Dp)
    print(f"Outer iter {out_iter}: E = {E_new:.12f}")

    if np.abs(E_new - E_old) < 1e-12:
        print("NEO-SCF converged!")
        break
    E_old = E_new
(
    ncas_e,
    ncas_p,
    n_elec,
    n_proton,
    spin,
    ecore,
    hcore_e,
    hcore_p,
    eri_ee,
    eri_pp,
    eri_ep,
) = get_mo_integrals(Ce, Cp, ncore_e=0, ncas_e=None, ncore_p=0, ncas_p=None)
print(
    "ncas_e:",
    ncas_e,
    "ncas_p:",
    ncas_p,
    "n_elec:",
    n_elec,
    "n_proton:",
    n_proton,
    "spin:",
    spin,
    "ecore:",
    ecore,
)

E = ecore
E += 2.0 * np.einsum("ii->", hcore_e[:docc, :docc], optimize=True)
E += np.einsum("iijj->", 2.0 * eri_ee[:docc, :docc, :docc, :docc], optimize=True)
E -= np.einsum("ijji->", eri_ee[:docc, :docc, :docc, :docc], optimize=True)
E += np.einsum("pp->", hcore_p[:Np, :Np], optimize=True)
E += 0.5 * np.einsum("ppqq->", eri_pp[:Np, :Np, :Np, :Np], optimize=True)
E -= 0.5 * np.einsum("pqqp->", eri_pp[:Np, :Np, :Np, :Np], optimize=True)
E -= 2.0 * np.einsum("iipp->", eri_ep[:docc, :docc, :Np, :Np], optimize=True)
print(E)
