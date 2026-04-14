from pyscf import gto, scf, ao2mo
import numpy as np
import scipy
import h5py
from pyscf.data.nist import MP_ME as Mp

# for elec part int_nuc
elec = gto.M(
    atom="""
    He     -1.74535105    0.000000    0.000000
    He     1.74535105      0.000000    0.000000
    X-H   0.000000       0.000000    0.000000
    """,
    basis={
        "He": "def2-svp",
        "X-H": "def2-svp",
    },
    charge=0,
    unit="bohr",
    cart=True,
    verbose=0,
)

# for proton part int_nuc
protonic = gto.M(
    atom="""
    He     -1.74535105    0.000000    0.000000
    He     1.74535105      0.000000    0.000000
    X-H   0.000000       0.000000    0.000000
    """,
    basis={
        "X-H": [  # PB4-D
            # 4s
            [0, (1.217, 1.0)],
            [0, (10.238, 1.0)],
            [0, (18.246, 1.0)],
            [0, (30.285, 1.0)],
            # 3p
            [1, (10.708, 1.0)],
            [1, (12.068, 1.0)],
            [1, (23.066, 1.0)],
            # 2d
            [2, (11.586, 1.0)],
            [2, (21.462, 1.0)],
        ],
    },
    charge=0,
    unit="bohr",
    cart=True,
    spin=0,
    verbose=0,
)

# for initial guess
mol = gto.M(
    atom="""
    He     -1.74535105    0.000000    0.000000
    He     1.74535105      0.000000    0.000000
    H   0.000000       0.000000    0.000000
    """,
    basis={
        "He": "def2-svp",
        "H": "def2-svp",
    },
    charge=+1,
    unit="bohr",
    cart=True,
    verbose=4,
)

neo = gto.M(
    atom="""
    He     -1.74535105    0.000000    0.000000
    He     1.74535105      0.000000    0.000000
    H      0.000000       0.000000    0.000000
    X-H    0.000000       0.000000    0.000000
    """,
    basis={
        "He": "def2-svp",
        "H": "def2-svp",
        "X-H": [  # PB4-D
            # 4s
            [0, (1.217, 1.0)],
            [0, (10.238, 1.0)],
            [0, (18.246, 1.0)],
            [0, (30.285, 1.0)],
            # 3p
            [1, (10.708, 1.0)],
            [1, (12.068, 1.0)],
            [1, (23.066, 1.0)],
            # 2d
            [2, (11.586, 1.0)],
            [2, (21.462, 1.0)],
        ],
    },
    charge=+1,
    unit="bohr",
    cart=True,
    verbose=4,
)


mf = scf.RHF(mol).run()

nao_e = elec.nao_nr()
nao_p = protonic.nao_nr()

nuc_e = elec.intor("int1e_nuc")
nuc_p = protonic.intor("int1e_nuc")


ovlp_all = neo.intor("int1e_ovlp")
ovlp_e = ovlp_all[:nao_e, :nao_e]
ovlp_p = ovlp_all[nao_e:, nao_e:]

Ae = scipy.linalg.fractional_matrix_power(ovlp_e, -0.5)
Ap = scipy.linalg.fractional_matrix_power(ovlp_p, -0.5)


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
Np = 1


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


def compute_neo_diis_res(
    F_tuple: tuple[np.ndarray, np.ndarray],
    D_tuple: tuple[np.ndarray, np.ndarray],
    S_tuple: tuple[np.ndarray, np.ndarray],
    A_tuple: tuple[np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """计算电子和质子的 DIIS 残差向量 e = A(FDS - SDF)A"""
    Fe, Fp = F_tuple
    De, Dp = D_tuple
    Se, Sp = S_tuple
    Ae, Ap = A_tuple

    res_e = Ae @ (Fe @ De @ Se - Se @ De @ Fe) @ Ae
    res_p = Ap @ (Fp @ Dp @ Sp - Sp @ Dp @ Fp) @ Ap

    return res_e, res_p


def apply_neo_diis(
    F_list: list[tuple[np.ndarray, np.ndarray]],
    DIIS_list: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray]:
    """应用 DIIS 更新电子和质子的 Fock 矩阵"""
    B_dim = len(F_list) + 1
    B = np.zeros((B_dim, B_dim))

    # 填充 B 矩阵：将电子和质子的残差内积相加
    for i in range(len(F_list)):
        for j in range(i, len(F_list)):
            val_e = np.einsum(
                "ij,ij->", DIIS_list[i][0], DIIS_list[j][0], optimize=True
            )
            val_p = np.einsum(
                "ij,ij->", DIIS_list[i][1], DIIS_list[j][1], optimize=True
            )
            B[i, j] = B[j, i] = val_e + val_p

    B[-1, :-1] = -1
    B[:-1, -1] = -1
    B[-1, -1] = 0

    rhs = np.zeros(B_dim)
    rhs[-1] = -1

    try:
        coeffs = np.linalg.solve(B, rhs)
    except np.linalg.LinAlgError:
        # 如果 B 矩阵奇异，退化为不使用 DIIS，返回最近的 Fock 矩阵
        coeffs = np.zeros(B_dim)
        coeffs[-2] = 1.0

    # 使用求出的同一套系数，同时外推 Fe 和 Fp
    Fe_new = np.einsum("i,ikl->kl", coeffs[:-1], [f[0] for f in F_list])
    Fp_new = np.einsum("i,ikl->kl", coeffs[:-1], [f[1] for f in F_list])

    return Fe_new, Fp_new


def get_mo_integrals(Ce, Cp, ncore_e=0, ncas_e=None, ncore_p=0, ncas_p=None):

    assert ncore_p == 0, "Are you sure you don't need proton?"

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

    h1e = np.einsum(
        "ip, ij, jq -> pq", mo_e_cas, he + hveff_e_ao, mo_e_cas, optimize=True
    )
    h1p = np.einsum("ip, ij, jq -> pq", mo_p_cas, hp, mo_p_cas, optimize=True)

    g2ee = ao2mo.full(I_ee, mo_e_cas)
    g2pp = ao2mo.full(I_pp, mo_p_cas)
    tmp = np.einsum("ijpq,qQ->ijpQ", I_ep, mo_p_cas, optimize=True)
    tmp = np.einsum("ijpQ,pP->ijPQ", tmp, mo_p_cas, optimize=True)
    tmp = np.einsum("ijPQ,jJ->iJPQ", tmp, mo_e_cas, optimize=True)
    g2ep = np.einsum("iJPQ,iI->IJPQ", tmp, mo_e_cas, optimize=True)

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
        h1e,
        h1p,
        g2ee,
        g2pp,
        g2ep,
    )


def get_spin_mo_integrals(Ce, Cp, ncore_e=0, ncas_e=None, ncore_p=0, ncas_p=None):
    (
        ncas_e,
        ncas_p,
        n_elec,
        n_proton,
        spin,
        ecore,
        h1e,
        h1p,
        g2ee,
        g2pp,
        g2ep,
    ) = get_mo_integrals(
        Ce, Cp, ncore_e=ncore_e, ncas_e=ncas_e, ncore_p=ncore_p, ncas_p=ncas_p
    )
    gh1e = np.zeros((ncas_e * 2, ncas_e * 2))
    gg2ee = np.zeros((ncas_e * 2, ncas_e * 2, ncas_e * 2, ncas_e * 2))
    gg2ep = np.zeros((ncas_e * 2, ncas_e * 2, ncas_p, ncas_p))

    for i in range(ncas_e * 2):
        for j in range(i % 2, ncas_e * 2, 2):
            gh1e[i, j] = h1e[i // 2, j // 2]

    for i in range(ncas_e * 2):
        for j in range(i % 2, ncas_e * 2, 2):
            for k in range(ncas_e * 2):
                for l in range(k % 2, ncas_e * 2, 2):
                    gg2ee[i, j, k, l] = g2ee[i // 2, j // 2, k // 2, l // 2]

    for i in range(ncas_e * 2):
        for j in range(i % 2, ncas_e * 2, 2):
            for k in range(ncas_p):
                for l in range(ncas_p):
                    gg2ep[i, j, k, l] = g2ep[i // 2, j // 2, k, l]

    assert np.linalg.norm(gh1e - gh1e.T.conj()) < 1e-10

    return (
        ncas_e * 2,
        ncas_p,
        n_elec,
        n_proton,
        spin,
        ecore,
        gh1e,
        h1p,
        gg2ee,
        g2pp,
        gg2ep,
    )


De = mf.make_rdm1() * 0.5

Fp_init = make_Fp(De, np.zeros_like(hp))
Dp = make_Dp(p_coeffs(Fp_init))
Ce = np.zeros_like(De)
Cp = np.zeros_like(Dp)
E_old = mf.energy_tot()
max_iter = 100

# DIIS
DIIS = True
F_list = []
DIIS_list = []
diis_space = 12
diis_start = 2

for iter_num in range(max_iter):
    Fe = make_Fe(De, Dp)
    Fp = make_Fp(De, Dp)
    E_new = energy(Fe, De, Fp, Dp)

    res_e, res_p = compute_neo_diis_res((Fe, Fp), (De, Dp), (ovlp_e, ovlp_p), (Ae, Ap))
    F_list.append((Fe, Fp))
    DIIS_list.append((res_e, res_p))

    if len(F_list) > diis_space:
        F_list.pop(0)
        DIIS_list.pop(0)

    if DIIS and iter_num >= diis_start:
        Fe_extrap, Fp_extrap = apply_neo_diis(F_list, DIIS_list)
    else:
        Fe_extrap, Fp_extrap = Fe, Fp
    Ce = e_coeffs(Fe_extrap)
    Cp = p_coeffs(Fp_extrap)
    De_new = make_De(Ce)
    Dp_new = make_Dp(Cp)

    dE = np.abs(E_new - E_old)
    dD = (np.linalg.norm(De_new - De) + np.linalg.norm(Dp_new - Dp)) / 2.0

    print(
        f"Iter {iter_num:3d}: E = {E_new:.12f}, dE = {dE:.3e}, dD = {dD:.3e}, {'CDIIS' if DIIS and iter_num >= diis_start else ''}"
    )

    if dE < 1e-8 and dD < 1e-6:
        print("NEO-SCF converged!")
        break

    De = De_new
    Dp = Dp_new
    E_old = E_new

(
    ncas_e,
    ncas_p,
    n_elec,
    n_proton,
    spin,
    ecore,
    h1e,
    h1p,
    g2ee,
    g2pp,
    g2ep,
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


with h5py.File("neo_sz.h5", "w") as f:
    f.create_dataset("ncas_e", data=ncas_e)
    f.create_dataset("ncas_p", data=ncas_p)
    f.create_dataset("n_elec", data=n_elec)
    f.create_dataset("n_proton", data=n_proton)
    f.create_dataset("spin", data=spin)
    f.create_dataset("ecore", data=ecore)
    f.create_dataset("h1e", data=h1e)
    f.create_dataset("h1p", data=h1p)
    f.create_dataset("g2ee", data=g2ee)
    f.create_dataset("g2pp", data=g2pp)
    f.create_dataset("g2ep", data=g2ep)


(
    ncas_e,
    ncas_p,
    n_elec,
    n_proton,
    spin,
    ecore,
    h1e,
    h1p,
    g2ee,
    g2pp,
    g2ep,
) = get_spin_mo_integrals(Ce, Cp, ncore_e=0, ncas_e=None, ncore_p=0, ncas_p=None)
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


with h5py.File("neo_u1.h5", "w") as f:
    f.create_dataset("ncas_e", data=ncas_e)
    f.create_dataset("ncas_p", data=ncas_p)
    f.create_dataset("n_elec", data=n_elec)
    f.create_dataset("n_proton", data=n_proton)
    f.create_dataset("spin", data=spin)
    f.create_dataset("ecore", data=ecore)
    f.create_dataset("h1e", data=h1e)
    f.create_dataset("h1p", data=h1p)
    f.create_dataset("g2ee", data=g2ee)
    f.create_dataset("g2pp", data=g2pp)
    f.create_dataset("g2ep", data=g2ep)


def test_mo_integrals():
    E = ecore
    E += 2.0 * np.einsum("ii->", h1e[:docc, :docc], optimize=True)
    E += np.einsum("iijj->", 2.0 * g2ee[:docc, :docc, :docc, :docc], optimize=True)
    E -= np.einsum("ijji->", g2ee[:docc, :docc, :docc, :docc], optimize=True)
    E += np.einsum("pp->", h1p[:Np, :Np], optimize=True)
    E += 0.5 * np.einsum("ppqq->", g2pp[:Np, :Np, :Np, :Np], optimize=True)
    E -= 0.5 * np.einsum("pqqp->", g2pp[:Np, :Np, :Np, :Np], optimize=True)
    E -= 2.0 * np.einsum("iipp->", g2ep[:docc, :docc, :Np, :Np], optimize=True)
    assert np.abs(E - E_new) < 1e-10, (
        f"MO integrals test failed! E={E}, expected {E_new}"
    )


# test_mo_integrals()
