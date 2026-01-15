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

A_e = scipy.linalg.fractional_matrix_power(ovlp_e, -0.5)
A_p = scipy.linalg.fractional_matrix_power(ovlp_p, -0.5)


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


def make_De(fock):
    _, C = scipy.linalg.eigh(fock, ovlp_e)
    C_occ = C[:, :docc]
    return np.einsum("pi,qi->pq", C_occ, C_occ)


def make_Dp(fock):
    _, C = scipy.linalg.eigh(fock, ovlp_p)
    C_occ = C[:, :Np]
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


def compute_diis_res(ovlp, fock, dm):
    return fock @ dm @ ovlp - ovlp @ dm @ fock


def diis(fock_list, diis_list):
    B_dim = len(fock_list) + 1
    B = np.zeros((B_dim, B_dim))
    B[-1, :] = -1
    B[:, -1] = -1
    B[-1, -1] = 0

    for i in range(len(fock_list)):
        for j in range(len(fock_list)):
            # Compute the inner product of residuals
            B[i, j] = np.einsum("ij,ij->", diis_list[i], diis_list[j], optimize=True)

    rhs = np.zeros((B_dim))
    rhs[-1] = -1
    coeff = np.linalg.solve(B, rhs)

    F_new = np.einsum("i,ikl->kl", coeff[:-1], fock_list)

    return F_new


De = mf.make_rdm1() * 0.5
Dp = make_Dp(hp)
De_old = De
Do_old = Dp
E_old = mf.energy_tot()
max_iter = 100
focke_list = []
diise_list = []
fockp_list = []
diisp_list = []
diis_space = 8

for out_iter in range(max_iter):
    Fe = make_Fe(De, Dp)
    for p_iter in range(50):
        Fp = make_Fp(De, Dp)
        E_new = energy(Fe, De, Fp, Dp)
        Dp_new = make_Dp(Fp)
        if np.linalg.norm(Dp_new - Dp) < 1e-5:
            break
        Dp = Dp_new

    Fp = make_Fp(De, Dp)
    for e_iter in range(50):
        Fe = make_Fe(De, Dp)
        E_new = energy(Fe, De, Fp, Dp)
        De_new = make_De(Fe)
        if np.linalg.norm(De_new - De) < 1e-5:
            break
        De = De_new
    
    E_new = energy(Fe, De, Fp, Dp)
    print(f"Outer iter {out_iter}: E = {E_new:.12f}")

    if np.abs(E_new - E_old) < 1e-12:
        print("NEO-SCF converged!")
        break
    E_old = E_new
