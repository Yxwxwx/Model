import numpy as np
import h5py
from pyblock2.driver.core import DMRGDriver, SymmetryTypes, MPOAlgorithmTypes

# const
bond_dims = [250] * 4 + [500] * 4
noises = [1e-4] * 4 + [1e-5] * 4 + [0]
thrds = [1e-10] * 8

# load integral
filename = "neo_sz.h5"

with h5py.File(filename, "r") as f:
    try:
        ncas_e = f["ncas_e"][()]
        ncas_p = f["ncas_p"][()]
        n_elec = f["n_elec"][()]
        n_proton = f["n_proton"][()]
        spin = f["spin"][()]
        ecore = f["ecore"][()]

        h1e = f["h1e"][...]
        h1p = f["h1p"][...]
        g2ee = f["g2ee"][...]
        g2pp = f["g2pp"][...]
        g2ep = f["g2ep"][...]
    except KeyError as e:
        print(f"Error: Missing expected dataset in HDF5 file: {e}")
        raise

driver = DMRGDriver(
    scratch="/nvme/Yxwxwx/neo_dmrg_sz/",
    symm_type=SymmetryTypes.SAny,
    n_threads=16,
    stack_mem=200 << 30,
)

# quantum number wrapper (U1 / n_elec, U1 / 2*Sz, U1 / n_proton)
driver.set_symmetry_groups("U1Fermi", "U1", "U1Fermi")
Q = driver.bw.SX
L = ncas_e + ncas_p

# [Part A] Set states and matrix representation of operators in local Hilbert space
site_basis, site_ops = [], []

for k in range(L):
    if k < ncas_e:
        basis = [
            (Q(0, 0, 0), 1),  # 0: |vac>
            (Q(1, 1, 0), 1),  # 1: |alpha>
            (Q(1, -1, 0), 1),  # 2: |beta>
            (Q(2, 0, 0), 1),  # 3: |alpha beta>
        ]
        ops = {
            "": np.eye(4),
            "c": np.array(
                [[0, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0]]
            ),  # alpha+
            "d": np.array(
                [[0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0]]
            ),  # alpha
            "C": np.array(
                [[0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 0], [0, -1, 0, 0]]
            ),  # beta+
            "D": np.array(
                [[0, 0, 1, 0], [0, 0, 0, -1], [0, 0, 0, 0], [0, 0, 0, 0]]
            ),  # beta
        }
    else:
        basis = [
            (Q(0, 0, 0), 1),  # 0: |vac>
            (Q(0, 0, 1), 1),  # 1: |proton>
        ]
        ops = {
            "": np.eye(2),
            "E": np.array([[0, 0], [1, 0]]),  # proton+
            "F": np.array([[0, 1], [0, 0]]),  # proton
        }

    site_basis.append(basis)
    site_ops.append(ops)

# [Part B] Set Hamiltonian terms in NEO model
driver.initialize_system(
    n_sites=L, vacuum=Q(0, 0, 0), target=Q(n_elec, spin, n_proton), hamil_init=False
)
driver.ghamil = driver.get_custom_hamiltonian(site_basis, site_ops)
b = driver.expr_builder()

# 1. electron-electron interaction
for i in range(ncas_e):
    for j in range(ncas_e):
        if abs(h1e[i, j]) > 1e-12:
            b.add_term("cd", [i, j], h1e[i, j])
            b.add_term("CD", [i, j], h1e[i, j])

for i in range(ncas_e):
    for j in range(ncas_e):
        for k in range(ncas_e):
            for l in range(ncas_e):
                if abs(g2ee[i, j, k, l]) > 1e-12:
                    coef = 0.5 * g2ee[i, j, k, l]
                    b.add_term("ccdd", [i, k, l, j], coef)  # alpha-alpha
                    b.add_term("CCDD", [i, k, l, j], coef)  # beta-beta
                    b.add_term("cCDd", [i, k, l, j], coef)  # alpha-beta
                    b.add_term("CcdD", [i, k, l, j], coef)  # beta-alpha

print("END set electron-electron interaction")
# 2. proton-proton interaction
for p in range(ncas_p):
    for q in range(ncas_p):
        if abs(h1p[p, q]) > 1e-12:
            idx = [p + ncas_e, q + ncas_e]
            b.add_term("EF", idx, h1p[p, q])

for p in range(ncas_p):
    for q in range(ncas_p):
        for r in range(ncas_p):
            for s in range(ncas_p):
                if abs(g2pp[p, q, r, s]) > 1e-12:
                    coef = 0.5 * g2pp[p, q, r, s]
                    idx = [p + ncas_e, r + ncas_e, s + ncas_e, q + ncas_e]
                    b.add_term("EEFF", idx, coef)
print("END set proton-proton interaction")
# 3. electron-proton interaction
for i in range(ncas_e):
    for j in range(ncas_e):
        for p in range(ncas_p):
            for q in range(ncas_p):
                if abs(g2ep[i, j, p, q]) > 1e-12:
                    coef = -1.0 * g2ep[i, j, p, q]
                    idx = [i, j, p + ncas_e, q + ncas_e]
                    b.add_term("cdEF", idx, coef)  # alpha
                    b.add_term("CDEF", idx, coef)  # beta
print("END set electron-proton interaction")
b.add_const(ecore)

# [Part C] Perform DMRG
mpo = driver.get_mpo(
    b.finalize(adjust_order=True),
    algo_type=MPOAlgorithmTypes.FastBipartite,
    iprint=1,
)
mps = driver.get_random_mps(tag="KET", bond_dim=250, nroots=1)
energy = driver.dmrg(
    mpo,
    mps,
    tol=1e-6,
    n_sweeps=20,
    bond_dims=bond_dims,
    noises=noises,
    thrds=thrds,
    iprint=2,
)
print("DMRG energy = %20.15f" % energy)  # −93.053328
