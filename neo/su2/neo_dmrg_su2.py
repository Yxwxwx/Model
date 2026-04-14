import numpy as np
import h5py
from pyblock2.driver.core import DMRGDriver, SymmetryTypes, MPOAlgorithmTypes

# const
bond_dims = [250] * 4 + [500] * 4
noises = [1e-4] * 4 + [1e-5] * 4 + [0]
thrds = [1e-10] * 8

# load integral
filename = "neo_sz.h5"  # the same as su2

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

print("END load integral")
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
driver = DMRGDriver(
    scratch="/nvme/Yxwxwx/neo_dmrg_su2/",
    symm_type=SymmetryTypes.SAnySU2,
    n_threads=16,
    stack_mem=200 << 30,
)

# quantum number wrapper (U1 / n_elec, SU2 / 2*S, U1 / n_proton)
driver.set_symmetry_groups("U1Fermi", "SU2", "SU2", "U1Fermi")
Q = driver.bw.SX
L = ncas_e + ncas_p

# [Part A] Set states and matrix representation of operators in local Hilbert space
site_basis, site_ops = [], []

for k in range(L):
    if k < ncas_e:
        basis = [
            (Q(0, 0, 0, 0), 1),  # 0: |0>
            (Q(1, 1, 1, 0), 1),  # 1: |1>
            (Q(2, 0, 0, 0), 1),  # 2: |2>
        ]
        ops = {
            "": np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),  # identity
            "C": np.array([[0, 0, 0], [1, 0, 0], [0, -(np.sqrt(2)), 0]]),  # +
            "D": np.array([[0, np.sqrt(2), 0], [0, 0, 1], [0, 0, 0]]),  # -
        }
    else:
        basis = [
            (Q(0, 0, 0, 0), 1),  # 0: |vac>
            (Q(0, 0, 0, 1), 1),  # 1: |proton>
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
    n_sites=L,
    vacuum=Q(0, 0, 0, 0),
    target=Q(n_elec, spin, spin, n_proton),
    hamil_init=False,
)
driver.ghamil = driver.get_custom_hamiltonian(site_basis, site_ops)
b = driver.expr_builder()

# 1. electron-electron interaction
for i in range(ncas_e):
    for j in range(ncas_e):
        if abs(h1e[i, j]) > 1e-12:
            coef = np.sqrt(2) * h1e[i, j]
            b.add_term("(C+D)0", [i, j], coef)

for i in range(ncas_e):
    for j in range(ncas_e):
        for k in range(ncas_e):
            for l in range(ncas_e):
                if abs(g2ee[i, j, k, l]) > 1e-12:
                    b.add_term("((C+(C+D)0)1+D)0", [i, k, l, j], g2ee[i, j, k, l])

print("END set electron-electron interaction")
# # 2. proton-proton interaction
# for p in range(ncas_p):
#     for q in range(ncas_p):
#         if abs(h1p[p, q]) > 1e-12:
#             idx = [p + ncas_e, q + ncas_e]
#             b.add_term("EF", idx, h1p[p, q])

# for p in range(ncas_p):
#     for q in range(ncas_p):
#         for r in range(ncas_p):
#             for s in range(ncas_p):
#                 if abs(g2pp[p, q, r, s]) > 1e-12:
#                     coef = 0.5 * g2pp[p, q, r, s]
#                     idx = [p + ncas_e, r + ncas_e, s + ncas_e, q + ncas_e]
#                     b.add_term("EEFF", idx, coef)
# print("END set proton-proton interaction")

# 3. electron-proton interaction
# for i in range(ncas_e):
#     for j in range(ncas_e):
#         for p in range(ncas_p):
#             for q in range(ncas_p):
#                 if abs(g2ep[i, j, p, q]) > 1e-12:
#                     coef = -1.0 * np.sqrt(2) * g2ep[i, j, p, q]
#                     idx = [i, j, p + ncas_e, q + ncas_e]
#                     b.add_term("(C+D)0EF", idx, coef)  # alpha
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
