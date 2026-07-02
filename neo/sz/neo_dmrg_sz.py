import numpy as np
import h5py
from pyblock2.driver.core import DMRGDriver, SymmetryTypes, MPOAlgorithmTypes
from time import time

# const
bond_dims = [250] * 4 + [500] * 4 + [1000]
noises = [1e-4] * 4 + [1e-5] * 4 + [0]
thrds = [1e-9]

# load integral
filename = "neo_u1.h5"

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
print(
    f"ncas_e: {ncas_e}, ncas_p: {ncas_p} , n_elec: {n_elec} , n_proton: {n_proton}, spin: {spin}, eore: {ecore}"
)

driver = DMRGDriver(
    scratch="./tmp/neo_dmrg_u1",
    symm_type=SymmetryTypes.SAny,
    n_threads=4,
    stack_mem=100 << 30,
)

start = time()  # Start MPO construction

# quantum number wrapper (U1 / n_elec, U1 / n_proton)
driver.set_symmetry_groups("U1Fermi", "U1Fermi")
Q = driver.bw.SX

# [Part A] Set states and matrix representation of operators in local Hilbert space
site_basis, site_ops = [], []
L = ncas_e + ncas_p
for k in range(L):
    if k < ncas_e:
        # elec ->
        basis = [
            (Q(0, 0), 1),
            (Q(1, 0), 1),
        ]
        ops = {
            "": np.eye(2),  # identity
            "c": np.array([[0, 0], [1, 0]]),  # e+
            "d": np.array([[0, 1], [0, 0]]),  # e
        }
    else:
        # proton ->
        basis = [
            (Q(0, 0), 1),
            (Q(0, 1), 1),
        ]
        ops = {
            "": np.eye(2),  # identity
            "C": np.array([[0, 0], [1, 0]]),  # p+
            "D": np.array([[0, 1], [0, 0]]),  # p
        }

    site_ops.append(ops)
    site_basis.append(basis)

# [Part B] Set Hamiltonian terms in NEO model
driver.initialize_system(
    n_sites=L,
    vacuum=Q(0, 0),
    target=Q(n_elec, n_proton),
    hamil_init=False,
)
driver.ghamil = driver.get_custom_hamiltonian(site_basis, site_ops)
b = driver.expr_builder()

h1e[np.abs(h1e) < 1e-12] = 0.0
h1p[np.abs(h1p) < 1e-12] = 0.0
g2ee[np.abs(g2ee) < 1e-12] = 0.0
g2pp[np.abs(g2pp) < 1e-12] = 0.0
g2ep[np.abs(g2ep) < 1e-12] = 0.0


# 1. electron-electron interaction
for i in range(ncas_e):
    for j in range(ncas_e):
        b.add_term("cd", [i, j], h1e[i, j])

for i in range(ncas_e):
    for j in range(ncas_e):
        for k in range(ncas_e):
            for l in range(ncas_e):
                b.add_term("ccdd", [i, k, l, j], 0.5 * g2ee[i, j, k, l])

# 2. proton-proton interaction
for p in range(ncas_p):
    for q in range(ncas_p):
        idx = [p + ncas_e, q + ncas_e]
        b.add_term("CD", idx, h1p[p, q])

for p in range(ncas_p):
    for q in range(ncas_p):
        for r in range(ncas_p):
            for s in range(ncas_p):
                idx = [p + ncas_e, r + ncas_e, s + ncas_e, q + ncas_e]
                b.add_term("CCDD", idx, 0.5 * g2pp[p, q, r, s])

# 3. electron-proton interaction
for i in range(ncas_e):
    for j in range(ncas_e):
        for p in range(ncas_p):
            for q in range(ncas_p):
                idx = [i, j, p + ncas_e, q + ncas_e]
                b.add_term("cdCD", idx, -1.0 * g2ep[i, j, p, q])

b.add_const(ecore)

# [Part C] Perform DMRG
mpo = driver.get_mpo(
    b.finalize(adjust_order=True),
    algo_type=MPOAlgorithmTypes.FastBipartite,
    iprint=1,
)

end = time()  # End MPO construction
print(f"MPO construction time: {end - start:.2f} seconds")

start = time()  # Start DMRG sweeps
mps = driver.get_random_mps(tag="KET", bond_dim=250, nroots=2)
energy = driver.dmrg(
    mpo,
    mps,
    tol=1e-8,
    n_sweeps=20,
    bond_dims=bond_dims,
    noises=noises,
    thrds=thrds,
    iprint=2,
)
print("DMRG energy = \n", energy)
# −5.849810,−5.840512
# [-5.849818741514213, -5.840515388972684]
end = time()  # End DMRG sweeps
print(f"DMRG sweeps time: {end - start:.2f} seconds")