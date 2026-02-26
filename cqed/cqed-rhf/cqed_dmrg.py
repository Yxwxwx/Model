import numpy as np
from pyblock2.driver.core import DMRGDriver, SymmetryTypes, MPOAlgorithmTypes

# const
bond_dims = [250] * 4 + [500] * 4 + [2000]
noises = [1e-4] * 4 + [1e-5] * 4 + [0]
thrds = [1e-10] * 8

from pyscf import gto

# molecular
mol = gto.M(
    atom="""
        Li   
        H   1  1.4
        """,
    basis="sto-3g",
)

mol.set_common_orig((0.0, 0.0, 0.0))
lam = np.array([0.0, 0.0, 0.05])

from cqed_rhf import CQED_RHF

# run cqed_rhf
cqed_rhf = CQED_RHF(mol, lam)
cqed_e, C = cqed_rhf.kernel()
print(f"Final SCF energy (PySCF): {cqed_rhf.pyscf_rhf_energy_:.10f}")
print(f"Final CQED_RHF energy: {cqed_e:.10f}")
ncas, n_elec, spin, ecore, h1e, g2e, dpq, de, orb_sym = cqed_rhf.get_mo_integrals(
    C, ncore=0, ncas=None
)
print("ncas:", ncas, "n_elec:", n_elec, "spin:", spin, "ecore:", ecore)

N_SITES_ELEC, N_SITES_PH = ncas, 1
L = N_SITES_ELEC + N_SITES_PH
N_ELEC, N_PH = n_elec, 2
OMEGA = 0.12086

driver = DMRGDriver(scratch="./tmp", symm_type=SymmetryTypes.SZ, n_threads=4)
driver.initialize_system(n_sites=L, n_elec=N_ELEC, spin=spin, orb_sym=None)


site_basis, site_ops = [], []
Q = driver.bw.SX  # quantum number wrapper (n_elec, 2 * spin, point group irrep)

for k in range(L):
    if k < N_SITES_ELEC:  # electron part
        basis = [
            (Q(0, 0, 0), 1),
            (Q(1, 1, 0), 1),
            (Q(1, -1, 0), 1),
            (Q(2, 0, 0), 1),
        ]  # [0ab2]
        ops = {
            "": np.array(
                [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
            ),  # identity
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
    else:  # phonon part
        basis = [(Q(0, 0, 0), N_PH)]
        ops = {
            "": np.identity(N_PH),  # identity
            "E": np.diag(np.sqrt(np.arange(1, N_PH)), k=-1),  # ph+
            "F": np.diag(np.sqrt(np.arange(1, N_PH)), k=1),  # ph
        }
    site_basis.append(basis)
    site_ops.append(ops)

driver.ghamil = driver.get_custom_hamiltonian(site_basis, site_ops)
b = driver.expr_builder()

# elec one-body terms
for i in range(N_SITES_ELEC):
    for j in range(N_SITES_ELEC):
        if abs(h1e[i, j]) > 1e-15:
            b.add_term("cd", [i, j], h1e[i, j])
            b.add_term("CD", [i, j], h1e[i, j])

# elec two-body terms
for i in range(N_SITES_ELEC):
    for j in range(N_SITES_ELEC):
        for k in range(N_SITES_ELEC):
            for l in range(N_SITES_ELEC):
                if abs(g2e[i, j, k, l]) > 1e-15:
                    coef = 0.5 * g2e[i, j, k, l]
                    # alpha-alpha
                    b.add_term("ccdd", [i, k, l, j], coef)
                    # beta-beta
                    b.add_term("CCDD", [i, k, l, j], coef)
                    # alpha-beta
                    b.add_term("cCDd", [i, k, l, j], coef)
                    # beta-alpha
                    b.add_term("CcdD", [i, k, l, j], coef)

# elec-phonon terms
for i in range(N_SITES_ELEC):
    for j in range(N_SITES_ELEC):
        if abs(dpq[i, j]) > 1e-15:
            coef = -1.0 * np.sqrt(OMEGA / 2.0) * dpq[i, j]
            b.add_term("cdE", [i, j, N_SITES_ELEC], coef)
            b.add_term("CDE", [i, j, N_SITES_ELEC], coef)
            b.add_term("cdF", [i, j, N_SITES_ELEC], coef)
            b.add_term("CDF", [i, j, N_SITES_ELEC], coef)

# phonon terms
b.add_term("EF", [N_SITES_ELEC, N_SITES_ELEC], OMEGA)
b.add_term("E", [N_SITES_ELEC], -1.0 * np.sqrt(OMEGA / 2.0) * de)
b.add_term("F", [N_SITES_ELEC], -1.0 * np.sqrt(OMEGA / 2.0) * de)

mpo = driver.get_mpo(
    b.finalize(adjust_order=True, fermionic_ops="cdCD"),
    algo_type=MPOAlgorithmTypes.FastBipartite,
)
mps = driver.get_random_mps(tag="KET", bond_dim=250, nroots=7)
energy = driver.dmrg(
    mpo,
    mps,
    n_sweeps=10,
    bond_dims=bond_dims,
    noises=noises,
    thrds=thrds,
    dav_max_iter=30,
    iprint=1,
)
for i in range(7):
    print(f"DMRG energy = {energy[i] + 0.5 * de**2 + ecore:.16f}")
