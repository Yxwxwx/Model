from pyscf import scf, gto
import pickle
import numpy as np
import os

from pyblock2._pyscf import mcscf as b2mcscf

title = "DyCl6"

b = 2.72

coords = [
    ["Dy", [0, 0, 0]],
    ["Cl", [-b, 0, 0]],
    ["Cl", [b, 0, 0]],
    ["Cl", [0, -b, 0]],
    ["Cl", [0, b, 0]],
    ["Cl", [0, 0, -b]],
    ["Cl", [0, 0, b]],
]

mol = gto.M(
    atom=coords,
    basis={"Dy": "ano@7s6p4d2f", "Cl": "ano@4s3p"},
    verbose=4,
    spin=5,
    charge=-3,
    max_memory=10000,
)
print("basis = dz nelec = %d nao = %d" % (mol.nelectron, mol.nao))

mf = scf.UKS(mol).x2c()
mf.xc = "bp86"
mf.conv_tol = 1e-10
mf.max_cycle = 500
mf.diis_space = 15
mf.init_guess = "atom"
mf.chkfile = title + ".chk"
if os.path.exists(mf.chkfile):
    mf.init_guess = "chk"

mf.kernel()

dmao = np.einsum("yij->ij", mf.make_rdm1(), optimize=True)


from lo import get_uno, select_active_space, sort_orbitals

lo_coeff, lo_occ, lo_energy = get_uno(mf)
selected = select_active_space(mol, lo_coeff, lo_occ, ao_labels=["Dy 4f"], iprint=1)

lo_coeff, lo_occ, lo_energy, nactorb, nactelec = sort_orbitals(
    mol,
    lo_coeff,
    lo_occ,
    lo_energy,
    cas_list=selected,
    do_loc=True,
    split_low=0.1,
    split_high=1.9,
)

from pyscf.tools import molden

with open(title + "_lo.molden", "w") as f:
    molden.header(mol, f)
    molden.orbital_coeff(mol, f, lo_coeff, ene=lo_energy, occ=lo_occ)

mf = scf.RHF(mol).x2c()
mf.mo_coeff = lo_coeff
mf.mo_occ = np.array([int(np.round(x) + 0.1) for x in lo_occ])
mf.mo_energy = lo_energy

assert sum(mf.mo_occ) == mol.nelectron

from pyscf import mcscf

mc = mcscf.CASSCF(mf, nactorb, nactelec)
ncore = mc.ncore
ncas = mc.ncas

from integral_helper import get_rhf_somf_integrals

ncas, n_elec, spin, ecore, h1e, g2e, orb_sym = get_rhf_somf_integrals(
    mf, ncore, ncas, dmao=dmao
)
from pyblock2.driver.core import DMRGDriver, SymmetryTypes

driver = DMRGDriver(scratch="./tmp", symm_type=SymmetryTypes.SGFCPX, n_threads=2)
driver.read_fcidump(filename="DyCl6_FCIDUMP")
idx = driver.orbital_reordering(np.abs(h1e), np.abs(g2e), method="fiedler")
print("idx = ", idx)

# assert np.allclose(h1e, h1e.conj().T)
# assert all(
#     np.allclose(g2e, x)
#     for x in [
#         g2e,
#         g2e.transpose(1, 0, 2, 3),
#         g2e.transpose(0, 1, 3, 2),
#         g2e.transpose(1, 0, 3, 2),
#         g2e.transpose(2, 3, 0, 1),
#         g2e.transpose(3, 2, 0, 1),
#         g2e.transpose(2, 3, 1, 0),
#         g2e.transpose(3, 2, 1, 0),
#     ]
# )


# from tools import dump_cpx_FCIDUMP

# assert ncas == 14
# dump_cpx_FCIDUMP(
#     title + "_FCIDUMP",
#     ncas,
#     n_elec,
#     spin,
#     1,
#     orb_sym,
#     ecore,
#     h1e,
#     g2e,
# )
