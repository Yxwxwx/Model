from pyscf import gto, scf, mcscf
from pyscf.lib import logger
from pyscf.tools import molden
from pyblock2.driver.core import DMRGDriver, SymmetryTypes, MPOAlgorithmTypes
from pyblock2._pyscf import mcscf as b2mcscf
from pyblock2._pyscf import scf as b2scf
from pyblock2._pyscf.ao2mo import integrals as itg
import numpy as np
import os

# const
model = "Naphtalene"
lams = [np.array([0.0, 0.05, 0.0])]
# act_idx
act_idx = [ix - 1 for ix in [27, 31, 32, 33, 34, 35, 36, 37, 41, 48]]
nroots = 50
# DMRG params
# DMRGDriver
driver = DMRGDriver(
    scratch="/nvme/Yxwxwx/pcqed/" + model, n_threads=16, symm_type=SymmetryTypes.SU2
)
bond_dims = [250] * 4 + [500] * 5 + [1000]
noises = [1e-5] * 5 + [1e-6] * 5 + [0]
thrds = [1e-5] * 5 + [1e-7] * 5 + [1e-8]

# molecular
mol = gto.M(
    atom="""
 C                  1.24488419    1.40227290    0.00000000
 H                 -1.24259572    2.48937110    0.00000000
 H                 -3.37724803   -1.24575773    0.00000000
 H                 -1.24259572   -2.48937110    0.00000000
 C                 -2.43345098   -0.70828680    0.00000000
 C                 -2.43345098    0.70828680   -0.00000000
 C                 -0.00000000    0.71720500    0.00000000
 H                  3.37724803    1.24575773    0.00000000
 H                 -3.37724803    1.24575773   -0.00000000
 C                  0.00000000   -0.71720500    0.00000000
 C                 -1.24488419   -1.40227290    0.00000000
 H                  3.37724803   -1.24575773   -0.00000000
 C                 -1.24488419    1.40227290    0.00000000
 C                  1.24488419   -1.40227290    0.00000000
 H                  1.24259572   -2.48937110    0.00000000
 H                  1.24259572    2.48937110   -0.00000000
 C                  2.43345098    0.70828680    0.00000000
 C                  2.43345098   -0.70828680   -0.00000000
        """,
    basis="6-31g*",
    spin=0,
    verbose=4,
)
mol.set_common_orig((0.0, 0.0, 0.0))  # For dipole

mf = scf.RHF(mol)
mf.chkfile = model + "_rhf.h5"
if os.path.exists(mf.chkfile):
    mf.init_guess = "chk"
mf.run()


# Stble=OPT
def stable_opt_internal(mf):
    log = logger.new_logger(mf)
    mo1, _, stable, _ = mf.stability(return_status=True)
    cyc = 0
    while not stable and cyc < 10:
        log.note("Try to optimize orbitals until stable, attempt %d" % cyc)
        dm1 = mf.make_rdm1(mo1, mf.mo_occ)
        mf = mf.run(dm1)
        mo1, _, stable, _ = mf.stability(return_status=True)
        cyc += 1
    if not stable:
        log.note("Stability Opt failed after %d attempts" % cyc)
    return mf


print("Stabe=Opt")
# mf = stable_opt_internal(mf)
with open(model + "_rhf.molden", "w") as f:
    molden.header(mol, f)
    molden.orbital_coeff(mol, f, mf.mo_coeff, ene=mf.mo_energy, occ=mf.mo_occ)

lo_coeff, lo_occ, lo_energy, nactorb, nactelec = b2mcscf.sort_orbitals(
    mol,
    mf.mo_coeff,
    mf.mo_occ,
    mf.mo_energy,
    cas_list=act_idx,
    do_loc=True,
    split_low=0.1,
    split_high=1.9,
)

# b2scf.mulliken_pop_dmao(mol, mf.make_rdm1())

assert nactorb == nactelec == 10
mf.mo_coeff = lo_coeff
mf.mo_occ = lo_occ
mf.mo_energy = lo_energy

mc = mcscf.CASSCF(mf, nactorb, nactelec)
ncore = mc.ncore
ncas = nactorb
n_elec = nactelec
print("ncore=", ncore, "ncas=", ncas, "n_elec=", n_elec)

with open(model + "_cas.molden", "w") as f:
    molden.header(mol, f)
    molden.orbital_coeff(mol, f, mf.mo_coeff, ene=mf.mo_energy, occ=mf.mo_occ)

ncas, n_elec, spin, ecore, h1e, g2e, orb_sym = itg.get_rhf_integrals(
    mf, ncore, ncas, pg_symm=False
)

# fielder reorder
idx = driver.orbital_reordering(np.abs(h1e), np.abs(g2e))
print("reordering = ", idx)
h1e = h1e[np.ix_(idx, idx)]
g2e = g2e[np.ix_(idx, idx, idx, idx)]

driver.initialize_system(
    n_sites=ncas, n_elec=n_elec, spin=spin, orb_sym=orb_sym, singlet_embedding=False
)

mpo = driver.get_qc_mpo(h1e=h1e, g2e=g2e, ecore=ecore, iprint=1)
ket = driver.get_random_mps(tag="KET", bond_dim=250, nroots=nroots)

energies = driver.dmrg(
    mpo,
    ket,
    n_sweeps=20,
    bond_dims=bond_dims,
    noises=noises,
    thrds=thrds,
    iprint=1,
    cutoff=1e-24,
)

assert len(energies) == nroots
np.save(model + "_energies.npy", energies)

print("DMRG energies:\n", energies)

skets = []
for i in range(len(energies)):
    sket = driver.split_mps(ket, i, "SKET%d" % i)
    print("split mps = ", i)
    skets.append(sket)

driver.reorder_idx = idx

# dipole moment in MO
dipole = mol.intor("cint1e_r_sph", comp=3)

dip_nuc = np.einsum("i,ix->x", mol.atom_charges(), mol.atom_coords())
dm_core = mf.make_rdm1(mf.mo_coeff[:, :ncore], mf.mo_occ[:ncore])
dip_core = np.einsum("xij,ji->x", dipole, dm_core)
dip_const = dip_nuc - dip_core

dip_cas = np.einsum(
    "rij,ip,jq->rpq",
    dipole,
    mf.mo_coeff[:, ncore : ncore + ncas],
    mf.mo_coeff[:, ncore : ncore + ncas],
    optimize=True,
)


dip_mat = np.zeros((3, nroots, nroots))
for i in range(nroots):
    for j in range(i, nroots):
        rdm1 = driver.get_trans_1pdm(skets[i], skets[j])

        for xyz in range(3):
            # -( Tr(R * rdm) + closedDip )
            elec_part = -(np.einsum("ij,ij->", dip_cas[xyz], rdm1) + dip_core[xyz])

            total_dip = elec_part
            if i == j:
                # transDip += nuclear_dipole_
                total_dip += dip_nuc[xyz]

            dip_mat[xyz, i, j] = total_dip
            dip_mat[xyz, j, i] = total_dip


np.save(model + "_dipole.npy", dip_mat)
