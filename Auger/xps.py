import numpy as np
from pyscf import gto, scf
from pyscf.data.nist import HARTREE2EV
from pyblock2._pyscf.ao2mo import integrals as itg
from pyblock2.driver.core import DMRGDriver, SymmetryTypes
from ras import get_random_restricted_mps

# patch
DMRGDriver.get_random_restricted_mps = get_random_restricted_mps
bond_dims = [250] * 4 + [500] * 4 + [1000]
noises = [1e-4] * 4 + [1e-5] * 4 + [0]
thrds = [1e-10] * 8

# initial state Ne+
print("===> initial state Ne+ <===")
NeI = gto.M(atom="Ne 0 0 0", basis="aug-cc-pVTZ", charge=1, spin=1)
mf = scf.ROHF(NeI).run()

ncas, n_elec, spin, ecore, h1e, g2e, orb_sym = itg.get_rhf_integrals(
    mf, ncore=0, ncas=None, g2e_symm=1
)
print(f"ncas = {ncas}, n_elec = {n_elec}, spin = {spin}")
driver = DMRGDriver(
    scratch="/nvme/Yxwxwx/xps/NeI",
    symm_type=SymmetryTypes.SZ,
    n_threads=32,
    stack_mem=400 << 30,
)
driver.initialize_system(n_sites=ncas, n_elec=n_elec, spin=spin, orb_sym=orb_sym)
mpo = driver.get_qc_mpo(h1e=h1e, g2e=g2e, ecore=ecore, iprint=1)
ket = driver.get_random_restricted_mps(
    tag="GS",
    bond_dim=250,
    nroots=1,
    n_hole=1,
    n_core=1,
    n_inact=0,
    n_exter=0,
    n_act=ncas - 1,
    core_sz=1,  # 【关键设定】=1 代表核心保留 1 个 alpha 电子；=-1 代表保留 beta 电子
)
energy1 = driver.dmrg(
    mpo, ket, n_sweeps=20, bond_dims=bond_dims, noises=noises, thrds=thrds, iprint=1
)
print("DMRG energy (Ne+ Core-Hole State) = %20.15f" % energy1)

# 打印对角线验证
rdm1 = driver.get_1pdm(ket)
rdm1 = rdm1[0] + rdm1[1]
print("Ne+ Core-Hole State:", rdm1.diagonal())

# Auger State Ne2+
print("===> initial state Ne2+ <===")
NeII = gto.M(atom="Ne 0 0 0", basis="aug-cc-pVTZ", charge=2, spin=2)
mf = scf.ROHF(NeII).run()
ncas, n_elec, spin, ecore, h1e, g2e, orb_sym = itg.get_rhf_integrals(
    mf, ncore=0, ncas=None, g2e_symm=1
)
print(f"ncas = {ncas}, n_elec = {n_elec}, spin = {spin}")
driver = DMRGDriver(
    scratch="/nvme/Yxwxwx/xps/NeII",
    symm_type=SymmetryTypes.SZ,
    n_threads=32,
    stack_mem=400 << 30,
)
driver.initialize_system(n_sites=ncas, n_elec=n_elec, spin=spin, orb_sym=orb_sym)
mpo = driver.get_qc_mpo(h1e=h1e, g2e=g2e, ecore=ecore, iprint=1)
ket = driver.get_random_restricted_mps(
    tag="GS",
    bond_dim=250,
    nroots=1,
    n_hole=0,
    n_core=0,
    n_inact=0,
    n_exter=0,
    n_act=ncas,
    core_sz=0,
)

energy2 = driver.dmrg(
    mpo, ket, n_sweeps=20, bond_dims=bond_dims, noises=noises, thrds=thrds, iprint=1
)
print("DMRG energy (Ne+ Core-Hole State) = %20.15f" % energy2)

# 打印对角线验证
rdm1 = driver.get_1pdm(ket)
rdm1 = rdm1[0] + rdm1[1]
print("Ne2+ Core-Hole State:", rdm1.diagonal())

print("Auger electron energy:", (energy1 - energy2) * HARTREE2EV)
