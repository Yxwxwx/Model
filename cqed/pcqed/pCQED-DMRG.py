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
# lams = [np.array([0.0, y, 0.0]) for y in np.arange(0, 0.06, 0.01)]
lams = [np.array([0.0, 0.05, 0.0])]
# act_idx
act_idx = [ix - 1 for ix in [27, 31, 32, 33, 34, 35, 36, 37, 41, 48]]

# DMRG params
# DMRGDriver
driver = DMRGDriver(
    scratch="/nvme/Yxwxwx/" + model, n_threads=4, symm_type=SymmetryTypes.SZ
)
bond_dims = [500] * 5 + [1000] * 5
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

with open(model + "_avas.molden", "w") as f:
    molden.header(mol, f)
    molden.orbital_coeff(mol, f, mf.mo_coeff, ene=mf.mo_energy, occ=mf.mo_occ)

for lam in lams:
    print("lam=", lam)
    # Build Hamiltonian
    ncas, n_elec, spin, ecore, h1e, g2e, orb_sym = itg.get_rhf_integrals(
        mf, ncore, ncas, pg_symm=False
    )
    mo_cas = mf.mo_coeff[:, ncore : ncore + ncas]
    ao_dipole = -1.0 * mol.intor("int1e_r", comp=3)
    mu_nuc_val = np.einsum(
        "i,ix->x", mol.atom_charges(), mol.atom_coords(), optimize=True
    )
    l_dot_mu_nuc = np.einsum("x,x->", lam, mu_nuc_val, optimize=True)

    # second term in h1e
    d = np.einsum(
        "ip, xij, jq -> xpq",
        mo_cas,
        ao_dipole,
        mo_cas,
        optimize=True,
    )
    dpq = np.einsum(
        "x, xpq -> pq",
        lam,
        d,
        optimize=True,
    )
    de = l_dot_mu_nuc - np.einsum(
        "x,x->",
        lam,
        2.0 * np.einsum("xij,ji->x", ao_dipole, mf.make_rdm1(), optimize=True)
        + mu_nuc_val,
        optimize=True,
    )
    de_dpq = de * dpq

    # third term in h1e
    q = np.einsum(
        "ip, xyij, jq->xypq",
        mo_cas,
        -1.0 * mol.intor("int1e_rr").reshape(3, 3, mol.nao, mol.nao),
        mo_cas,
        optimize=True,
    )
    qpq = (
        lam[0] ** 2 * q[0, 0]
        + lam[1] ** 2 * q[1, 1]
        + lam[2] ** 2 * q[2, 2]
        + 2.0 * lam[0] * lam[1] * q[0, 1]
        + 2.0 * lam[0] * lam[2] * q[0, 2]
        + 2.0 * lam[1] * lam[2] * q[1, 2]
    )
    h1e = h1e + de_dpq - 0.5 * qpq
    # second term in g2e
    dd = np.einsum("pq, rs->pqrs", dpq, dpq, optimize=True)
    g2e = g2e + dd

    # reorder
    idx = driver.orbital_reordering(np.abs(h1e), np.abs(g2e))
    print("reordering = ", idx)
    h1e = h1e[idx][:, idx]
    dpq = dpq[idx][:, idx]
    g2e = g2e[idx][:, idx][:, :, idx][:, :, :, idx]

    # The constant terms in CQED Hamiltonian
    N_SITES_ELEC, N_SITES_PH = ncas, 1  # only one phonon mode
    L = N_SITES_ELEC + N_SITES_PH
    N_ELEC, N_PH = n_elec, 7  # 6 phonons
    OMEGA = 0.160984

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
    mps = driver.get_random_mps(tag="KET", bond_dim=250, nroots=5)
    energies = driver.dmrg(
        mpo,
        mps,
        n_sweeps=40,
        bond_dims=bond_dims,
        noises=noises,
        thrds=thrds,
        iprint=2,
    )

    energies = [e + 0.5 * de**2 + ecore for e in energies]
    skets = []
    for i in range(len(energies)):
        sket = driver.split_mps(mps, i, "SKET%d" % i)
        # print("split mps = ", i)
        skets.append(sket)

    # <S^2> MPO
    b2 = driver.expr_builder()
    ix1 = np.mgrid[:N_SITES_ELEC].ravel()
    ix2 = np.mgrid[:N_SITES_ELEC, :N_SITES_ELEC].reshape((2, -1))
    b2.add_terms("cd", 0.75 * np.ones(ix1.shape[0]), np.array([ix1, ix1]).T)
    b2.add_terms("CD", 0.75 * np.ones(ix1.shape[0]), np.array([ix1, ix1]).T)
    b2.add_terms(
        "ccdd",
        0.25 * np.ones(ix2.shape[1]),
        np.array([ix2[0], ix2[1], ix2[1], ix2[0]]).T,
    )
    b2.add_terms(
        "cCDd",
        -0.25 * np.ones(ix2.shape[1]),
        np.array([ix2[0], ix2[1], ix2[1], ix2[0]]).T,
    )
    b2.add_terms(
        "CcdD",
        -0.25 * np.ones(ix2.shape[1]),
        np.array([ix2[0], ix2[1], ix2[1], ix2[0]]).T,
    )
    b2.add_terms(
        "CCDD",
        0.25 * np.ones(ix2.shape[1]),
        np.array([ix2[0], ix2[1], ix2[1], ix2[0]]).T,
    )
    b2.add_terms(
        "cCDd",
        -0.5 * np.ones(ix2.shape[1]),
        np.array([ix2[0], ix2[1], ix2[0], ix2[1]]).T,
    )
    b2.add_terms(
        "CcdD",
        -0.5 * np.ones(ix2.shape[1]),
        np.array([ix2[1], ix2[0], ix2[1], ix2[0]]).T,
    )

    # ssq_mpo = driver.get_spin_square_mpo(iprint=0)
    ssq_mpo = driver.get_mpo(
        b2.finalize(adjust_order=True, fermionic_ops="cdCD"),
        algo_type=MPOAlgorithmTypes.FastBipartite,
    )

    # phonon number MPO
    b3 = driver.expr_builder()
    b3.add_term("EF", [N_SITES_ELEC, N_SITES_ELEC], 1.0)
    nph_mpo = driver.get_mpo(
        b3.finalize(adjust_order=True, fermionic_ops="cdCD"),
        algo_type=MPOAlgorithmTypes.FastBipartite,
    )

    # print results
    for i in range(len(energies)):
        ssq = driver.expectation(skets[i], ssq_mpo, skets[i])
        nph = driver.expectation(skets[i], nph_mpo, skets[i])
        print(
            f"DMRG energy = {energies[i]:.16f}, Spin square = {ssq:.16f}, Phonon number = {nph:.16f}"
        )
