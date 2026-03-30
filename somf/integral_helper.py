import numpy as np
import copy


def _get_jk_incore(mol, dm0, hso2e=None):
    if hso2e is None:
        hso2e = mol.intor("int2e_p1vxp1", 3).reshape(
            3, mol.nao, mol.nao, mol.nao, mol.nao
        )

    vj = np.einsum("yijkl,lk->yij", hso2e, dm0, optimize=True)
    vk1 = np.einsum("yijkl,jk->yil", hso2e, dm0, optimize=True)
    vk2 = np.einsum("yijkl,li->ykj", hso2e, dm0, optimize=True)
    return vj, vk1, vk2


def _get_jk_direct(mol, dm0):
    from pyscf import scf

    vj, vk1, vk2 = scf.jk.get_jk(
        mol,
        [dm0, dm0, dm0],
        ["ijkl,kl->ij", "ijkl,jk->il", "ijkl,li->kj"],
        intor="int2e_p1vxp1",
        comp=3,
    )
    return vj, vk1, vk2


def get_jk(mol, dm0, hso2e=None, direct=False):
    """Direct or incore evaluation of AMFI integrals"""
    if direct:
        return _get_jk_direct(mol, dm0)
    else:
        return _get_jk_incore(mol, dm0, hso2e)


def get_jk_amfi(mol, dm0, hso2e=None):
    """Atomic-mean-field approximation"""
    ao_loc = mol.ao_loc_nr()
    nao = ao_loc[-1]
    vj = np.zeros((3, nao, nao))
    vk1 = np.zeros((3, nao, nao))
    vk2 = np.zeros((3, nao, nao))
    atom = copy.copy(mol)
    aoslices = mol.aoslice_by_atom(ao_loc)
    for ia in range(mol.natm):
        b0, b1, p0, p1 = aoslices[ia]
        atom._bas = mol._bas[b0:b1]
        vj1p, vk1p, vk2p = get_jk(atom, dm0[p0:p1, p0:p1])
        vj[:, p0:p1, p0:p1] = vj1p
        vk1[:, p0:p1, p0:p1] = vk1p
        vk2[:, p0:p1, p0:p1] = vk2p

    return vj, vk1, vk2


def get_hso_ao(mol, dm0, qed_fac=1, amfi=False):
    from pyscf.data import nist

    alpha2 = nist.ALPHA**2
    hso1e = mol.intor_asymmetric("int1e_pnucxp", 3)
    uhf = True if len(dm0) == 2 else False
    rhf = not uhf
    if rhf:
        vj, vk1, vk2 = get_jk_amfi(mol, dm0) if amfi else get_jk(mol, dm0)
        hso2e = vj - (vk1 + vk2) * 1.5
        hso = qed_fac * (alpha2 / 2) * (hso1e + hso2e)
    else:
        nao = dm0[0].shape[0]
        vja, vk1a, vk2a = get_jk_amfi(mol, dm0[0]) if amfi else get_jk(mol, dm0[0])
        vjb, vk1b, vk2b = get_jk_amfi(mol, dm0[1]) if amfi else get_jk(mol, dm0[1])
        vj = vja + vjb
        hso = np.zeros((6, nao, nao))
        for i in range(6):
            hso[i, :, :] += hso1e[i // 2, :, :] + vj[i // 2, :, :]
        hso[0, :, :] -= (vk1b[0, :, :] + vk2a[0, :, :]) * 1.5
        hso[1, :, :] -= (vk1a[0, :, :] + vk2b[0, :, :]) * 1.5
        hso[2, :, :] -= (vk1b[1, :, :] + vk2a[1, :, :]) * 1.5
        hso[3, :, :] -= (vk1a[1, :, :] + vk2b[1, :, :]) * 1.5
        hso[4, :, :] -= (vk1a[2, :, :] + vk2a[2, :, :]) * 1.5
        hso[5, :, :] -= (vk1b[2, :, :] + vk2b[2, :, :]) * 1.5
        hso *= qed_fac * (alpha2 / 2)
    return hso * 1j


def get_rhf_integrals(mf, ncore=0, ncas=None):
    mol = mf.mol
    mo = mf.mo_coeff

    from pyscf import ao2mo

    if ncas is None:
        ncas = mo.shape[1] - ncore

    orb_sym = [0] * ncas

    ecore = mol.energy_nuc()
    mo_core = mo[:, :ncore]
    mo_cas = mo[:, ncore : ncore + ncas]
    hcore_ao = mf.get_hcore()
    hveff_ao = 0

    if ncore != 0:
        core_dmao = 2 * mo_core @ mo_core.T.conj()
        vj, vk = mf.get_jk(mol, core_dmao)
        hveff_ao = vj - 0.5 * vk
        ecore += np.einsum(
            "ij,ji->", core_dmao, hcore_ao + 0.5 * hveff_ao, optimize=True
        )
    h1e = mo_cas.T.conj() @ (hcore_ao + hveff_ao) @ mo_cas
    eri_ao = mol if mf._eri is None else mf._eri
    g2e = ao2mo.full(eri_ao, mo_cas)
    g2e = ao2mo.restore(1, g2e, ncas)

    n_elec = mol.nelectron - 2 * ncore
    spin = mol.spin

    return ncas, n_elec, spin, ecore, h1e, g2e, orb_sym


def get_rhf_somf_integrals(
    mf,
    ncore=0,
    ncas=None,
    dmao=None,
    amfi=True,
):
    from pyscf import scf

    assert isinstance(mf, scf.hf.RHF)

    n_elec, spin, ecore, hsf1e, gsf2e, orb_sym = get_rhf_integrals(mf, ncore, ncas)[1:]

    mol = mf.mol
    mo = mf.mo_coeff
    if ncas is None:
        ncas = mo.shape[1] - ncore

    gh1e = np.zeros((ncas * 2, ncas * 2), dtype=complex)
    gg2e = np.zeros((ncas * 2, ncas * 2, ncas * 2, ncas * 2), dtype=complex)

    for i in range(ncas * 2):
        for j in range(i % 2, ncas * 2, 2):
            gh1e[i, j] = hsf1e[i // 2, j // 2]

    for i in range(ncas * 2):
        for j in range(i % 2, ncas * 2, 2):
            for k in range(ncas * 2):
                for l in range(k % 2, ncas * 2, 2):
                    gg2e[i, j, k, l] = gsf2e[i // 2, j // 2, k // 2, l // 2]

    hsoao = get_hso_ao(mol, dmao, amfi=amfi)
    hso = np.einsum(
        "rij,ip,jq->rpq",
        hsoao,
        mo[:, ncore : ncore + ncas],
        mo[:, ncore : ncore + ncas],
        optimize=True,
    )
    for i in range(ncas * 2):
        for j in range(ncas * 2):
            if i % 2 == 0 and j % 2 == 0:  # aa
                gh1e[i, j] += hso[2, i // 2, j // 2] * 0.5
            elif i % 2 == 1 and j % 2 == 1:  # bb
                gh1e[i, j] -= hso[2, i // 2, j // 2] * 0.5
            elif i % 2 == 0 and j % 2 == 1:  # ab
                gh1e[i, j] += (
                    hso[0, i // 2, j // 2] - hso[1, i // 2, j // 2] * 1j
                ) * 0.5
            elif i % 2 == 1 and j % 2 == 0:  # ba
                gh1e[i, j] += (
                    hso[0, i // 2, j // 2] + hso[1, i // 2, j // 2] * 1j
                ) * 0.5

    assert np.linalg.norm(gh1e - gh1e.T.conj()) < 1e-10

    orb_sym = [orb_sym[x // 2] for x in range(ncas * 2)]

    return ncas * 2, n_elec, spin, ecore, gh1e, gg2e, orb_sym


def get_uhf_integrals(mf, ncore=0, ncas=None):
    mol = mf.mol
    mo_a, mo_b = mf.mo_coeff

    from pyscf import ao2mo

    if ncas is None:
        ncas = mo_a.shape[1] - ncore

    orb_sym = [0] * ncas

    ecore = mol.energy_nuc()
    mo_core = mo_a[:, :ncore], mo_b[:, :ncore]
    mo_cas = mo_a[:, ncore : ncore + ncas], mo_b[:, ncore : ncore + ncas]
    hcore_ao = mf.get_hcore()
    hveff_ao = (0, 0)

    if ncore != 0:
        core_dmao = mo_core[0] @ mo_core[0].T.conj(), mo_core[1] @ mo_core[1].T.conj()
        vj, vk = mf.get_jk(mol, core_dmao)
        hveff_ao = vj[0] + vj[1] - vk
        ecore += np.einsum(
            "ij,ji->", core_dmao[0], hcore_ao + 0.5 * hveff_ao[0], optimize=True
        )
        ecore += np.einsum(
            "ij,ji->", core_dmao[1], hcore_ao + 0.5 * hveff_ao[1], optimize=True
        )
    h1e_a = mo_cas[0].T.conj() @ (hcore_ao + hveff_ao[0]) @ mo_cas[0]
    h1e_b = mo_cas[1].T.conj() @ (hcore_ao + hveff_ao[1]) @ mo_cas[1]

    eri_ao = mol if mf._eri is None else mf._eri
    mo_a, mo_b = mo_cas

    g2e_aa = ao2mo.restore(1, ao2mo.full(eri_ao, mo_a), ncas)
    g2e_ab = ao2mo.restore(
        min(1, 4), ao2mo.general(eri_ao, (mo_a, mo_a, mo_b, mo_b)), ncas
    )
    g2e_bb = ao2mo.restore(1, ao2mo.full(eri_ao, mo_b), ncas)

    n_elec = mol.nelectron - ncore * 2
    spin = mol.spin

    return ncas, n_elec, spin, ecore, (h1e_a, h1e_b), (g2e_aa, g2e_ab, g2e_bb), orb_sym


def get_uhf_somf_integrals(
    mf,
    ncore=0,
    ncas=None,
    dmao=None,
    amfi=True,
):
    from pyscf import scf

    assert isinstance(mf, scf.uhf.UHF)
    n_elec, spin, ecore, hsf1e, gsf2e, orb_sym = get_uhf_integrals(mf, ncore, ncas)[1:]

    mo = mf.mo_coeff
    mol = mf.mol

    if ncas is None:
        ncas = mo[0].shape[1] - ncore
    gh1e = np.zeros((ncas * 2, ncas * 2), dtype=complex)
    gg2e = np.zeros((ncas * 2, ncas * 2, ncas * 2, ncas * 2), dtype=complex)

    for i in range(ncas * 2):
        for j in range(i % 2, ncas * 2, 2):
            if i % 2 == 0:
                gh1e[i, j] = hsf1e[0][i // 2, j // 2]
            else:
                gh1e[i, j] = hsf1e[1][i // 2, j // 2]

    for i in range(ncas * 2):
        for j in range(i % 2, ncas * 2, 2):
            for k in range(ncas * 2):
                for l in range(k % 2, ncas * 2, 2):
                    if i % 2 == 0 and k % 2 == 0:
                        gg2e[i, j, k, l] = gsf2e[0][i // 2, j // 2, k // 2, l // 2]
                    elif i % 2 == 0 and k % 2 != 0:
                        gg2e[i, j, k, l] = gsf2e[1][i // 2, j // 2, k // 2, l // 2]
                    elif i % 2 != 0 and k % 2 == 0:
                        gg2e[i, j, k, l] = gsf2e[1][k // 2, l // 2, i // 2, j // 2]
                    else:
                        gg2e[i, j, k, l] = gsf2e[2][i // 2, j // 2, k // 2, l // 2]

    hsoao = get_hso_ao(mol, dmao, amfi=amfi)

    hso = np.einsum(
        "rij,xip,yjq->xyrpq",
        hsoao,
        np.array(mo)[:, :, ncore : ncore + ncas],
        np.array(mo)[:, :, ncore : ncore + ncas],
    )

    for i in range(ncas * 2):
        for j in range(ncas * 2):
            if i % 2 == 0 and j % 2 == 0:  # aa
                gh1e[i, j] += hso[0, 0, 2, i // 2, j // 2] * 0.5
            elif i % 2 == 1 and j % 2 == 1:  # bb
                gh1e[i, j] -= hso[1, 1, 2, i // 2, j // 2] * 0.5
            elif i % 2 == 0 and j % 2 == 1:  # ab
                gh1e[i, j] += (
                    hso[0, 1, 0, i // 2, j // 2] - hso[0, 1, 1, i // 2, j // 2] * 1j
                ) * 0.5
            elif i % 2 == 1 and j % 2 == 0:  # ba
                gh1e[i, j] += (
                    hso[1, 0, 0, i // 2, j // 2] + hso[1, 0, 1, i // 2, j // 2] * 1j
                ) * 0.5

    assert np.linalg.norm(gh1e - gh1e.T.conj()) < 1e-10

    orb_sym = [orb_sym[x // 2] for x in range(ncas * 2)]

    return ncas * 2, n_elec, spin, ecore, gh1e, gg2e, orb_sym
