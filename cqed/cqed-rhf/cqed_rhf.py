import numpy as np


def nuclear_dipole(mol):
    r"""
    #\mu_{nuc} = \sum_A Z_A*R_A
    """
    charges = mol.atom_charges()
    coords = mol.atom_coords()
    return np.einsum("i,ij->j", charges, coords, optimize=True)


def mu_exp(ao_dipole, nuclear_dipole, dm):
    r"""
    # <\mu> = \sum_i^N_{occ} <i|\mu|i> + \mu_{nuc}
    """
    mu_elec = np.einsum("xij,ji->x", ao_dipole, dm, optimize=True)
    return nuclear_dipole + mu_elec


def cqed_h1e(int1e, ao_quadrupole, ao_dipole, mu_nuc_val, mu_exp_val, lambda_vec):
    r"""
    # H_{ij} = H_{ij}
    #          -1/2 * \sum_{\zeta, \zeta'} \lambda^{\zeta} * \lambda^{\zeta'} * Q_{ij}^{\zeta\zeta'}
    #          + \sum_{\zeta} \lambda_{\zeta} * \mu_{ij}^{\zeta} * (\lambda \dot \mu_{nuc} - \lambda \dot <\mu>)
    """
    Qxx, Qxy, Qxz, Qyy, Qyz, Qzz = ao_quadrupole
    Qij = -0.5 * lambda_vec[0] * lambda_vec[0] * Qxx
    Qij -= 0.5 * lambda_vec[1] * lambda_vec[1] * Qyy
    Qij -= 0.5 * lambda_vec[2] * lambda_vec[2] * Qzz
    Qij -= lambda_vec[0] * lambda_vec[1] * Qxy
    Qij -= lambda_vec[0] * lambda_vec[2] * Qxz
    Qij -= lambda_vec[1] * lambda_vec[2] * Qyz

    dij = (
        np.einsum("x,x->", lambda_vec, mu_nuc_val, optimize=True)
        - np.einsum("x,x->", lambda_vec, mu_exp_val, optimize=True)
    ) * np.einsum("x, xij->ij", lambda_vec, ao_dipole, optimize=True)

    return int1e + Qij + dij


def cqed_g(int2e, ao_dipole, lambda_vec, dm):
    r"""
    # G_{ij} = \sum_{kl} (2(ij|kl) - (ik|jl)) * D_{kl} +
    # \sum_{\zeta, \zeta'}\lambda^{\zeta} * \lambda^{\zeta'} *
    # (2\mu_{ij}^{\zeta}\mu_{kl}^{\zeta'} - \mu_{ik}^{\zeta}\mu_{jl}^{\zeta'}) * D_{kl}
    """
    gij = (
        np.einsum("ijkl,kl->ij", int2e, dm, optimize=True)
        - 0.5 * np.einsum("ikjl,kl->ij", int2e, dm, optimize=True)
        + np.einsum(
            "x, y, xij, ykl, kl->ij",
            lambda_vec,
            lambda_vec,
            ao_dipole,
            ao_dipole,
            dm,
            optimize=True,
        )
        - 0.5
        * np.einsum(
            "x, y, xik, yjl, kl->ij",
            lambda_vec,
            lambda_vec,
            ao_dipole,
            ao_dipole,
            dm,
            optimize=True,
        )
    )
    return gij


def cqed_fock(h1e, g):
    r"""
    # F = H + G
    """
    return h1e + g


def cqed_dc(lambda_vec, nuclear_dipole, mu_exp_val):
    r"""
    d_c = 0.5 * (\lambda \dot \mu_{nuc}) ** 2 - (\lambda \dot <\mu>)(\lambda \dot \mu_{nuc}) + 0.5 * (\lambda \dot <\mu>)**2
    """
    l_dot_nuc = np.einsum("x,x", lambda_vec, nuclear_dipole, optimize=True)
    l_dot_mu = np.einsum("x,x", lambda_vec, mu_exp_val, optimize=True)
    return 0.5 * l_dot_nuc**2 - l_dot_mu * l_dot_nuc + 0.5 * l_dot_mu**2


def cqed_energy(H, F, dm, dc_val):
    r"""
    E = np.einsum("ij,ij->", F + H, dm, optimize=True) + dc_val
    """
    return 0.5 * np.einsum("ij,ij->", F + H, dm, optimize=True) + dc_val


if __name__ == "__main__":
    from pyscf import gto, scf

    mol = gto.M(
        atom="""
    O      0.000000000000   0.000000000000  -0.068516219320
    H      0.000000000000  -0.790689573744   0.543701060715
    H      0.000000000000   0.790689573744   0.543701060715
                """,
        basis="cc-pvdz",
        max_memory=2000,
        verbose=4,
    )
    mf = scf.RHF(mol)
    mf.kernel()

    cqed_hf = mf
    cqed_hf.diis_space = 0
    lambda_vec = np.array([0.0, 0.0, 0.05])

    h1e = mf.get_hcore()
    mu_nuc_val = nuclear_dipole(mol)
    ao_quadrupole = mol.intor_symmetric("int1e_rr_sph", comp=6)
    ao_dipole = mol.intor_symmetric("int1e_r_sph", comp=3)
    cqed_hf._eri = mol.intor("int2e_sph", aosym="s1")

    cqed_hf.get_hcore = lambda *args: cqed_h1e(
        h1e,
        ao_quadrupole,
        ao_dipole,
        mu_nuc_val,
        mu_exp(ao_dipole, mu_nuc_val, cqed_hf.make_rdm1()),
        lambda_vec,
    )
    cqed_hf.get_veff = lambda *args: cqed_g(
        cqed_hf._eri,
        ao_dipole,
        lambda_vec,
        cqed_hf.make_rdm1(),
    )

    def get_fock_override(hf_obj, *args, **kwargs):
        h1e = hf_obj.get_hcore()
        g = hf_obj.get_veff()
        return cqed_fock(h1e, g)

    import types

    cqed_hf.get_fock = types.MethodType(get_fock_override, cqed_hf)

    cqed_hf.energy_elec = lambda *args: cqed_energy(
        cqed_hf.get_hcore(),
        cqed_hf.get_fock(),
        cqed_hf.make_rdm1(),
        cqed_dc(
            lambda_vec, mu_nuc_val, mu_exp(ao_dipole, mu_nuc_val, cqed_hf.make_rdm1())
        ),
    )
    cqed_hf.energy_tot = lambda *args: cqed_hf.energy_elec() + cqed_hf.energy_nuc()

    cqed_hf.kernel()
