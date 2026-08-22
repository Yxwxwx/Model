import pytest
import quimb.tensor as qtn

from tddmrg.dmrg import DMRG


def test_dmrg():
    # build Hamiltonian
    builder = qtn.SpinHam1D(S=1)
    builder += 1 / 2, "+", "-"
    builder += 1 / 2, "-", "+"
    builder += 1, "Z", "Z"
    H = builder.build_mpo(L=100)

    ref_dmrg = dmrg = qtn.DMRG2(H, bond_dims=[10, 20, 100, 100, 200], cutoffs=1e-10)
    ref_dmrg.solve(tol=1e-6, verbosity=1)

    my_dmrg = DMRG(H, bond_dims=[10, 20, 100, 100, 200], cutoffs=1e-10)
    my_dmrg.solve(tol=1e-6)

    assert abs(my_dmrg.energy - ref_dmrg.energy) < 1e-6
