import numpy as np
import pytest
import quimb.tensor as qtn

from tddmrg.tebd import TEBD


def test_tebd():
    L = 44
    zeros = "0" * ((L - 2) // 3)
    binary = zeros + "1" + zeros + "1" + zeros
    print("psi0:", f"|{binary}>")

    psi0 = qtn.MPS_computational_state(binary)

    H = qtn.ham_1d_heis(L)

    tebd = TEBD(psi0, H)

    # Since entanglement will not grow too much, we can set quite
    #     a small cutoff for splitting after each gate application
    tebd.split_opts["cutoff"] = 1e-12

    # times we are interested in
    ts = np.linspace(0, 80, 101)

    mz_t_j = []  # z-magnetization
    be_t_b = []  # block entropy
    sg_t_b = []  # schmidt gap

    # range of bonds, and sites
    js = np.arange(0, L)
    bs = np.arange(1, L)

    # generate the state at each time in ts
    #     and target error 1e-3 for whole evolution
    for psit in tebd.at_times(ts, tol=1e-3):
        mz_j = []
        be_b = []
        sg_b = []

        # there is one more site than bond, so start with mag
        #     this also sets the orthog center to 0, which ``info`` then tracks
        info = {"cur_orthog": None}
        mz_j += [psit.magnetization(0, info=info)]

        for j in range(1, L):
            mz_j += [psit.magnetization(j, info=info)]
            be_b += [psit.entropy(j, info=info)]
            sg_b += [psit.schmidt_gap(j, info=info)]

        mz_t_j += [mz_j]
        be_t_b += [be_b]
        sg_t_b += [sg_b]
