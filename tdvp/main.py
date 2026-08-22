import numpy as np
import quimb as qu
import quimb.tensor as qtn
import matplotlib.pyplot as plt
from tddmrg.tebd import TEBD

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

print(tebd.err)  #  should be < tol=1e-3)
print(tebd.pt.H @ tebd.pt)

H = qtn.MPO_ham_heis(L)
print("Initial energy:", qtn.expec_TN_1D(psi0.H, H, psi0))
print("Final energy:", qtn.expec_TN_1D(tebd.pt.H, H, tebd.pt))

with plt.style.context(qu.NEUTRAL_STYLE):
    plt.figure(figsize=(12, 7))

    # plot the magnetization
    ax1 = plt.subplot(131)
    plt.pcolormesh(js, ts, np.real(mz_t_j), vmin=-0.5, vmax=0.5)
    plt.set_cmap("RdYlBu")
    plt.colorbar()
    plt.title("Z-Magnetization")
    plt.xlabel("Site")
    plt.ylabel("time [ $Jt$ ]")

    # plot the entropy
    ax2 = plt.subplot(132, sharey=ax1)
    plt.pcolormesh(bs, ts, be_t_b)
    plt.setp(ax2.get_yticklabels(), visible=False)
    plt.set_cmap("viridis"), plt.colorbar()
    plt.title("Block Entropy")
    plt.xlabel("Bond")

    # plot the schmidt gap
    ax3 = plt.subplot(133, sharey=ax1)
    plt.pcolormesh(bs, ts, sg_t_b, vmin=0, vmax=1)
    plt.setp(ax3.get_yticklabels(), visible=False)
    plt.set_cmap("magma_r")
    plt.colorbar()
    plt.title("Schmidt Gap")
    plt.xlabel("Bond")

    plt.savefig("tebd.png")
    plt.close()
