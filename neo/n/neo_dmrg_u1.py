import numpy as np
import h5py
from pyblock2.driver.core import DMRGDriver, SymmetryTypes, MPOAlgorithmTypes

# const
bond_dims = [250] * 4 + [500] * 4 + [1000]
noises = [1e-4] * 4 + [1e-5] * 4 + [0]
thrds = [1e-10] * 8

# load integral
filename = "neo_u1.h5"

with h5py.File(filename, "r") as f:
    try:
        ncas_e = f["ncas_e"][()]
        ncas_p = f["ncas_p"][()]
        n_elec = f["n_elec"][()]
        n_proton = f["n_proton"][()]
        spin = f["spin"][()]
        ecore = f["ecore"][()]

        h1e = f["h1e"][...]
        h1p = f["h1p"][...]
        g2ee = f["g2ee"][...]
        g2pp = f["g2pp"][...]
        g2ep = f["g2ep"][...]
    except KeyError as e:
        print(f"Error: Missing expected dataset in HDF5 file: {e}")
        raise
print(
    f"ncas_e: {ncas_e}, ncas_p: {ncas_p} , n_elec: {n_elec} , n_proton: {n_proton}, spin: {spin}, eore: {ecore}"
)

driver = DMRGDriver(
    scratch="/nvme/Yxwxwx/neo_dmrg_u1/",
    symm_type=SymmetryTypes.SAny,
    n_threads=32,
    stack_mem=400 << 30,
)

# Fielder ordering
# ordering separately
idx_e = driver.orbital_reordering(np.abs(h1e), np.abs(g2ee))
idx_p = driver.orbital_reordering(np.abs(h1p), np.abs(g2pp))


# print("idx_e:", idx_e)
# print("idx_p:", idx_p)

# h1e = h1e[np.ix_(idx_e, idx_e)]
# h1p = h1p[np.ix_(idx_p, idx_p)]

# g2ee = g2ee[np.ix_(idx_e, idx_e, idx_e, idx_e)]
# g2pp = g2pp[np.ix_(idx_p, idx_p, idx_p, idx_p)]

# g2ep = g2ep[np.ix_(idx_e, idx_e, idx_p, idx_p)]

# quantum number wrapper (U1 / n_elec, U1 / n_proton)
driver.set_symmetry_groups("U1Fermi", "U1Fermi")
Q = driver.bw.SX

# [Part A] Set states and matrix representation of operators in local Hilbert space
site_basis, site_ops = [], []
L = ncas_e + ncas_p
for k in range(L):
    if k < ncas_e:
        # elec ->
        basis = [
            (Q(0, 0), 1),
            (Q(1, 0), 1),
        ]
        ops = {
            "": np.eye(2),  # identity
            "c": np.array([[0, 0], [1, 0]]),  # e+
            "d": np.array([[0, 1], [0, 0]]),  # e
        }
    else:
        # proton ->
        basis = [
            (Q(0, 0), 1),
            (Q(0, 1), 1),
        ]
        ops = {
            "": np.eye(2),  # identity
            "C": np.array([[0, 0], [1, 0]]),  # p+
            "D": np.array([[0, 1], [0, 0]]),  # p
        }

    site_ops.append(ops)
    site_basis.append(basis)

# [Part B] Set Hamiltonian terms in NEO model
driver.initialize_system(
    n_sites=L, vacuum=Q(0, 0), target=Q(n_elec, n_proton), hamil_init=False
)
driver.ghamil = driver.get_custom_hamiltonian(site_basis, site_ops)
b = driver.expr_builder()

# 1. electron-electron interaction
for i in range(ncas_e):
    for j in range(ncas_e):
        if abs(h1e[i, j]) > 1e-12:
            b.add_term("cd", [i, j], h1e[i, j])

for i in range(ncas_e):
    for j in range(ncas_e):
        for k in range(ncas_e):
            for l in range(ncas_e):
                if abs(g2ee[i, j, k, l]) > 1e-12:
                    b.add_term("ccdd", [i, k, l, j], 0.5 * g2ee[i, j, k, l])

# 2. proton-proton interaction
for p in range(ncas_p):
    for q in range(ncas_p):
        if abs(h1p[p, q]) > 1e-12:
            idx = [p + ncas_e, q + ncas_e]
            b.add_term("CD", idx, h1p[p, q])

for p in range(ncas_p):
    for q in range(ncas_p):
        for r in range(ncas_p):
            for s in range(ncas_p):
                if abs(g2pp[p, q, r, s]) > 1e-12:
                    idx = [p + ncas_e, r + ncas_e, s + ncas_e, q + ncas_e]
                    b.add_term("CCDD", idx, 0.5 * g2pp[p, q, r, s])

# 3. electron-proton interaction
for i in range(ncas_e):
    for j in range(ncas_e):
        for p in range(ncas_p):
            for q in range(ncas_p):
                if abs(g2ep[i, j, p, q]) > 1e-12:
                    idx = [i, j, p + ncas_e, q + ncas_e]
                    b.add_term("cdCD", idx, -1.0 * g2ep[i, j, p, q])

# [Part C] Perform DMRG
mpo = driver.get_mpo(
    b.finalize(adjust_order=True),
    algo_type=MPOAlgorithmTypes.FastBipartite,
    iprint=1,
)
mps = driver.get_random_mps(tag="KET", bond_dim=250, nroots=1)
# mps = driver.load_mps(tag="KET", nroots=1)
# sweep_start = 20
# forward = sweep_start % 2 == 0

energy = driver.dmrg(
    mpo,
    mps,
    tol=1e-6,
    n_sweeps=20,
    bond_dims=bond_dims,
    noises=noises,
    thrds=thrds,
    iprint=2,
)
print("DMRG energy = %20.15f" % (energy + ecore))  # −93.053328


# TODO:
# [Part D] Plot entropy matrix
def measure_obs(term, sites):
    b_obs = driver.expr_builder()
    b_obs.add_term(term, sites, 1.0)
    mpo_obs = driver.get_mpo(b_obs.finalize(), iprint=0)
    return driver.expectation(mps, mpo_obs, mps)


# 1. 1/2-RDM
rdm1 = np.zeros((L, L))
nn = np.zeros((L, L))

for i in range(L):
    for j in range(i, L):
        # electron (e) or proton (p)
        is_e_i = i < ncas_e
        is_e_j = j < ncas_e

        # === 1-RDM: ===
        if is_e_i and is_e_j:
            # e-e
            rdm1[i, j] = measure_obs("cd", [i, j])
        elif not is_e_i and not is_e_j:
            # p-p
            rdm1[i, j] = measure_obs("CD", [i, j])
        else:
            # e-p no hopping
            rdm1[i, j] = 0.0

        rdm1[j, i] = rdm1[i, j].conj()

        # === nn: <n_i n_j> ===
        if i == j:
            nn[i, i] = np.real(rdm1[i, i])
        else:
            if is_e_i and is_e_j:
                val = measure_obs("cdcd", [i, i, j, j])
            elif not is_e_i and not is_e_j:
                val = measure_obs("CDCD", [i, i, j, j])
            else:
                # e-p
                val = measure_obs("cdCD", [i, i, j, j])

            nn[i, j] = nn[j, i] = np.real(val)

print("occupied number(Electron): \n", rdm1.diagonal()[:ncas_e])
print("occupied number(Proton): \n", rdm1.diagonal()[ncas_e : ncas_e + ncas_p])

rdm_e = rdm1[:ncas_e, :ncas_e]
rdm_p = rdm1[ncas_e:, ncas_e:]
evals_e, evecs_e = np.linalg.eigh(rdm_e)
evals_p, evecs_p = np.linalg.eigh(rdm_p)
occ_e = evals_e[::-1]
occ_p = evals_p[::-1]

print("Natural occupations (Electron): \n", occ_e)
print("Natural occupations (Proton): \n", occ_p)


# 2. ordm1 (Single-orbital entropies)
ordm1 = np.zeros(L)
for i in range(L):
    occ = np.clip(np.real(rdm1[i, i]), 1e-14, 1.0 - 1e-14)
    ordm1[i] = -occ * np.log(occ) - (1 - occ) * np.log(1 - occ)

# 3. ordm2 (Two-orbital entropies)
ordm2 = np.zeros((L, L))
for i in range(L):
    for j in range(L):
        if i == j:
            ordm2[i, j] = ordm1[i]
        else:
            ni, nj, nij = np.real(rdm1[i, i]), np.real(rdm1[j, j]), nn[i, j]
            rho = np.zeros((4, 4), dtype=complex)
            rho[0, 0] = 1.0 - ni - nj + nij  # |00>
            rho[1, 1] = nj - nij  # |01>
            rho[2, 2] = ni - nij  # |10>
            rho[3, 3] = nij  # |11>

            rho[1, 2] = rdm1[j, i]
            rho[2, 1] = rdm1[i, j]

            evals = np.linalg.eigvalsh(rho)
            s2 = -np.sum([v * np.log(v) for v in evals if v > 1e-14])
            ordm2[i, j] = s2

# 4. minfo (Mutual Information)
minfo = 0.5 * (ordm1[:, None] + ordm1[None, :] - ordm2) * (1 - np.identity(L))

import matplotlib.pyplot as plt


def plot_neo_minfo(minfo, ncas_e, ncas_p):
    L = ncas_e + ncas_p
    fig, ax = plt.subplots(figsize=(8, 7), dpi=150)

    # 使用与文献风格接近的 colormap (比如 'viridis' 或 'inferno')
    # 互信息通常是非负的，ocean_r 也可以，但 viridis 更容易看出微小数值
    im = ax.matshow(minfo, cmap="ocean_r")

    # 添加颜色条
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Mutual Information $I_{i,j}$", rotation=270, labelpad=15)

    # --- 核心优化：绘制区分电子和质子的分界线 ---
    # 绘制水平线和垂直线
    line_style = {
        "colors": "red",
        "linewidths": 1.5,
        "linestyles": "--",
        "alpha": 0.8,
    }
    ax.vlines(ncas_e - 0.5, -0.5, L - 0.5, **line_style)
    ax.hlines(ncas_e - 0.5, -0.5, L - 0.5, **line_style)

    # --- 添加区域标注 ---
    # 电子区中心位置
    e_mid = ncas_e / 2 - 0.5
    # 质子区中心位置
    p_mid = ncas_e + ncas_p / 2 - 0.5

    # 设置刻度
    ax.set_xticks([e_mid, p_mid])
    ax.set_xticklabels(["Electrons", "Protons"], fontsize=12, fontweight="bold")
    ax.set_yticks([e_mid, p_mid])
    ax.set_yticklabels(
        ["Electrons", "Protons"],
        fontsize=12,
        fontweight="bold",
        rotation=90,
        va="center",
    )

    # 去掉不必要的刻度线
    ax.tick_params(axis="both", which="both", length=0)

    # --- 在图中添加文本标识 (可选) ---
    text_style = {
        "color": "gray",
        "ha": "center",
        "va": "center",
        "fontsize": 10,
        "fontweight": "bold",
    }
    ax.text(e_mid, e_mid, "e-e", **text_style)
    ax.text(p_mid, p_mid, "p-p", **text_style)
    ax.text(p_mid, e_mid, "e-p", **text_style)
    ax.text(e_mid, p_mid, "e-p", **text_style)

    ax.set_title("NEO-DMRG Mutual Information Map", pad=20, fontsize=14)

    plt.tight_layout()
    plt.savefig("minfo_optimized.png", bbox_inches="tight")
    # plt.show()


# 调用
plot_neo_minfo(minfo, ncas_e, ncas_p)
