import numpy as np
import scipy.linalg

# ========= Parameter Settings =========
# Cavity frequency (omega_c)
OMEGA = 0.160984
# Coupling vector (lambda)
lam_vec = np.array([0.0, 0.05, 0.0])
# Number of photon Fock states
N_PH = 7
model = "Naphtalene"
nroots = 50

# Load electronic structure data (energies and dipole matrices)
dip_mat_raw = np.load(model + "_dipole.npy")
energies = np.load(model + "_energies.npy")

# 1. Project Dipole Matrix onto the coupling vector: d = lambda . mu
# d_mat corresponds to the operator (\lambda \cdot \hat{d})
d_mat = np.einsum("x,xij->ij", lam_vec, dip_mat_raw)
d_ref = d_mat[0, 0]  # Reference dipole of S0 state

# 2. Construct Photon Operators (Fock space basis)
# Annihilation operator b
b = np.diag(np.sqrt(np.arange(1, N_PH)), 1)
# Creation operator b_dagger
bd = b.T
I_ph = np.eye(N_PH)
I_el = np.eye(nroots)

# 3. Construct Hamiltonian Components using Kronecker products
# H_total = H_ele + H_cav + H_int + H_dse

# Electronic part: H_ele \otimes I_ph
# Diagonal matrix of electronic eigen-energies
H_ele = np.kron(np.diag(energies), I_ph)

# Cavity part: I_el \otimes omega * b_dagger * b
# Describes the harmonic oscillator of the cavity mode
H_cav = np.kron(I_el, OMEGA * (bd @ b))

# Interaction part: -sqrt(omega/2) * d \otimes (b + b_dagger)
# Standard light-matter coupling in the length gauge
g_factor = np.sqrt(OMEGA / 2.0)
H_int = -np.kron(g_factor * d_mat, (b + bd))

# Dipole Self-Energy (DSE) part: 0.5 * d^2 \otimes I_ph
# Essential for gauge invariance and vacuum stability
# Note: d_mat @ d_mat calculates the square of the projected dipole operator
H_dse = np.kron(0.5 * (d_mat @ d_mat), I_ph)

# Total pPF Hamiltonian
H_pPF = H_ele + H_cav + H_int + H_dse
# Enforce hermiticity to eliminate minor numerical noise
H_pPF = 0.5 * (H_pPF + H_pPF.T)

# 4. Diagonalization to obtain polaritonic states
E_coupled, V_coupled = scipy.linalg.eigh(H_pPF)


# 5. Polariton Analysis Function
def analyze_polariton(E, V, el_e, n_roots, n_ph, top_k=20):
    """
    Analyzes the composition of the coupled light-matter states.
    <n>: Average photon number
    Max Prob: Contribution of the dominant basis state |S_alpha>|n>
    """
    print(
        f"\n{'State':<8} | {'Energy (a.u.)':<15} | {'<n>':<8} | {'Max Prob':<10} | {'Character'}"
    )
    print("-" * 80)

    # Pre-generate photon number indices for expectation value calculation
    n_indices = np.tile(np.arange(n_ph), n_roots)

    for i in range(min(len(E), top_k)):
        probs = np.abs(V[:, i]) ** 2
        # Calculate average photon occupancy: <n> = sum( P_i * n_i )
        avg_n = np.sum(probs * n_indices)

        # Identify the dominant electronic state and photon number
        max_idx = np.argmax(probs)
        alpha = max_idx // n_ph  # Electronic index
        n_val = max_idx % n_ph  # Photon index

        print(
            f"P-{i:<3}    | {E[i]:15.8f} | {avg_n:8.4f} | {probs[max_idx]:.4f}     | "
            f"S{alpha:<2} \u2297 |{n_val}>"
        )


# Execute Analysis
analyze_polariton(E_coupled, V_coupled, energies, nroots, N_PH)

print(f"\n[Info] S0 Permanent Projected Dipole: {d_ref:.6f} a.u.")
