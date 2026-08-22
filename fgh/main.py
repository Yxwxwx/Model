import numpy as np
from pyscf import fci, gto, scf
from pyscf.data.nist import MP_ME

from fgh.fgh3d import FGH3D
from fgh.pes import build_pes_3d

# ============================================================
# Literature parameters
# ============================================================

R_HE = 3.40151 / 2.0
PROTON_MASS = MP_ME
BASIS = "6-31g"

HARTREE_TO_CM = 219474.63136320

REFERENCE_EXCITATION_CM = 1541.0


# ============================================================
# Electronic structure
# ============================================================


def electronic_energy(
    x: float,
    y: float,
    z: float,
) -> float:
    """
    Born-Oppenheimer FCI energy of HeHHe+ for a fixed
    proton coordinate (x, y, z).

    All coordinates are in bohr.
    """
    mol = gto.M(
        atom=[
            ("He", (-R_HE, 0.0, 0.0)),
            ("He", (+R_HE, 0.0, 0.0)),
            ("H", (x, y, z)),
        ],
        basis=BASIS,
        charge=1,
        spin=0,
        unit="Bohr",
        verbose=0,
    )

    # RHF orbitals
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-12
    mf.max_cycle = 100
    mf.kernel()

    if not mf.converged:
        raise RuntimeError(f"RHF failed at proton position ({x:.8f}, {y:.8f}, {z:.8f})")

    # Full CI for the four electrons.
    cisolver = fci.FCI(mf)
    cisolver.conv_tol = 1e-12
    cisolver.max_cycle = 100

    energy, _ = cisolver.kernel()

    # fci.FCI(mf) returns the total FCI energy,
    # including nuclear repulsion.
    return float(energy)


# ============================================================
# Fourier grid
# ============================================================


def make_grid(
    n: int,
    half_width: float,
) -> np.ndarray:
    dx = 2.0 * half_width / n

    return (np.arange(n) - n // 2) * dx


# ============================================================
# Calculation parameters
# ============================================================

# First production calculation.
#
# These are numerical FGH parameters, not physical parameters.
# After reproducing the result approximately, converge them.
NGRID = 64

X_HALF_WIDTH = 1.20
Y_HALF_WIDTH = 1.20
Z_HALF_WIDTH = 1.20

NSTATES = 4

CHECKPOINT = "hehhe_fci_6-31g_pes.npz"


# ============================================================
# Main
# ============================================================


def main():
    x = make_grid(
        NGRID,
        X_HALF_WIDTH,
    )

    y = make_grid(
        NGRID,
        Y_HALF_WIDTH,
    )

    z = make_grid(
        NGRID,
        Z_HALF_WIDTH,
    )

    print("=" * 70)
    print("HeHHe+ : FCI / 6-31G + 3D FGH")
    print("=" * 70)

    print()
    print("Physical parameters")
    print(f"  R(He-He)      = {2.0 * R_HE:.8f} bohr")
    print(f"  basis         = {BASIS}")
    print(f"  proton mass   = {PROTON_MASS:.6f} m_e")

    print()
    print("FGH grid")
    print(f"  shape         = {len(x)} x {len(y)} x {len(z)}")
    print(f"  total points  = {len(x) * len(y) * len(z)}")
    print(f"  dx            = {x[1] - x[0]:.8f} bohr")
    print(f"  dy            = {y[1] - y[0]:.8f} bohr")
    print(f"  dz            = {z[1] - z[0]:.8f} bohr")
    print(f"  x range       = [{x[0]:+.6f}, {x[-1]:+.6f}] bohr")
    print(f"  y range       = [{y[0]:+.6f}, {y[-1]:+.6f}] bohr")
    print(f"  z range       = [{z[0]:+.6f}, {z[-1]:+.6f}] bohr")

    # ========================================================
    # Generate Born-Oppenheimer PES
    # ========================================================

    print()
    print("=" * 70)
    print("Generating electronic FCI potential-energy surface")
    print("=" * 70)
    print()

    pes = build_pes_3d(
        x=x,
        y=y,
        z=z,
        energy_function=electronic_energy,
        checkpoint=CHECKPOINT,
        checkpoint_interval=100,
        verbose=True,
    )

    print()
    print("PES completed.")
    print(f"  Vmin = {np.min(pes):.12f} Eh")
    print(f"  Vmax = {np.max(pes):.12f} Eh")

    # The minimum should occur at/near the center.
    center = NGRID // 2

    v0 = pes[center, center, center]

    dx = x[1] - x[0]
    dy = y[1] - y[0]
    dz = z[1] - z[0]

    kx = (
        pes[center + 1, center, center] - 2.0 * v0 + pes[center - 1, center, center]
    ) / dx**2

    ky = (
        pes[center, center + 1, center] - 2.0 * v0 + pes[center, center - 1, center]
    ) / dy**2

    kz = (
        pes[center, center, center + 1] - 2.0 * v0 + pes[center, center, center - 1]
    ) / dz**2

    def force_constant_to_cm(k):
        omega = np.sqrt(k / PROTON_MASS)
        return omega * HARTREE_TO_CM

    print()
    print("Local harmonic analysis")
    print(f"kx = {kx:.10f} Eh/bohr^2  omega_x = {force_constant_to_cm(kx):.3f} cm^-1")
    print(f"ky = {ky:.10f} Eh/bohr^2  omega_y = {force_constant_to_cm(ky):.3f} cm^-1")
    print(f"kz = {kz:.10f} Eh/bohr^2  omega_z = {force_constant_to_cm(kz):.3f} cm^-1")

    # PES symmetry sanity check
    print()
    print("PES symmetry check")

    print(
        "V(+x,0,0) - V(-x,0,0) =",
        pes[center + 1, center, center] - pes[center - 1, center, center],
    )

    print(
        "V(0,+y,0) - V(0,0,+z) =",
        pes[center, center + 1, center] - pes[center, center, center + 1],
    )

    print(f"  V(0,0,0) = {pes[center, center, center]:.12f} Eh")

    # ========================================================
    # 3D FGH
    # ========================================================

    print()
    print("=" * 70)
    print("Solving proton 3D FGH")
    print("=" * 70)
    print()

    solver = FGH3D(
        x=x,
        y=y,
        z=z,
        potential=pes,
        mass=PROTON_MASS,
    )

    result = solver.solve(
        nstates=NSTATES,
        method="iterative",
        tol=1e-11,
    )

    excitation_hartree = result.excitation_energies

    excitation_cm = excitation_hartree * HARTREE_TO_CM

    # ========================================================
    # Results
    # ========================================================

    print("FGH eigenstates")
    print()
    print(f"{'State':>6s}{'Energy / Eh':>22s}{'Delta / Eh':>18s}{'Delta / cm^-1':>20s}")

    print("-" * 66)

    for state in range(result.nstates):
        print(
            f"{state:6d}"
            f"{result.energies[state]:22.12f}"
            f"{excitation_hartree[state]:18.10f}"
            f"{excitation_cm[state]:20.3f}"
        )

    # ========================================================
    # Comparison with literature
    # ========================================================

    calculated = excitation_cm[1]

    error = calculated - REFERENCE_EXCITATION_CM

    print()
    print("=" * 70)
    print("Comparison")
    print("=" * 70)

    print(f"Literature FGH : {REFERENCE_EXCITATION_CM:.1f} cm^-1")

    print(f"This calculation: {calculated:.3f} cm^-1")

    print(f"Error           : {error:+.3f} cm^-1")


if __name__ == "__main__":
    main()
