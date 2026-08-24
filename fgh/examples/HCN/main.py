import numpy as np

from pyscf import fci, gto, scf
from pyscf.data.nist import MP_ME

from fgh.analysis import harmonic_projection, print_harmonic_projection
from fgh.fgh3d import FGH3D
from fgh.pes import build_pes_3d


# ============================================================
# Literature parameters
# ============================================================

# He-He distance = 3.40151 bohr ~= 1.8 Angstrom
R_HE = 3.40151 / 2.0

PROTON_MASS = MP_ME

BASIS = "6-31g"

HARTREE_TO_CM = 219474.63136320

# NEO-MRCI paper:
# FCI/6-31G + 3D FGH fundamental proton bend
REFERENCE_BEND_CM = 771.0


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

    Molecular axis: x
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

    # RHF
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-12
    mf.max_cycle = 100
    mf.kernel()

    if not mf.converged:
        mf = mf.newton()
        mf.conv_tol = 1e-12
        mf.max_cycle = 100
        mf.kernel()

    if not mf.converged:
        raise RuntimeError(
            f"RHF failed at proton position ({x:.8f}, {y:.8f}, {z:.8f}) bohr"
        )

    # FCI
    cisolver = fci.FCI(mf)
    cisolver.conv_tol = 1e-12
    cisolver.max_cycle = 100

    energy, _ = cisolver.kernel()

    return float(energy)


# ============================================================
# Fourier grid
# ============================================================


def make_grid(
    n: int,
    half_width: float,
) -> np.ndarray:
    """
    Uniform periodic Fourier grid centered at zero.
    """
    dx = 2.0 * half_width / n

    return (np.arange(n) - n // 2) * dx


# ============================================================
# Calculation parameters
# ============================================================

NGRID = 64

X_HALF_WIDTH = 1.20
Y_HALF_WIDTH = 1.20
Z_HALF_WIDTH = 1.20

# Four states are enough for:
#
#   ground
#   bend doublet
#   next excitation
#
NSTATES = 4

CHECKPOINT = "hehhe_fci_6-31g_pes.npz"


# ============================================================
# Main
# ============================================================


def main():
    # --------------------------------------------------------
    # Grid
    # --------------------------------------------------------

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

    print("=" * 72)
    print("HeHHe+ : FCI / 6-31G + 3D FGH")
    print("=" * 72)

    print()
    print("Physical parameters")

    print(f"  R(He-He)     = {2.0 * R_HE:.8f} bohr")

    print(f"  basis        = {BASIS}")

    print(f"  proton mass  = {PROTON_MASS:.8f} m_e")

    print()
    print("Coordinate convention")

    print("  x             = molecular axis / proton stretch")

    print("  y, z          = degenerate transverse proton bends")

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
    # PES
    # ========================================================

    print()
    print("=" * 72)
    print("Generating electronic FCI potential-energy surface")
    print("=" * 72)
    print()

    pes = build_pes_3d(
        x=x,
        y=y,
        z=z,
        energy_function=electronic_energy,
        checkpoint=CHECKPOINT,
        checkpoint_interval=1000,
        verbose=True,
    )

    print()
    print("PES completed.")

    print(f"  Vmin = {np.min(pes):.12f} Eh")

    print(f"  Vmax = {np.max(pes):.12f} Eh")

    # ========================================================
    # Local harmonic analysis
    # ========================================================

    center = NGRID // 2

    v0 = pes[
        center,
        center,
        center,
    ]

    dx = x[1] - x[0]
    dy = y[1] - y[0]
    dz = z[1] - z[0]

    kx = (
        pes[
            center + 1,
            center,
            center,
        ]
        - 2.0 * v0
        + pes[
            center - 1,
            center,
            center,
        ]
    ) / dx**2

    ky = (
        pes[
            center,
            center + 1,
            center,
        ]
        - 2.0 * v0
        + pes[
            center,
            center - 1,
            center,
        ]
    ) / dy**2

    kz = (
        pes[
            center,
            center,
            center + 1,
        ]
        - 2.0 * v0
        + pes[
            center,
            center,
            center - 1,
        ]
    ) / dz**2

    omega_x = np.sqrt(kx / PROTON_MASS)

    omega_y = np.sqrt(ky / PROTON_MASS)

    omega_z = np.sqrt(kz / PROTON_MASS)

    print()
    print("Local harmonic analysis")

    print(
        f"  stretch x : "
        f"k = {kx:.10f} Eh/bohr^2   "
        f"omega = "
        f"{omega_x * HARTREE_TO_CM:.3f} cm^-1"
    )

    print(
        f"  bend y    : "
        f"k = {ky:.10f} Eh/bohr^2   "
        f"omega = "
        f"{omega_y * HARTREE_TO_CM:.3f} cm^-1"
    )

    print(
        f"  bend z    : "
        f"k = {kz:.10f} Eh/bohr^2   "
        f"omega = "
        f"{omega_z * HARTREE_TO_CM:.3f} cm^-1"
    )

    # ========================================================
    # PES symmetry
    # ========================================================

    print()
    print("PES symmetry check")

    inversion_error = (
        pes[
            center + 1,
            center,
            center,
        ]
        - pes[
            center - 1,
            center,
            center,
        ]
    )

    transverse_error = (
        pes[
            center,
            center + 1,
            center,
        ]
        - pes[
            center,
            center,
            center + 1,
        ]
    )

    print(f"  V(+x,0,0) - V(-x,0,0) = {inversion_error:+.6e} Eh")

    print(f"  V(0,+y,0) - V(0,0,+z) = {transverse_error:+.6e} Eh")

    print(f"  V(0,0,0) = {v0:.12f} Eh")

    # ========================================================
    # 3D FGH
    # ========================================================

    print()
    print("=" * 72)
    print("Solving proton 3D FGH")
    print("=" * 72)
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
    # Transition-coordinate analysis
    # ========================================================

    tx = np.zeros(result.nstates)

    ty = np.zeros(result.nstates)

    tz = np.zeros(result.nstates)

    for state in range(
        1,
        result.nstates,
    ):
        tx[state] = abs(
            result.matrix_element(
                lambda X, Y, Z: X,
                bra=0,
                ket=state,
            )
        )

        ty[state] = abs(
            result.matrix_element(
                lambda X, Y, Z: Y,
                bra=0,
                ket=state,
            )
        )

        tz[state] = abs(
            result.matrix_element(
                lambda X, Y, Z: Z,
                bra=0,
                ket=state,
            )
        )

    # ========================================================
    # Eigenstate energies
    # ========================================================

    print("FGH eigenstates")
    print()

    print(
        f"{'State':>6s}"
        f"{'Energy / Eh':>22s}"
        f"{'Delta / cm^-1':>18s}"
        f"{'|<0|x|n>|':>16s}"
        f"{'|<0|y|n>|':>16s}"
        f"{'|<0|z|n>|':>16s}"
    )

    print("-" * 94)

    for state in range(result.nstates):
        print(
            f"{state:6d}"
            f"{result.energies[state]:22.12f}"
            f"{excitation_cm[state]:18.3f}"
            f"{tx[state]:16.8f}"
            f"{ty[state]:16.8f}"
            f"{tz[state]:16.8f}"
        )

    # ========================================================
    # Harmonic projection
    # ========================================================

    print()
    print("=" * 72)
    print("Harmonic projection analysis")
    print("=" * 72)

    print()
    print("Harmonic labels use the present Cartesian convention:")
    print("  (nx, ny, nz)")
    print("  nx = proton stretch quanta")
    print("  ny, nz = transverse bend quanta")

    for state in range(result.nstates):
        projection = harmonic_projection(
            result=result,
            state=state,
            mass=PROTON_MASS,
            omegas=(
                omega_x,
                omega_y,
                omega_z,
            ),
            center=(
                0.0,
                0.0,
                0.0,
            ),
            nmax=(
                4,
                6,
                6,
            ),
        )

        print()
        print(f"State {state}, dE = {excitation_cm[state]:.3f} cm^-1")

        print_harmonic_projection(
            projection,
            threshold=0.01,
            max_components=8,
        )

        # ----------------------------------------------------
        # Useful symmetry/manifold weights
        # ----------------------------------------------------

        weights = {
            component.quanta: component.weight for component in projection.components
        }

        ground_weight = weights.get(
            (0, 0, 0),
            0.0,
        )

        stretch_weight = weights.get(
            (1, 0, 0),
            0.0,
        )

        bend_weight = weights.get(
            (0, 1, 0),
            0.0,
        ) + weights.get(
            (0, 0, 1),
            0.0,
        )

        bend_overtone_weight = (
            weights.get(
                (0, 2, 0),
                0.0,
            )
            + weights.get(
                (0, 1, 1),
                0.0,
            )
            + weights.get(
                (0, 0, 2),
                0.0,
            )
        )

        print()
        print("  Selected manifold weights")
        print(f"    ground |000>         : {ground_weight:.6f}")
        print(f"    stretch fundamental  : {stretch_weight:.6f}")
        print(f"    bend fundamental     : {bend_weight:.6f}")
        print(f"    bend N=2 manifold    : {bend_overtone_weight:.6f}")

    # ========================================================
    # Literature comparison
    # ========================================================

    bend_1 = excitation_cm[1]

    bend_2 = excitation_cm[2]

    calculated_bend = 0.5 * (bend_1 + bend_2)

    error = calculated_bend - REFERENCE_BEND_CM

    print()
    print("=" * 72)
    print("Comparison with NEO-MRCI paper FGH reference")
    print("=" * 72)

    print()
    print("Fundamental proton bend")

    print(f"  Literature FGH : {REFERENCE_BEND_CM:.1f} cm^-1")
    print(f"  This calculation: {calculated_bend:.3f} cm^-1")
    print(f"  Degenerate pair : {bend_1:.3f}, {bend_2:.3f} cm^-1")
    print(f"  Error           : {error:+.3f} cm^-1")


if __name__ == "__main__":
    main()
