import numpy as np
from pyscf import cc, gto, scf
from pyscf.data.nist import MP_ME

from fgh.fgh3d import FGH3D
from fgh.helper import second_order_scf, stable_opt_internal
from fgh.pes import build_pes_3d

# ============================================================
# Unit conversion
# ============================================================

ANG_TO_BOHR = 1.8897261254578281
HARTREE_TO_CM = 219474.63136320


# ============================================================
# Literature parameters
# ============================================================

# HCN geometry from the NEO-MRCI paper:
#
# H ---- C ---- N
#
# H: z = 0
# C: z = 1.065 Angstrom
# N: z = 2.218 Angstrom
#
# Therefore:
#
# R(C-H) = 1.065 Angstrom
# R(C-N) = 1.153 Angstrom

R_CH_ANG = 1.065
R_CN_ANG = 1.153

R_CH = R_CH_ANG * ANG_TO_BOHR
R_CN = R_CN_ANG * ANG_TO_BOHR

PROTON_MASS = MP_ME

BASIS = "cc-pvtz"

REFERENCE_BEND_CM = 632.0
REFERENCE_STRETCH_CM = 3152.0


# ============================================================
# Electronic structure
# ============================================================


def electronic_energy(
    x: float,
    y: float,
    z: float,
) -> float:
    """
    Born-Oppenheimer CCSD energy of HCN for a fixed
    proton coordinate (x, y, z).

    All coordinates are in bohr.
    """

    mol = gto.M(
        atom=[
            (
                "H",
                (x, y, z),
            ),
            (
                "C",
                (0.0, 0.0, R_CH),
            ),
            (
                "N",
                (0.0, 0.0, R_CH + R_CN),
            ),
        ],
        basis=BASIS,
        charge=0,
        spin=0,
        unit="Bohr",
        verbose=0,
    )

    # ========================================================
    # RHF
    # ========================================================

    mf = scf.RHF(mol)

    mf.conv_tol = 1e-12
    mf.max_cycle = 100

    mf.kernel()

    # ========================================================
    # Second-order SCF fallback
    # ========================================================

    if not mf.converged:
        mf = second_order_scf(mf)

    if not mf.converged:
        raise RuntimeError(
            f"RHF failed at proton position ({x:.8f}, {y:.8f}, {z:.8f}) bohr"
        )

    # ========================================================
    # Internal stability optimization
    # ========================================================

    mf = stable_opt_internal(mf)

    if not mf.converged:
        raise RuntimeError(
            "Stable RHF optimization failed at proton position "
            f"({x:.8f}, {y:.8f}, {z:.8f}) bohr"
        )

    # ========================================================
    # CCSD
    # ========================================================

    mycc = cc.CCSD(mf)

    mycc.conv_tol = 1e-9
    mycc.max_cycle = 100

    mycc.kernel()

    if not mycc.converged:
        raise RuntimeError(
            f"CCSD failed at proton position ({x:.8f}, {y:.8f}, {z:.8f}) bohr"
        )

    return float(mycc.e_tot)


# ============================================================
# Fourier grid
# ============================================================


def make_grid(
    n: int,
    minimum_angstrom: float,
    maximum_angstrom: float,
) -> np.ndarray:
    """
    Construct a uniform Fourier grid.

    The literature provides the physical grid range in Angstrom,
    while the FGH implementation uses atomic units.

    endpoint=False is required for the periodic Fourier grid.
    """

    minimum = minimum_angstrom * ANG_TO_BOHR
    maximum = maximum_angstrom * ANG_TO_BOHR

    return np.linspace(
        minimum,
        maximum,
        n,
        endpoint=False,
    )


# ============================================================
# FGH calculation parameters
# ============================================================

# The article specifies the spatial range but does not state
# the number of grid points in the main text.
#
# 64^3 is used here as the production grid and should be
# checked for convergence if high precision is required.

NGRID = 64

GRID_MIN_ANG = -0.7935
GRID_MAX_ANG = +0.8464

# Need enough states to reach the fundamental stretch,
# which occurs above several bend overtone/combination states.
NSTATES = 10

CHECKPOINT = "hcn_ccsd_ccpvtz_pes.npz"

CHECKPOINT_INTERVAL = 1000


# ============================================================
# Main
# ============================================================


def main():
    # --------------------------------------------------------
    # Grid
    # --------------------------------------------------------

    x = make_grid(
        NGRID,
        GRID_MIN_ANG,
        GRID_MAX_ANG,
    )

    y = make_grid(
        NGRID,
        GRID_MIN_ANG,
        GRID_MAX_ANG,
    )

    z = make_grid(
        NGRID,
        GRID_MIN_ANG,
        GRID_MAX_ANG,
    )

    print("=" * 72)
    print("HCN : CCSD / cc-pVTZ + 3D FGH")
    print("=" * 72)

    print()
    print("Physical parameters")

    print(f"  R(C-H)       = {R_CH_ANG:.6f} Angstrom")

    print(f"  R(C-N)       = {R_CN_ANG:.6f} Angstrom")

    print(f"  basis        = {BASIS}")

    print(f"  proton mass  = {PROTON_MASS:.8f} m_e")

    print()
    print("FGH grid")

    print(f"  shape        = {len(x)} x {len(y)} x {len(z)}")

    print(f"  total points = {len(x) * len(y) * len(z)}")

    print(f"  literature range = [{GRID_MIN_ANG:+.4f}, {GRID_MAX_ANG:+.4f}) Angstrom")

    print(f"  dx           = {x[1] - x[0]:.8f} bohr")

    print(f"  dy           = {y[1] - y[0]:.8f} bohr")

    print(f"  dz           = {z[1] - z[0]:.8f} bohr")

    print(f"  x range      = [{x[0]:+.6f}, {x[-1]:+.6f}] bohr")

    print(f"  y range      = [{y[0]:+.6f}, {y[-1]:+.6f}] bohr")

    print(f"  z range      = [{z[0]:+.6f}, {z[-1]:+.6f}] bohr")

    # ========================================================
    # Generate electronic PES
    # ========================================================

    print()
    print("=" * 72)
    print("Generating CCSD / cc-pVTZ potential-energy surface")
    print("=" * 72)
    print()

    pes = build_pes_3d(
        x=x,
        y=y,
        z=z,
        energy_function=electronic_energy,
        checkpoint=CHECKPOINT,
        checkpoint_interval=CHECKPOINT_INTERVAL,
        verbose=True,
    )

    print()
    print("PES completed.")

    print(f"  Vmin = {np.min(pes):.12f} Eh")

    print(f"  Vmax = {np.max(pes):.12f} Eh")

    # ========================================================
    # PES symmetry check
    # ========================================================
    #
    # HCN is linear along z.
    #
    # The PES must therefore be cylindrically symmetric:
    #
    #     V(x, 0, z) = V(0, y, z)
    #
    # for x = y.
    #
    # Unlike HeHHe+, there is NO symmetry under z -> -z.
    # ========================================================

    iz0 = int(np.argmin(np.abs(z)))

    ix0 = int(np.argmin(np.abs(x)))

    iy0 = int(np.argmin(np.abs(y)))

    # Pick equivalent transverse displacements one step away.
    ix1 = min(
        ix0 + 1,
        len(x) - 1,
    )

    iy1 = min(
        iy0 + 1,
        len(y) - 1,
    )

    transverse_difference = pes[ix1, iy0, iz0] - pes[ix0, iy1, iz0]

    print()
    print("PES symmetry check")

    print(f"  V(+x,0,z0) - V(0,+y,z0) = {transverse_difference:+.6e} Eh")

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
    #
    # Molecular axis = z
    #
    # x/y matrix elements identify fundamental bends.
    # z matrix elements identify the fundamental stretch.
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
    # Results
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
    # Literature comparison
    # ========================================================

    # For linear HCN the lowest two excited states should be
    # the degenerate proton bend modes.

    calculated_bend_1 = excitation_cm[1]

    calculated_bend_2 = excitation_cm[2]

    bend_average = 0.5 * (calculated_bend_1 + calculated_bend_2)

    # The fundamental stretch should have by far the largest
    # transition matrix element along the molecular z-axis.
    stretch_state = int(np.argmax(tz[1:])) + 1

    calculated_stretch = excitation_cm[stretch_state]

    print()
    print("=" * 72)
    print("Comparison with NEO-MRCI paper FGH reference")
    print("=" * 72)

    print()
    print("Fundamental proton bend")

    print(f"  Literature FGH : {REFERENCE_BEND_CM:.1f} cm^-1")

    print(f"  This calculation: {bend_average:.3f} cm^-1")

    print(f"  Degenerate pair : {calculated_bend_1:.3f}, {calculated_bend_2:.3f} cm^-1")

    print(f"  Error           : {bend_average - REFERENCE_BEND_CM:+.3f} cm^-1")

    print()
    print("Fundamental proton stretch")

    print(f"  Assigned state  : {stretch_state}")

    print(f"  Literature FGH : {REFERENCE_STRETCH_CM:.1f} cm^-1")

    print(f"  This calculation: {calculated_stretch:.3f} cm^-1")

    print(f"  Error           : {calculated_stretch - REFERENCE_STRETCH_CM:+.3f} cm^-1")


if __name__ == "__main__":
    main()
