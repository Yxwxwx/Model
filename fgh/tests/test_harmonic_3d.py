import numpy as np

from fgh.fgh3d import FGH3D


def exact_3d_harmonic_energies(
    omegas,
    nstates,
):
    energies = []

    for nx in range(6):
        for ny in range(6):
            for nz in range(6):
                energy = (
                    (nx + 0.5) * omegas[0]
                    + (ny + 0.5) * omegas[1]
                    + (nz + 0.5) * omegas[2]
                )

                energies.append(energy)

    energies = np.sort(energies)

    return energies[:nstates]


def test_harmonic_oscillator_3d():
    ngrid = 24

    x = np.linspace(
        -6.5,
        6.5,
        ngrid,
        endpoint=False,
    )

    y = np.linspace(
        -6.5,
        6.5,
        ngrid,
        endpoint=False,
    )

    z = np.linspace(
        -6.5,
        6.5,
        ngrid,
        endpoint=False,
    )

    mass = 1.0

    omega_x = 0.7
    omega_y = 1.1
    omega_z = 1.6

    omegas = (
        omega_x,
        omega_y,
        omega_z,
    )

    def potential(
        X,
        Y,
        Z,
    ):
        return 0.5 * mass * (omega_x**2 * X**2 + omega_y**2 * Y**2 + omega_z**2 * Z**2)

    solver = FGH3D(
        x=x,
        y=y,
        z=z,
        potential=potential,
        mass=mass,
    )

    nstates = 8

    result = solver.solve(
        nstates=nstates,
        method="iterative",
        tol=1e-11,
    )

    exact = exact_3d_harmonic_energies(
        omegas,
        nstates,
    )

    np.testing.assert_allclose(
        result.energies,
        exact,
        atol=5e-7,
    )


def test_real_mass_3d_harmonic():
    from pyscf.data.nist import MP_ME

    HARTREE_TO_CM = 219474.63136320

    mass = MP_ME

    omega_y = 1541.0 / HARTREE_TO_CM
    omega_z = 1541.0 / HARTREE_TO_CM

    # Deliberately make x excitation higher.
    omega_x = 2500.0 / HARTREE_TO_CM

    n = 64
    half_width = 1.2

    grid = (np.arange(n) - n // 2) * (2.0 * half_width / n)

    def potential(X, Y, Z):
        return 0.5 * mass * (omega_x**2 * X**2 + omega_y**2 * Y**2 + omega_z**2 * Z**2)

    solver = FGH3D(
        x=grid,
        y=grid,
        z=grid,
        potential=potential,
        mass=mass,
    )

    result = solver.solve(
        nstates=4,
        tol=1e-11,
    )

    excitation_cm = result.excitation_energies * HARTREE_TO_CM

    print(excitation_cm)

    np.testing.assert_allclose(
        excitation_cm[1:3],
        [1541.0, 1541.0],
        atol=1.0,
    )
