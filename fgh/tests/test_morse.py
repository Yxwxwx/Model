import numpy as np

from fgh.fgh1d import FGH1D


def morse_exact_energy(
    n: int,
    De: float,
    a: float,
    mass: float,
) -> float:
    omega = a * np.sqrt(
        2.0 * De / mass
    )

    q = n + 0.5

    return (
        omega * q
        - omega**2
        / (4.0 * De)
        * q**2
    )


def test_morse_oscillator():
    De = 8.0
    a = 0.5
    mass = 1.0
    xe = 0.0

    x = np.linspace(
        -8.0,
        20.0,
        256,
        endpoint=False,
    )

    def potential(x):
        return De * (
            1.0
            - np.exp(
                -a * (x - xe)
            )
        ) ** 2

    solver = FGH1D(
        x=x,
        potential=potential,
        mass=mass,
    )

    result = solver.solve(
        nstates=4,
        method="iterative",
    )

    exact = np.array(
        [
            morse_exact_energy(
                n,
                De,
                a,
                mass,
            )
            for n in range(4)
        ]
    )

    np.testing.assert_allclose(
        result.energies,
        exact,
        atol=1e-9,
    )