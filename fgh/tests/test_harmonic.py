import numpy as np

from fgh.fgh1d import FGH1D


def test_dense_matrix_and_fft_matvec():
    x = np.linspace(
        -8.0,
        8.0,
        128,
        endpoint=False,
    )

    solver = FGH1D(
        x=x,
        potential=lambda x: 0.5 * x**2,
        mass=1.0,
    )

    H = solver.hamiltonian_matrix()

    rng = np.random.default_rng(42)
    psi = rng.normal(size=x.size)

    y_dense = H @ psi
    y_fft = solver.matvec(psi)

    np.testing.assert_allclose(
        y_fft,
        y_dense,
        atol=1e-11,
        rtol=1e-11,
    )


def test_dense_and_iterative_solvers():
    x = np.linspace(
        -10.0,
        10.0,
        256,
        endpoint=False,
    )

    solver = FGH1D(
        x=x,
        potential=lambda x: 0.5 * x**2,
        mass=1.0,
    )

    dense = solver.solve(
        nstates=5,
        method="dense",
    )

    iterative = solver.solve(
        nstates=5,
        method="iterative",
    )

    np.testing.assert_allclose(
        iterative.energies,
        dense.energies,
        atol=1e-10,
    )

    for state in range(5):
        overlap = dense.dx * np.vdot(
            dense.wavefunctions[:, state],
            iterative.wavefunctions[:, state],
        )

        np.testing.assert_allclose(
            abs(overlap),
            1.0,
            atol=1e-9,
        )
