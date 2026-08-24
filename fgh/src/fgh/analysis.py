from dataclasses import dataclass
from math import factorial

import numpy as np
from scipy.special import eval_hermite

from .fgh3d import FGH3DResult


@dataclass(frozen=True)
class HarmonicComponent:
    quanta: tuple[int, int, int]
    coefficient: complex
    weight: float


@dataclass(frozen=True)
class HarmonicProjection:
    state: int
    components: list[HarmonicComponent]
    total_weight: float


def harmonic_wavefunction(
    grid: np.ndarray,
    n: int,
    mass: float,
    omega: float,
    center: float = 0.0,
) -> np.ndarray:
    """
    One-dimensional harmonic-oscillator wavefunction.

    Atomic units are assumed.

    Parameters
    ----------
    grid
        Coordinate grid in bohr.

    n
        Harmonic quantum number.

    mass
        Particle mass in electron masses.

    omega
        Harmonic frequency in Hartree
        (angular frequency in atomic units).

    center
        Harmonic minimum in bohr.
    """
    alpha = mass * omega

    q = np.sqrt(alpha) * (grid - center)

    normalization = (alpha / np.pi) ** 0.25 / np.sqrt(2**n * factorial(n))

    psi = normalization * eval_hermite(n, q) * np.exp(-0.5 * q**2)

    # Renormalize on the actual finite FGH grid.
    dq = grid[1] - grid[0]

    norm = np.sqrt(dq * np.sum(np.abs(psi) ** 2))

    return psi / norm


def harmonic_projection(
    result: FGH3DResult,
    state: int,
    mass: float,
    omegas: tuple[float, float, float],
    center: tuple[float, float, float] = (
        0.0,
        0.0,
        0.0,
    ),
    nmax: tuple[int, int, int] = (
        6,
        6,
        3,
    ),
) -> HarmonicProjection:
    """
    Project a 3D FGH eigenstate onto Cartesian harmonic
    product states

        |nx, ny, nz>.

    Parameters
    ----------
    result
        FGH3D result.

    state
        FGH eigenstate index.

    mass
        Nuclear mass in electron masses.

    omegas
        Harmonic frequencies (omega_x, omega_y, omega_z)
        in atomic units.

    center
        Harmonic reference center in bohr.

    nmax
        Maximum quantum number along x, y, z.

    Returns
    -------
    HarmonicProjection
        Sorted harmonic components and total represented weight.
    """
    omega_x, omega_y, omega_z = omegas

    cx, cy, cz = center

    nx_max, ny_max, nz_max = nmax

    basis_x = np.column_stack(
        [
            harmonic_wavefunction(
                result.x,
                n,
                mass,
                omega_x,
                cx,
            )
            for n in range(nx_max + 1)
        ]
    )

    basis_y = np.column_stack(
        [
            harmonic_wavefunction(
                result.y,
                n,
                mass,
                omega_y,
                cy,
            )
            for n in range(ny_max + 1)
        ]
    )

    basis_z = np.column_stack(
        [
            harmonic_wavefunction(
                result.z,
                n,
                mass,
                omega_z,
                cz,
            )
            for n in range(nz_max + 1)
        ]
    )

    psi = result.wavefunctions[
        ...,
        state,
    ]

    # C[nx, ny, nz] =
    #
    # integral dx dy dz
    # phi_nx*(x) phi_ny*(y) phi_nz*(z) psi(x,y,z)
    #
    coefficients = result.dv * np.einsum(
        "ia,jb,kc,ijk->abc",
        basis_x.conj(),
        basis_y.conj(),
        basis_z.conj(),
        psi,
        optimize=True,
    )

    components = []

    for nx in range(nx_max + 1):
        for ny in range(ny_max + 1):
            for nz in range(nz_max + 1):
                coefficient = coefficients[
                    nx,
                    ny,
                    nz,
                ]

                weight = float(np.abs(coefficient) ** 2)

                components.append(
                    HarmonicComponent(
                        quanta=(
                            nx,
                            ny,
                            nz,
                        ),
                        coefficient=coefficient,
                        weight=weight,
                    )
                )

    components.sort(
        key=lambda component: component.weight,
        reverse=True,
    )

    total_weight = float(np.sum(np.abs(coefficients) ** 2))

    return HarmonicProjection(
        state=state,
        components=components,
        total_weight=total_weight,
    )


def print_harmonic_projection(
    projection: HarmonicProjection,
    threshold: float = 1e-3,
    max_components: int = 10,
) -> None:
    """
    Print dominant harmonic configurations.
    """
    print(f"State {projection.state}:")

    print(f"  represented weight = {projection.total_weight:.6f}")

    shown = 0

    for component in projection.components:
        if component.weight < threshold:
            continue

        nx, ny, nz = component.quanta

        print(
            f"  ({nx},{ny},{nz})  "
            f"weight = "
            f"{component.weight:10.6f}  "
            f"C = "
            f"{component.coefficient.real:+.6f}"
        )

        shown += 1

        if shown >= max_components:
            break
