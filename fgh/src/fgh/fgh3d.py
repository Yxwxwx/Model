from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt
from scipy.sparse.linalg import LinearOperator, eigsh

from .fgh1d import kinetic_matrix_1d

FloatArray = npt.NDArray[np.float64]

Potential3D = (
    Callable[
        [FloatArray, FloatArray, FloatArray],
        npt.ArrayLike,
    ]
    | npt.ArrayLike
)

SolverMethod = Literal[
    "dense",
    "iterative",
    "auto",
]


@dataclass(frozen=True)
class FGH3DResult:
    """
    Result of a three-dimensional Fourier Grid Hamiltonian
    calculation.

    Atomic units are used throughout.

    Attributes
    ----------
    x, y, z
        Coordinate grids in bohr.

    energies
        Eigenenergies in Hartree.

    wavefunctions
        Wavefunctions with shape

            (nx, ny, nz, nstates)

        normalized according to

            integral |psi(x, y, z)|^2 dx dy dz = 1.
    """

    x: FloatArray
    y: FloatArray
    z: FloatArray

    energies: FloatArray
    wavefunctions: FloatArray

    @property
    def dx(self) -> float:
        return float(self.x[1] - self.x[0])

    @property
    def dy(self) -> float:
        return float(self.y[1] - self.y[0])

    @property
    def dz(self) -> float:
        return float(self.z[1] - self.z[0])

    @property
    def dv(self) -> float:
        return self.dx * self.dy * self.dz

    @property
    def shape(self) -> tuple[int, int, int]:
        return (
            self.x.size,
            self.y.size,
            self.z.size,
        )

    @property
    def nstates(self) -> int:
        return self.energies.size

    @property
    def excitation_energies(self) -> FloatArray:
        return self.energies - self.energies[0]

    def density(
        self,
        state: int = 0,
    ) -> FloatArray:
        """
        Proton/nuclear probability density

            rho(x, y, z) = |psi(x, y, z)|^2.
        """
        psi = self.wavefunctions[..., state]

        return np.abs(psi) ** 2

    def overlap(
        self,
        state_i: int,
        state_j: int,
    ) -> complex:
        psi_i = self.wavefunctions[..., state_i]
        psi_j = self.wavefunctions[..., state_j]

        return complex(
            self.dv
            * np.vdot(
                psi_i,
                psi_j,
            )
        )

    def matrix_element(
        self,
        operator: Potential3D,
        bra: int,
        ket: int,
    ) -> complex:
        """
        Compute

            <bra | O(x,y,z) | ket>

        for an operator diagonal in coordinate space.
        """
        X = self.x[:, None, None]
        Y = self.y[None, :, None]
        Z = self.z[None, None, :]

        if callable(operator):
            op = np.asarray(operator(X, Y, Z))
        else:
            op = np.asarray(operator)

        try:
            op = np.broadcast_to(
                op,
                self.shape,
            )
        except ValueError as exc:
            raise ValueError(
                f"Operator shape {op.shape} cannot be "
                f"broadcast to grid shape {self.shape}."
            ) from exc

        psi_bra = self.wavefunctions[..., bra]
        psi_ket = self.wavefunctions[..., ket]

        return complex(
            self.dv
            * np.vdot(
                psi_bra,
                op * psi_ket,
            )
        )

    def expectation(
        self,
        operator: Potential3D,
        state: int = 0,
    ) -> float:
        value = self.matrix_element(
            operator=operator,
            bra=state,
            ket=state,
        )

        return float(np.real_if_close(value).real)

    def marginal_density(
        self,
        axis: int,
        state: int = 0,
    ) -> FloatArray:
        """
        Integrate the 3D density over two coordinates.

        axis = 0 -> rho(x)
        axis = 1 -> rho(y)
        axis = 2 -> rho(z)
        """
        rho = self.density(state)

        if axis == 0:
            return (
                self.dy
                * self.dz
                * np.sum(
                    rho,
                    axis=(1, 2),
                )
            )

        if axis == 1:
            return (
                self.dx
                * self.dz
                * np.sum(
                    rho,
                    axis=(0, 2),
                )
            )

        if axis == 2:
            return (
                self.dx
                * self.dy
                * np.sum(
                    rho,
                    axis=(0, 1),
                )
            )

        raise ValueError("axis must be 0, 1, or 2.")


class FGH3D:
    """
    Three-dimensional Fourier Grid Hamiltonian solver.

    The Hamiltonian is

        H =
            -(1/2m) nabla^2
            + V(x,y,z)

    in atomic units.

    The kinetic operator is evaluated as

        T psi =
            IFFTN[
                (kx^2 + ky^2 + kz^2)/(2m)
                * FFTN[psi]
            ].
    """

    def __init__(
        self,
        x: npt.ArrayLike,
        y: npt.ArrayLike,
        z: npt.ArrayLike,
        potential: Potential3D,
        mass: float = 1.0,
    ):
        self.x = self._validate_grid(
            x,
            "x",
        )
        self.y = self._validate_grid(
            y,
            "y",
        )
        self.z = self._validate_grid(
            z,
            "z",
        )

        if mass <= 0.0:
            raise ValueError("mass must be positive.")

        self.mass = float(mass)

        self.dx = float(self.x[1] - self.x[0])
        self.dy = float(self.y[1] - self.y[0])
        self.dz = float(self.z[1] - self.z[0])

        self.dv = self.dx * self.dy * self.dz

        self.shape = (
            self.x.size,
            self.y.size,
            self.z.size,
        )

        self.size = int(np.prod(self.shape))

        self.potential = self._evaluate_potential(potential)

        self.kx = (
            2.0
            * np.pi
            * np.fft.fftfreq(
                self.x.size,
                d=self.dx,
            )
        )

        self.ky = (
            2.0
            * np.pi
            * np.fft.fftfreq(
                self.y.size,
                d=self.dy,
            )
        )

        self.kz = (
            2.0
            * np.pi
            * np.fft.fftfreq(
                self.z.size,
                d=self.dz,
            )
        )

        #
        # Broadcasting avoids explicitly creating
        # three full k-space coordinate tensors.
        #
        self.kinetic_k = (
            self.kx[:, None, None] ** 2
            + self.ky[None, :, None] ** 2
            + self.kz[None, None, :] ** 2
        ) / (2.0 * self.mass)

    @staticmethod
    def _validate_grid(
        grid: npt.ArrayLike,
        name: str,
    ) -> FloatArray:
        grid = np.asarray(
            grid,
            dtype=float,
        )

        if grid.ndim != 1:
            raise ValueError(f"{name} must be one-dimensional.")

        if grid.size < 2:
            raise ValueError(f"{name} must contain at least two points.")

        if not np.all(np.isfinite(grid)):
            raise ValueError(f"{name} contains non-finite values.")

        d = grid[1] - grid[0]

        if d <= 0.0:
            raise ValueError(f"{name} must be strictly increasing.")

        if not np.allclose(
            np.diff(grid),
            d,
            rtol=1e-12,
            atol=1e-14,
        ):
            raise ValueError(f"{name} must be a uniform grid.")

        return grid

    def _evaluate_potential(
        self,
        potential: Potential3D,
    ) -> FloatArray:
        #
        # Coordinate views use broadcasting:
        #
        # X : (nx, 1, 1)
        # Y : (1, ny, 1)
        # Z : (1, 1, nz)
        #
        X = self.x[:, None, None]
        Y = self.y[None, :, None]
        Z = self.z[None, None, :]

        if callable(potential):
            v = np.asarray(
                potential(X, Y, Z),
                dtype=float,
            )
        else:
            v = np.asarray(
                potential,
                dtype=float,
            )

        try:
            v = np.broadcast_to(
                v,
                self.shape,
            )
        except ValueError as exc:
            raise ValueError(
                f"Potential shape {v.shape} "
                f"cannot be broadcast to "
                f"grid shape {self.shape}."
            ) from exc

        if not np.all(np.isfinite(v)):
            raise ValueError("Potential contains non-finite values.")

        return np.asarray(
            v,
            dtype=float,
        )

    def apply_kinetic(
        self,
        psi: npt.ArrayLike,
    ) -> FloatArray:
        """
        Apply the 3D kinetic operator.
        """
        psi = np.asarray(
            psi,
            dtype=float,
        )

        if psi.shape != self.shape:
            raise ValueError(f"psi must have shape {self.shape}.")

        psi_k = np.fft.fftn(
            psi,
            norm="ortho",
        )

        tpsi_k = self.kinetic_k * psi_k

        tpsi = np.fft.ifftn(
            tpsi_k,
            norm="ortho",
        )

        return np.asarray(
            np.real_if_close(tpsi).real,
            dtype=float,
        )

    def matvec(
        self,
        psi: npt.ArrayLike,
        potential_shift: float = 0.0,
    ) -> FloatArray:
        """
        Apply H to either a 3D grid wavefunction
        or a flattened vector.
        """
        psi = np.asarray(
            psi,
            dtype=float,
        )

        return_flat = False

        if psi.ndim == 1:
            if psi.size != self.size:
                raise ValueError(f"Flattened psi must contain {self.size} elements.")

            psi = psi.reshape(self.shape)

            return_flat = True

        elif psi.shape != self.shape:
            raise ValueError(f"psi must have shape {self.shape} or ({self.size},).")

        result = self.apply_kinetic(psi) + (self.potential - potential_shift) * psi

        if return_flat:
            return result.ravel()

        return result

    def hamiltonian_matrix(
        self,
        force: bool = False,
    ) -> FloatArray:
        """
        Explicitly construct the dense 3D Hamiltonian.

        This is ONLY intended for small debugging grids.

        The matrix dimension is

            N = nx * ny * nz

        so memory scales as N^2.
        """
        if self.size > 2048 and not force:
            raise MemoryError(
                "Dense 3D Hamiltonian disabled for "
                f"{self.size} grid points. "
                "Use the matrix-free iterative solver."
            )

        Tx = kinetic_matrix_1d(
            self.x,
            mass=self.mass,
        )

        Ty = kinetic_matrix_1d(
            self.y,
            mass=self.mass,
        )

        Tz = kinetic_matrix_1d(
            self.z,
            mass=self.mass,
        )

        Ix = np.eye(self.x.size)
        Iy = np.eye(self.y.size)
        Iz = np.eye(self.z.size)

        #
        # C-order flattening:
        #
        # z is the fastest index,
        # then y,
        # then x.
        #
        T = (
            np.kron(
                np.kron(
                    Tx,
                    Iy,
                ),
                Iz,
            )
            + np.kron(
                np.kron(
                    Ix,
                    Ty,
                ),
                Iz,
            )
            + np.kron(
                np.kron(
                    Ix,
                    Iy,
                ),
                Tz,
            )
        )

        V = np.diag(self.potential.ravel())

        H = T + V

        H = np.real_if_close(H).real

        return np.asarray(
            H,
            dtype=float,
        )

    def _solve_iterative(
        self,
        nstates: int,
        potential_shift: float,
        tol: float,
        maxiter: int | None,
    ) -> tuple[FloatArray, FloatArray]:
        operator = LinearOperator(
            shape=(
                self.size,
                self.size,
            ),
            matvec=lambda psi: self.matvec(
                psi,
                potential_shift=potential_shift,
            ),
            dtype=np.float64,
        )

        energies, vectors = eigsh(
            operator,
            k=nstates,
            which="SA",
            tol=tol,
            maxiter=maxiter,
        )

        order = np.argsort(energies)

        return (
            energies[order],
            vectors[:, order],
        )

    def _solve_dense(
        self,
        nstates: int | None,
    ) -> tuple[FloatArray, FloatArray]:
        H = self.hamiltonian_matrix()

        energies, vectors = np.linalg.eigh(H)

        if nstates is not None:
            energies = energies[:nstates]
            vectors = vectors[:, :nstates]

        return energies, vectors

    def solve(
        self,
        nstates: int = 10,
        method: SolverMethod = "auto",
        tol: float = 1e-10,
        maxiter: int | None = None,
        shift_potential: bool = True,
    ) -> FGH3DResult:
        """
        Solve the 3D FGH problem.

        For realistic calculations the matrix-free
        iterative method should be used.
        """
        if nstates <= 0:
            raise ValueError("nstates must be positive.")

        if nstates >= self.size:
            raise ValueError(
                "nstates must be smaller than the total number of grid points."
            )

        if method not in {
            "dense",
            "iterative",
            "auto",
        }:
            raise ValueError(f"Unknown method: {method}")

        if method == "auto":
            method = "iterative"

        if shift_potential:
            potential_shift = float(np.min(self.potential))
        else:
            potential_shift = 0.0

        if method == "iterative":
            energies, vectors = self._solve_iterative(
                nstates=nstates,
                potential_shift=potential_shift,
                tol=tol,
                maxiter=maxiter,
            )

            energies = energies + potential_shift

        else:
            #
            # Dense Hamiltonian currently contains
            # the original unshifted V.
            #
            energies, vectors = self._solve_dense(
                nstates=nstates,
            )

        #
        # eigsh/eigh eigenvectors satisfy
        #
        # sum_ijk |c_ijk|^2 = 1
        #
        # whereas the physical wavefunction satisfies
        #
        # dx dy dz sum_ijk |psi_ijk|^2 = 1.
        #
        vectors = vectors / np.sqrt(self.dv)

        wavefunctions = vectors.reshape(
            *self.shape,
            nstates,
        )

        wavefunctions = self._fix_signs(wavefunctions)

        return FGH3DResult(
            x=self.x.copy(),
            y=self.y.copy(),
            z=self.z.copy(),
            energies=np.asarray(
                energies,
                dtype=float,
            ),
            wavefunctions=np.asarray(
                wavefunctions,
                dtype=float,
            ),
        )

    @staticmethod
    def _fix_signs(
        wavefunctions: FloatArray,
    ) -> FloatArray:
        wavefunctions = wavefunctions.copy()

        nstates = wavefunctions.shape[-1]

        for state in range(nstates):
            psi = wavefunctions[
                ...,
                state,
            ]

            flat = psi.ravel()

            index = np.argmax(np.abs(flat))

            if flat[index] < 0.0:
                wavefunctions[..., state] *= -1.0

        return wavefunctions
