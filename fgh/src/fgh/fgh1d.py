from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt
from scipy.sparse.linalg import LinearOperator, eigsh

FloatArray = npt.NDArray[np.float64]
Potential = Callable[[FloatArray], npt.ArrayLike] | npt.ArrayLike
SolverMethod = Literal["dense", "iterative", "auto"]


@dataclass(frozen=True)
class FGH1DResult:
    """
    Result of a one-dimensional Fourier Grid Hamiltonian calculation.

    All quantities are in atomic units:
        x        : bohr
        energies : Hartree
        mass     : electron masses
    """

    x: FloatArray
    energies: FloatArray
    wavefunctions: FloatArray

    @property
    def dx(self) -> float:
        return float(self.x[1] - self.x[0])

    @property
    def nstates(self) -> int:
        return self.energies.size

    @property
    def excitation_energies(self) -> FloatArray:
        """
        Excitation energies relative to the ground state.
        """
        return self.energies - self.energies[0]

    def density(self, state: int = 0) -> FloatArray:
        """
        Probability density |psi(x)|^2.
        """
        psi = self.wavefunctions[:, state]
        return np.abs(psi) ** 2

    def matrix_element(
        self,
        operator: Callable[[FloatArray], npt.ArrayLike] | npt.ArrayLike,
        bra: int,
        ket: int,
    ) -> complex:
        """
        Compute a matrix element

            <bra | O(x) | ket>

        for an operator diagonal in coordinate space.

        Examples
        --------
        <x>:
            result.matrix_element(lambda x: x, 0, 0)

        <x^2>:
            result.matrix_element(lambda x: x**2, 0, 0)

        transition matrix element <0|x|1>:
            result.matrix_element(lambda x: x, 0, 1)
        """
        if callable(operator):
            op = np.asarray(operator(self.x))
        else:
            op = np.asarray(operator)

        if op.shape != self.x.shape:
            raise ValueError(
                f"Operator shape {op.shape} does not match grid shape {self.x.shape}."
            )

        psi_bra = self.wavefunctions[:, bra]
        psi_ket = self.wavefunctions[:, ket]

        value = self.dx * np.vdot(
            psi_bra,
            op * psi_ket,
        )

        return complex(value)

    def expectation(
        self,
        operator: Callable[[FloatArray], npt.ArrayLike] | npt.ArrayLike,
        state: int = 0,
    ) -> float:
        """
        Compute <state | O(x) | state>.
        """
        value = self.matrix_element(
            operator=operator,
            bra=state,
            ket=state,
        )

        return float(np.real_if_close(value).real)

    def overlap(
        self,
        state_i: int,
        state_j: int,
    ) -> complex:
        """
        Compute <psi_i | psi_j>.
        """
        psi_i = self.wavefunctions[:, state_i]
        psi_j = self.wavefunctions[:, state_j]

        return complex(self.dx * np.vdot(psi_i, psi_j))


class FGH1D:
    """
    One-dimensional Fourier Grid Hamiltonian solver.

    The Hamiltonian is

        H = -(1 / 2m) d^2/dx^2 + V(x)

    using atomic units (hbar = 1).

    The kinetic-energy operator is evaluated through a discrete
    Fourier representation:

        T psi =
            IFFT[
                k^2 / (2m) * FFT[psi]
            ].
    """

    def __init__(
        self,
        x: npt.ArrayLike,
        potential: Potential,
        mass: float = 1.0,
    ):
        self.x = self._validate_grid(x)

        if mass <= 0.0:
            raise ValueError("mass must be positive.")

        self.mass = float(mass)

        self.dx = float(self.x[1] - self.x[0])
        self.ngrid = self.x.size

        self.potential = self._evaluate_potential(potential)

        # Fourier wave vectors:
        #
        # k = 2*pi * fftfreq(N, dx)
        #
        self.k = (
            2.0
            * np.pi
            * np.fft.fftfreq(
                self.ngrid,
                d=self.dx,
            )
        )

        # Kinetic energy in momentum representation.
        self.kinetic_k = self.k**2 / (2.0 * self.mass)

    @staticmethod
    def _validate_grid(
        x: npt.ArrayLike,
    ) -> FloatArray:
        x = np.asarray(x, dtype=float)

        if x.ndim != 1:
            raise ValueError("x must be a one-dimensional array.")

        if x.size < 2:
            raise ValueError("x must contain at least two points.")

        if not np.all(np.isfinite(x)):
            raise ValueError("x contains non-finite values.")

        dx = x[1] - x[0]

        if dx <= 0.0:
            raise ValueError("x must be strictly increasing.")

        if not np.allclose(
            np.diff(x),
            dx,
            rtol=1e-12,
            atol=1e-14,
        ):
            raise ValueError("FGH requires a uniform grid.")

        return x

    def _evaluate_potential(
        self,
        potential: Potential,
    ) -> FloatArray:
        if callable(potential):
            v = np.asarray(
                potential(self.x),
                dtype=float,
            )
        else:
            v = np.asarray(
                potential,
                dtype=float,
            )

        if v.shape != self.x.shape:
            raise ValueError(
                f"Potential shape {v.shape} does not match grid shape {self.x.shape}."
            )

        if not np.all(np.isfinite(v)):
            raise ValueError("Potential contains non-finite values.")

        return v

    def apply_kinetic(
        self,
        psi: npt.ArrayLike,
    ) -> FloatArray:
        """
        Apply the kinetic-energy operator to a grid vector.

        T psi =
            IFFT[
                k^2/(2m) * FFT(psi)
            ]
        """
        psi = np.asarray(psi, dtype=float)

        if psi.shape != (self.ngrid,):
            raise ValueError(f"psi must have shape ({self.ngrid},).")

        psi_k = np.fft.fft(
            psi,
            norm="ortho",
        )

        tpsi_k = self.kinetic_k * psi_k

        tpsi = np.fft.ifft(
            tpsi_k,
            norm="ortho",
        )

        # For a real Hamiltonian acting on a real vector,
        # imaginary parts are only numerical roundoff.
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
        Apply

            H = T + V - potential_shift

        without explicitly constructing the Hamiltonian matrix.
        """
        psi = np.asarray(psi, dtype=float)

        return self.apply_kinetic(psi) + (self.potential - potential_shift) * psi

    def kinetic_matrix(self) -> FloatArray:
        """
        Explicitly construct the dense kinetic-energy matrix.

        This method is primarily intended for testing and
        small-grid reference calculations.
        """
        identity = np.eye(
            self.ngrid,
            dtype=float,
        )

        # Each column of the identity is a coordinate-space
        # basis vector. Apply T to all columns simultaneously.
        identity_k = np.fft.fft(
            identity,
            axis=0,
            norm="ortho",
        )

        t_k = self.kinetic_k[:, None] * identity_k

        T = np.fft.ifft(
            t_k,
            axis=0,
            norm="ortho",
        )

        T = np.real_if_close(T).real

        # Remove tiny numerical asymmetry.
        T = 0.5 * (T + T.T)

        return np.asarray(T, dtype=float)

    def hamiltonian_matrix(
        self,
        potential_shift: float = 0.0,
    ) -> FloatArray:
        """
        Explicit dense Hamiltonian matrix.

        Intended for testing and small systems.
        """
        T = self.kinetic_matrix()

        V = np.diag(self.potential - potential_shift)

        return T + V

    @staticmethod
    def _normalize_wavefunctions(
        wavefunctions: FloatArray,
        dx: float,
    ) -> FloatArray:
        """
        Convert Euclidean-normalized eigenvectors into
        continuously normalized grid wavefunctions:

            dx * sum_i |psi_i|^2 = 1.
        """
        wavefunctions = wavefunctions / np.sqrt(dx)

        return wavefunctions

    @staticmethod
    def _fix_signs(
        wavefunctions: FloatArray,
    ) -> FloatArray:
        """
        Remove the arbitrary +/- sign of real eigenvectors.

        The grid point with the largest absolute amplitude
        is chosen to be positive.

        This is useful for deterministic plotting and
        comparison between different solvers.
        """
        wavefunctions = wavefunctions.copy()

        for state in range(wavefunctions.shape[1]):
            psi = wavefunctions[:, state]

            index = np.argmax(np.abs(psi))

            if psi[index] < 0.0:
                wavefunctions[:, state] *= -1.0

        return wavefunctions

    def _solve_dense(
        self,
        potential_shift: float,
    ) -> tuple[FloatArray, FloatArray]:
        H = self.hamiltonian_matrix(potential_shift=potential_shift)

        energies, eigenvectors = np.linalg.eigh(H)

        return energies, eigenvectors

    def _solve_iterative(
        self,
        nstates: int,
        potential_shift: float,
        tol: float,
        maxiter: int | None,
    ) -> tuple[FloatArray, FloatArray]:
        if nstates >= self.ngrid:
            raise ValueError("Iterative solver requires nstates < ngrid.")

        operator = LinearOperator(
            shape=(
                self.ngrid,
                self.ngrid,
            ),
            matvec=lambda psi: self.matvec(
                psi,
                potential_shift=potential_shift,
            ),
            dtype=np.float64,
        )

        energies, eigenvectors = eigsh(
            operator,
            k=nstates,
            which="SA",
            tol=tol,
            maxiter=maxiter,
        )

        # eigsh does not guarantee sorted eigenvalues.
        order = np.argsort(energies)

        energies = energies[order]
        eigenvectors = eigenvectors[:, order]

        return energies, eigenvectors

    def solve(
        self,
        nstates: int | None = None,
        method: SolverMethod = "auto",
        tol: float = 1e-12,
        maxiter: int | None = None,
        shift_potential: bool = True,
    ) -> FGH1DResult:
        """
        Solve the FGH eigenvalue problem.

        Parameters
        ----------
        nstates
            Number of lowest states requested.

            If None, all states are returned and the dense
            solver is used.

        method
            "dense"
                Explicitly construct H and diagonalize it.

            "iterative"
                Matrix-free FFT Hamiltonian with scipy eigsh.

            "auto"
                Use the iterative solver when only a subset
                of states is requested; otherwise dense.

        tol
            Tolerance passed to scipy eigsh.

        maxiter
            Maximum eigsh iterations.

        shift_potential
            Internally subtract min(V) before diagonalization
            and add it back afterwards.

            This is particularly useful for electronic
            potential-energy surfaces where the absolute
            electronic energy may be much larger than the
            vibrational energy scale.

        Returns
        -------
        FGH1DResult
        """
        if nstates is not None:
            if nstates <= 0:
                raise ValueError("nstates must be positive.")

            if nstates > self.ngrid:
                raise ValueError("nstates cannot exceed ngrid.")

        if method not in {
            "dense",
            "iterative",
            "auto",
        }:
            raise ValueError(f"Unknown solver method: {method}")

        if shift_potential:
            potential_shift = float(np.min(self.potential))
        else:
            potential_shift = 0.0

        if method == "auto":
            if nstates is None:
                method = "dense"
            else:
                method = "iterative"

        if method == "dense":
            energies, eigenvectors = self._solve_dense(potential_shift)

            if nstates is not None:
                energies = energies[:nstates]
                eigenvectors = eigenvectors[:, :nstates]

        else:
            if nstates is None:
                raise ValueError("nstates must be specified for the iterative solver.")

            energies, eigenvectors = self._solve_iterative(
                nstates=nstates,
                potential_shift=potential_shift,
                tol=tol,
                maxiter=maxiter,
            )

        # Restore the absolute potential-energy zero.
        energies = energies + potential_shift

        wavefunctions = self._normalize_wavefunctions(
            eigenvectors,
            self.dx,
        )

        wavefunctions = self._fix_signs(wavefunctions)

        return FGH1DResult(
            x=self.x.copy(),
            energies=np.asarray(
                energies,
                dtype=float,
            ),
            wavefunctions=np.asarray(
                wavefunctions,
                dtype=float,
            ),
        )


# ----------------------------------------------------------------------
# Functional interface
#
# Keep these wrappers so simple calculations do not require explicitly
# constructing an FGH1D object.
# ----------------------------------------------------------------------


def kinetic_matrix_1d(
    x: npt.ArrayLike,
    mass: float = 1.0,
) -> FloatArray:
    solver = FGH1D(
        x=x,
        potential=np.zeros_like(np.asarray(x, dtype=float)),
        mass=mass,
    )

    return solver.kinetic_matrix()


def hamiltonian_1d(
    x: npt.ArrayLike,
    potential: Potential,
    mass: float = 1.0,
) -> FloatArray:
    solver = FGH1D(
        x=x,
        potential=potential,
        mass=mass,
    )

    return solver.hamiltonian_matrix()


def solve_fgh_1d(
    x: npt.ArrayLike,
    potential: Potential,
    mass: float = 1.0,
    nstates: int | None = None,
    method: SolverMethod = "auto",
    tol: float = 1e-12,
) -> tuple[FloatArray, FloatArray]:
    """
    Backwards-compatible functional interface.
    """
    solver = FGH1D(
        x=x,
        potential=potential,
        mass=mass,
    )

    result = solver.solve(
        nstates=nstates,
        method=method,
        tol=tol,
    )

    return (
        result.energies,
        result.wavefunctions,
    )
