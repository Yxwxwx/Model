"""Sampled quantum geometry shared by SR-VMC and future TDVP drivers.

For real parameters the quantum geometric tensor is the covariance of
logarithmic derivatives,

``S = <(O - <O>)^T (O - <O>)>``.

``SampledQGT`` stores the centered, square-root-weighted Jacobian ``W``.  It can
solve either the parameter-space system ``(W^T W + lambda I) x = f`` or the
algebraically equivalent sample-space MinSR system.  The target vector is
generic; only ``energy_sr_direction`` supplies the energy-specific covariance,
which keeps the geometry reusable by real/imaginary-time TDVP integrators.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn

from qcnqs.autodiff import log_derivative_matrix

QGTMethod = Literal["auto", "dense", "minsr"]


@dataclass(frozen=True)
class QGTSolveResult:
    """Preconditioned direction and diagnostics from a regularized QGT solve."""

    update: torch.Tensor
    force: torch.Tensor
    method: Literal["dense", "minsr"]
    diagonal_shift: float
    solve_attempts: int


def _safe_cholesky_solve(
    matrix: torch.Tensor,
    rhs: torch.Tensor,
    diagonal_shift: float,
    *,
    max_attempts: int = 8,
    minimum_shift: float = 1.0e-12,
    shift_multiplier: float = 10.0,
) -> tuple[torch.Tensor, float, int]:
    """Solve a positive-semidefinite system with adaptive diagonal recovery.

    QGT estimates can be rank deficient when samples repeat or parameter
    directions are redundant.  ``cholesky_ex`` reports failure without raising
    inside the CUDA kernel.  The solve is performed in FP64 even if a future
    model uses a lower precision, and the shift is increased geometrically only
    when factorization or finiteness checks fail.
    """

    shift = float(diagonal_shift)
    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        regularized = 0.5 * (matrix + matrix.T)
        if shift:
            regularized = regularized + shift * torch.eye(
                matrix.shape[0],
                dtype=matrix.dtype,
                device=matrix.device,
            )
        try:
            factor, info = torch.linalg.cholesky_ex(regularized.to(torch.float64))
            if bool(torch.any(info)):
                raise torch.linalg.LinAlgError("Cholesky factorization failed")
            solution = torch.cholesky_solve(
                rhs.to(torch.float64).unsqueeze(1),
                factor,
            ).squeeze(1)
            solution = solution.to(rhs.dtype)
            if not bool(torch.all(torch.isfinite(solution))):
                raise torch.linalg.LinAlgError("QGT solve produced non-finite values")
            return solution, shift, attempt
        except (RuntimeError, torch.linalg.LinAlgError) as error:
            last_error = error
            shift = shift * shift_multiplier if shift else minimum_shift
    raise torch.linalg.LinAlgError(
        f"QGT solve failed after {max_attempts} attempts; final shift={shift:.3e}"
    ) from last_error


class SampledQGT:
    """Materialized real QGT represented by a weighted Jacobian factor.

    ``weighted = sqrt(p) * (O - <O>)`` has shape ``[samples, parameters]``.
    Keeping this factor rather than eagerly constructing a square QGT permits
    automatic selection of the smaller parameter- or sample-space system.
    """

    def __init__(
        self,
        log_derivatives: torch.Tensor,
        probabilities: torch.Tensor,
        *,
        validate: bool = True,
    ) -> None:
        if log_derivatives.ndim != 2:
            raise ValueError("log derivatives must have shape [samples, parameters]")
        if probabilities.shape != (log_derivatives.shape[0],):
            raise ValueError("probabilities must have one weight per sample")
        probabilities = probabilities.to(log_derivatives)
        total_probability = probabilities.sum()
        if validate and (bool(torch.any(probabilities < 0)) or not bool(total_probability > 0)):
            raise ValueError("probabilities must be non-negative with positive sum")
        self.log_derivatives = log_derivatives.detach()
        self.probabilities = probabilities / total_probability
        mean = (self.probabilities[:, None] * self.log_derivatives).sum(dim=0, keepdim=True)
        self.centered = self.log_derivatives - mean
        self.weighted = self.centered * torch.sqrt(self.probabilities)[:, None]

    @property
    def n_samples(self) -> int:
        return self.log_derivatives.shape[0]

    @property
    def n_parameters(self) -> int:
        return self.log_derivatives.shape[1]

    @classmethod
    def from_model(
        cls,
        model: nn.Module,
        states: torch.Tensor,
        probabilities: torch.Tensor,
    ) -> SampledQGT:
        """Differentiate a PyTorch model and construct its sampled geometry."""

        # Samplers construct non-negative normalized count/exact weights.  Skip
        # redundant device-to-host validation reductions in the VMC hot path.
        return cls(log_derivative_matrix(model, states), probabilities, validate=False)

    def force(self, centered_target: torch.Tensor) -> torch.Tensor:
        """Return ``Cov(O, target)`` in flattened parameter order."""

        if centered_target.shape != (self.n_samples,):
            raise ValueError("target must have one value per sample")
        target = centered_target.to(self.log_derivatives)
        return self.centered.T @ (self.probabilities * target)

    def solve(
        self,
        centered_target: torch.Tensor,
        *,
        diagonal_shift: float,
        method: QGTMethod = "auto",
        max_attempts: int = 8,
    ) -> QGTSolveResult:
        """Solve the regularized QGT system in dense or MinSR form.

        ``auto`` chooses the smaller square matrix: parameter space when
        ``n_parameters <= n_samples`` and sample space otherwise.  Both paths
        are mathematically identical for a positive diagonal shift.
        """

        if method == "auto":
            selected: Literal["dense", "minsr"] = (
                "dense" if self.n_parameters <= self.n_samples else "minsr"
            )
        elif method in {"dense", "minsr"}:
            selected = method
        else:
            raise ValueError(f"unknown QGT method: {method}")
        target = centered_target.to(self.log_derivatives)
        force = self.force(target)
        if selected == "dense":
            matrix = self.weighted.T @ self.weighted
            update, shift, attempts = _safe_cholesky_solve(
                matrix,
                force,
                diagonal_shift,
                max_attempts=max_attempts,
            )
        else:
            matrix = self.weighted @ self.weighted.T
            rhs = torch.sqrt(self.probabilities) * target
            dual, shift, attempts = _safe_cholesky_solve(
                matrix,
                rhs,
                diagonal_shift,
                max_attempts=max_attempts,
            )
            update = self.weighted.T @ dual
        return QGTSolveResult(
            update=update,
            force=force,
            method=selected,
            diagonal_shift=shift,
            solve_attempts=attempts,
        )


def energy_sr_direction(
    qgt: SampledQGT,
    local_energies: torch.Tensor,
    *,
    diagonal_shift: float,
    method: QGTMethod = "auto",
) -> QGTSolveResult:
    """Build the real-energy gradient target and solve for the SR direction."""

    mean = (qgt.probabilities * local_energies).sum()
    # For real parameters d<E>/dtheta = 2 Cov(O, E_local).
    target = 2.0 * (local_energies - mean)
    return qgt.solve(target, diagonal_shift=diagonal_shift, method=method)
