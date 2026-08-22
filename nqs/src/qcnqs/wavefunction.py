"""Minimal real log-wave-function value interface.

Amplitudes are represented as a detached discrete sign and a differentiable
logarithmic magnitude.  This prevents overflow in determinant ratios and keeps
the model contract small enough for samplers, local-energy estimators, exact
diagnostics, and future time-evolution drivers to share.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import torch


@dataclass(frozen=True)
class LogAmplitude:
    """Stable ``psi = sign * exp(logabs)`` representation for real states."""

    logabs: torch.Tensor
    sign: torch.Tensor

    def amplitude(self) -> torch.Tensor:
        """Materialize amplitudes when their dynamic range is known to be safe."""

        return self.sign * torch.exp(self.logabs)


@runtime_checkable
class LogWaveFunction(Protocol):
    """Minimal state interface shared by VMC, exact diagnostics, and TDVP."""

    def log_amplitude(self, states: torch.Tensor) -> LogAmplitude:
        """Evaluate signs and log magnitudes for a state batch."""

        ...
