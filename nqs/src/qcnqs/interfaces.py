"""Narrow contracts shared by the operator, sampler, and driver layers.

This module contains transport objects and structural protocols only.  Concrete
Hamiltonians and Markov-chain implementations deliberately live elsewhere.
Keeping these contracts free of domain implementations prevents the
VMC driver from depending on a particular Hamiltonian or sampler class.

Tensor conventions
------------------
States are boolean tensors whose last axis stores spin-orbital occupations.  A
batch of states therefore has shape ``[..., n_spin_orbitals]``.  All numerical
weights use the wave function's floating-point dtype and stay on the execution
device; diagnostic Python scalars are produced only after a sampling stage ends.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import torch

from qcnqs.wavefunction import LogAmplitude


@dataclass(frozen=True)
class Connections:
    """Fixed-width sparse rows returned by a connected operator.

    ``states[b, c]`` is the determinant connected to input row ``b`` through
    slot ``c`` and ``matrix_elements[b, c]`` is the corresponding matrix
    element.  Fixed width is intentional: it lets PyTorch evaluate all
    determinant transitions with gather/scatter kernels and avoids Python
    lists whose lengths vary from sample to sample.
    """

    states: torch.Tensor
    matrix_elements: torch.Tensor

    def __post_init__(self) -> None:
        if self.states.ndim != 3 or self.states.dtype != torch.bool:
            raise ValueError("connected states must have shape [batch, connections, orbitals]")
        if self.matrix_elements.shape != self.states.shape[:2]:
            raise ValueError("matrix elements must match the connection batch")


@dataclass(frozen=True)
class SamplerStats:
    """Aggregate proposal counters copied to Python once per sample batch."""

    attempted: int
    valid: int
    accepted: int

    @property
    def valid_fraction(self) -> float:
        return self.valid / self.attempted if self.attempted else 0.0

    @property
    def acceptance_rate(self) -> float:
        return self.accepted / self.valid if self.valid else 0.0


@dataclass(frozen=True)
class SampleBatch:
    """Unique configurations and exact multiplicity weights for one VMC batch.

    MCMC chains can revisit the same determinant many times.  ``states`` stores
    each determinant once, while ``counts`` and ``probabilities`` preserve the
    estimator exactly.  This batch-local compression reduces wave-function,
    local-energy, and Jacobian work without changing the number of Markov-chain
    samples or introducing a persistent cache.
    """

    states: torch.Tensor
    counts: torch.Tensor
    probabilities: torch.Tensor
    inverse: torch.Tensor
    raw_states: torch.Tensor
    amplitudes: LogAmplitude
    stats: SamplerStats
    exact_probabilities: bool = False

    @property
    def total_samples(self) -> int:
        return self.raw_states.shape[0]


@runtime_checkable
class StateSpace(Protocol):
    """Constraint needed by generic estimators and samplers."""

    @property
    def n_spin_orbitals(self) -> int: ...

    @property
    def requires_projection(self) -> bool:
        """Whether connected states must be projected back into this space."""

        ...

    def contains(self, states: torch.Tensor) -> torch.Tensor:
        """Return one validity flag for every state prefix index."""

        ...


@runtime_checkable
class ConnectedOperator(Protocol):
    """Operator interface required by exact local-energy evaluation."""

    device: torch.device

    @property
    def n_spin_orbitals(self) -> int: ...

    def connections(
        self,
        states: torch.Tensor,
        *,
        validate: bool = True,
    ) -> Connections:
        """Return all connected states and matrix elements for each input row."""

        ...


@runtime_checkable
class VariationalSampler(Protocol):
    """Minimal persistent-sampler contract consumed by an optimization driver."""

    device: torch.device
    space: StateSpace

    def sample(self, *, discard: bool = False) -> SampleBatch:
        """Return one weighted sample batch."""

        ...

    def refresh(self) -> None:
        """Refresh cached amplitudes after the model parameters change."""

        ...
