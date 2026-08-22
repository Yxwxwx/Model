"""Framework-level VMC orchestration independent of concrete QC components.

``VMCDriver`` consumes structural contracts for a connected operator, state
space, and sampler.  A user script selects the concrete Hamiltonian and fermion
sampler.  This direction of dependency is important:
the optimization loop can be tested with small stand-ins and does not force a
future TDVP driver to inherit chemistry- or sampler-specific behavior.

Stage timings synchronize only at explicit measurement boundaries.  Numerical
work inside a stage remains asynchronous, batched PyTorch execution.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
from math import sqrt
from time import perf_counter

import torch
from torch import nn

from qcnqs.geometry import QGTMethod, QGTSolveResult, SampledQGT, energy_sr_direction
from qcnqs.hilbert import SectorBasis
from qcnqs.interfaces import (
    ConnectedOperator,
    SampleBatch,
    StateSpace,
    VariationalSampler,
)
from qcnqs.local_energy import LocalEnergyResult, local_energy
from qcnqs.wavefunction import LogWaveFunction


@dataclass(frozen=True)
class StageTimings:
    """Synchronized wall times for the four VMC pipeline stages."""

    sampling_seconds: float
    local_energy_seconds: float
    qgt_seconds: float
    update_seconds: float

    @property
    def total_seconds(self) -> float:
        return (
            self.sampling_seconds
            + self.local_energy_seconds
            + self.qgt_seconds
            + self.update_seconds
        )


@dataclass(frozen=True)
class EnergyEstimate:
    """Weighted energy statistics and the underlying reusable batch data."""

    mean: float
    variance: float
    standard_error: float
    unique_samples: int
    local_energy: LocalEnergyResult
    samples: SampleBatch


@dataclass(frozen=True)
class VMCStep:
    """Serializable diagnostics for one sample/estimate/SR/update transaction."""

    iteration: int
    energy: float
    variance: float
    standard_error: float
    unique_samples: int
    active_connections: int
    unique_connected_states: int
    acceptance_rate: float
    valid_proposal_fraction: float
    gradient_norm: float
    update_norm: float
    qgt_method: str
    diagonal_shift: float
    qgt_solve_attempts: int
    timings: StageTimings

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class VMCResult:
    """Complete optimization trace, elapsed time, and optional final estimate."""

    steps: tuple[VMCStep, ...]
    training_seconds: float
    final_estimate: EnergyEstimate | None


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _timed(device: torch.device, function: Callable[[], object]) -> tuple[object, float]:
    """Measure one asynchronous PyTorch stage with synchronization at its edges."""

    _synchronize(device)
    start = perf_counter()
    result = function()
    _synchronize(device)
    return result, perf_counter() - start


@torch.no_grad()
def apply_flat_update(model: nn.Module, update: torch.Tensor, step_size: float) -> None:
    """Apply ``theta <- theta - step_size * update`` in named-parameter order.

    The same deterministic order is used by ``log_derivative_matrix`` when it
    flattens per-parameter Jacobian blocks.  Updating parameter views directly
    avoids constructing an optimizer state for plain SR-preconditioned SGD.
    """

    offset = 0
    for _, parameter in model.named_parameters():
        count = parameter.numel()
        value = update[offset : offset + count].reshape_as(parameter)
        parameter.add_(value.to(parameter), alpha=-step_size)
        offset += count
    if offset != update.numel():
        raise ValueError("flat update size does not match model parameters")


class VMCDriver:
    """Stochastic-reconfiguration VMC driver for real PyTorch wave functions.

    The driver owns orchestration and scalar hyperparameters only.  Operator
    connectivity, state constraints, Markov transitions, wave-function values,
    and sampled geometry remain replaceable components behind narrow contracts.
    """

    def __init__(
        self,
        operator: ConnectedOperator,
        space: StateSpace,
        model: nn.Module,
        sampler: VariationalSampler,
        *,
        learning_rate: float = 0.05,
        diagonal_shift: float = 0.05,
        qgt_method: QGTMethod = "auto",
    ) -> None:
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if diagonal_shift < 0:
            raise ValueError("diagonal_shift must be non-negative")
        if operator.device != sampler.device:
            raise ValueError("operator and sampler devices differ")
        if sampler.space != space:
            raise ValueError("sampler and driver state spaces differ")
        if not isinstance(model, LogWaveFunction):
            raise TypeError("model must implement log_amplitude(states)")
        self.operator = operator
        self.space = space
        self.model = model
        self.sampler = sampler
        self.learning_rate = float(learning_rate)
        self.diagonal_shift = float(diagonal_shift)
        self.qgt_method = qgt_method
        self.device = operator.device
        self.iteration = 0

    def _estimate_from_samples(self, samples: SampleBatch) -> EnergyEstimate:
        result = local_energy(
            self.operator,
            self.model,
            samples.states,
            space=self.space,
            sample_amplitudes=samples.amplitudes,
            validate=False,
        )
        mean_tensor = (samples.probabilities * result.values).sum()
        centered = result.values - mean_tensor
        variance_tensor = (samples.probabilities * centered.square()).sum()
        mean = float(mean_tensor.item())
        variance = float(variance_tensor.item())
        error = (
            0.0 if samples.exact_probabilities else sqrt(max(variance, 0.0) / samples.total_samples)
        )
        return EnergyEstimate(
            mean=mean,
            variance=variance,
            standard_error=error,
            unique_samples=samples.states.shape[0],
            local_energy=result,
            samples=samples,
        )

    def estimate(self, *, discard: bool = False) -> EnergyEstimate:
        """Draw one batch and return its weighted energy statistics."""

        samples = self.sampler.sample(discard=discard)
        return self._estimate_from_samples(samples)

    def step(self, *, discard: bool = False) -> VMCStep:
        """Execute one sample, local-energy, QGT, and parameter-update cycle."""

        sampled, sampling_seconds = _timed(
            self.device,
            lambda: self.sampler.sample(discard=discard),
        )
        assert isinstance(sampled, SampleBatch)
        estimate, local_energy_seconds = _timed(
            self.device,
            lambda: self._estimate_from_samples(sampled),
        )
        assert isinstance(estimate, EnergyEstimate)

        def solve_qgt() -> tuple[SampledQGT, QGTSolveResult]:
            qgt = SampledQGT.from_model(
                self.model,
                sampled.states,
                sampled.probabilities,
            )
            direction = energy_sr_direction(
                qgt,
                estimate.local_energy.values,
                diagonal_shift=self.diagonal_shift,
                method=self.qgt_method,
            )
            return qgt, direction

        solved, qgt_seconds = _timed(self.device, solve_qgt)
        _, direction = solved

        def update_parameters() -> None:
            apply_flat_update(self.model, direction.update, self.learning_rate)
            self.sampler.refresh()

        _, update_seconds = _timed(self.device, update_parameters)
        timings = StageTimings(
            sampling_seconds=sampling_seconds,
            local_energy_seconds=local_energy_seconds,
            qgt_seconds=qgt_seconds,
            update_seconds=update_seconds,
        )
        step = VMCStep(
            iteration=self.iteration,
            energy=estimate.mean,
            variance=estimate.variance,
            standard_error=estimate.standard_error,
            unique_samples=estimate.unique_samples,
            active_connections=estimate.local_energy.active_connections,
            unique_connected_states=estimate.local_energy.unique_connected_states,
            acceptance_rate=sampled.stats.acceptance_rate,
            valid_proposal_fraction=sampled.stats.valid_fraction,
            gradient_norm=float(direction.force.norm().item()),
            update_norm=float(direction.update.norm().item()),
            qgt_method=direction.method,
            diagonal_shift=direction.diagonal_shift,
            qgt_solve_attempts=direction.solve_attempts,
            timings=timings,
        )
        self.iteration += 1
        return step

    def run(
        self,
        iterations: int,
        *,
        final_estimate: bool = True,
        callback: Callable[[VMCStep], None] | None = None,
    ) -> VMCResult:
        """Run a fixed number of VMC updates and optionally estimate the final state."""

        if iterations < 0:
            raise ValueError("iterations must be non-negative")
        _synchronize(self.device)
        start = perf_counter()
        steps = []
        for _ in range(iterations):
            step = self.step(discard=self.iteration == 0)
            steps.append(step)
            if callback is not None:
                callback(step)
        terminal = self.estimate(discard=True) if final_estimate else None
        _synchronize(self.device)
        elapsed = perf_counter() - start
        return VMCResult(tuple(steps), elapsed, terminal)


@torch.no_grad()
def exact_model_expectation(
    operator: ConnectedOperator,
    wave_function: LogWaveFunction,
    basis: SectorBasis,
    *,
    batch_size: int = 2048,
    missing_tolerance: float = 1.0e-10,
) -> float:
    """Compute ``<psi|H|psi>/<psi|psi>`` over an enumerated finite sector.

    The complete variational vector is assembled in stable log form.  Operator
    rows are then applied in batches, so the diagnostic avoids materializing a
    dense Hamiltonian even though the full state vector is enumerated.
    """

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if basis.states.device != operator.device:
        raise ValueError("basis and operator devices differ")
    logabs_parts = []
    sign_parts = []
    for begin in range(0, basis.states.shape[0], batch_size):
        amplitudes = wave_function.log_amplitude(basis.states[begin : begin + batch_size])
        logabs_parts.append(amplitudes.logabs)
        sign_parts.append(amplitudes.sign)
    logabs = torch.cat(logabs_parts)
    sign = torch.cat(sign_parts)
    shift = logabs[torch.isfinite(logabs)].max()
    vector = sign * torch.exp(logabs - shift)
    h_vector_parts = []
    for begin in range(0, basis.states.shape[0], batch_size):
        states = basis.states[begin : begin + batch_size]
        connections = operator.connections(states, validate=False)
        columns, valid = basis.index(connections.states)
        missing = ~valid & (connections.matrix_elements.abs() > missing_tolerance)
        if bool(torch.any(missing)):
            largest = connections.matrix_elements[missing].abs().max().item()
            raise ValueError(f"basis omits a connected state with |H|={largest:.3e}")
        coefficients = torch.where(valid, connections.matrix_elements, 0.0)
        h_vector_parts.append((coefficients * vector[columns]).sum(dim=1))
    h_vector = torch.cat(h_vector_parts)
    return float((vector @ h_vector / vector.square().sum()).item())
