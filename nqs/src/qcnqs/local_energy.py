"""Exact connected-operator local energy with batch-local state reuse.

For sampled determinant ``x``, the estimator evaluates

``E_local(x) = sum_y H[x, y] psi(y) / psi(x)``.

The operator supplies all exact connected states.  Repeated ``y`` states across
the batch are packed into integer keys and evaluated once, then scattered back
to their originating rows.  This is an exact algebraic reuse within one batch,
not a persistent cache or stochastic Hamiltonian approximation.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from qcnqs.hilbert import decode_states, encode_states
from qcnqs.interfaces import ConnectedOperator, StateSpace
from qcnqs.wavefunction import LogAmplitude, LogWaveFunction


@dataclass(frozen=True)
class LocalEnergyResult:
    """Local-energy values plus diagnostics about exact connection reuse."""

    values: torch.Tensor
    sample_amplitudes: LogAmplitude
    active_connections: int
    unique_connected_states: int


@torch.no_grad()
def local_energy(
    operator: ConnectedOperator,
    wave_function: LogWaveFunction,
    samples: torch.Tensor,
    *,
    space: StateSpace | None = None,
    sample_amplitudes: LogAmplitude | None = None,
    validate: bool = True,
) -> LocalEnergyResult:
    """Evaluate exact local energies for a determinant batch.

    Args:
        operator: Any fixed-width connected operator on the sample device.
        wave_function: Real log-amplitude provider for sample and connected states.
        samples: Boolean tensor with shape ``[batch, n_spin_orbitals]``.
        space: Optional projection constraint, such as a target point-group irrep.
        sample_amplitudes: Reuse amplitudes already computed by the sampler.
        validate: Check for zero sampled amplitudes at an external API boundary.
            Trusted VMC batches disable this check to avoid a CUDA-to-host sync.

    Returns:
        One local energy per input sample and connection-count diagnostics.
    """

    samples = samples.to(operator.device).bool()
    if sample_amplitudes is None:
        sample_amplitudes = wave_function.log_amplitude(samples)
    if validate and bool(torch.any(sample_amplitudes.sign == 0)):
        raise ValueError("sampled a determinant with zero wave-function amplitude")
    # Sampler/basis states have already crossed their validation boundary.  The
    # operator still checks metadata (shape/device/dtype) but skips particle
    # reductions that would synchronize CUDA on every optimization step.
    connections = operator.connections(samples, validate=False)
    values = connections.matrix_elements[:, 0].clone()
    off_diagonal = connections.states[:, 1:, :]
    coefficients = connections.matrix_elements[:, 1:]
    active = coefficients != 0
    if space is not None and space.requires_projection:
        active = active & space.contains(off_diagonal)
    rows = torch.arange(samples.shape[0], device=samples.device)[:, None]
    rows = rows.expand_as(coefficients)[active]
    connected = off_diagonal[active]
    active_coefficients = coefficients[active]
    if not connected.numel():
        return LocalEnergyResult(values, sample_amplitudes, 0, 0)

    keys = encode_states(connected)
    unique_keys, inverse = torch.unique(keys, sorted=False, return_inverse=True)
    unique_states = decode_states(unique_keys, operator.n_spin_orbitals)
    connected_amplitudes = wave_function.log_amplitude(unique_states)
    # Form psi(y)/psi(x) from a discrete sign and a log-magnitude difference.
    # Individual amplitudes are never materialized, avoiding overflow when the
    # neural determinant spans a large dynamic range.
    log_ratio = connected_amplitudes.logabs[inverse] - sample_amplitudes.logabs[rows]
    sign_ratio = connected_amplitudes.sign[inverse] / sample_amplitudes.sign[rows]
    contributions = active_coefficients * sign_ratio * torch.exp(log_ratio)
    values.index_add_(0, rows, contributions)
    return LocalEnergyResult(
        values=values,
        sample_amplitudes=sample_amplitudes,
        active_connections=int(connected.shape[0]),
        unique_connected_states=int(unique_keys.shape[0]),
    )
