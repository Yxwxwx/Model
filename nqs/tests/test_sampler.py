from __future__ import annotations

import torch

from qcnqs.hilbert import FermionHilbert
from qcnqs.sampler import FermionSampler
from qcnqs.wavefunction import LogAmplitude


class ConstantWaveFunction:
    def log_amplitude(self, states: torch.Tensor) -> LogAmplitude:
        shape = states.shape[:-1]
        return LogAmplitude(
            torch.zeros(shape, dtype=torch.float64, device=states.device),
            torch.ones(shape, dtype=torch.float64, device=states.device),
        )


def test_metropolis_is_reproducible_and_preserves_full_sector() -> None:
    hilbert = FermionHilbert(4, 2, 1)
    model = ConstantWaveFunction()
    options = dict(
        n_samples=32,
        n_chains=8,
        seed=29,
        device="cpu",
        burn_in_sweeps=2,
        sweep_size=4,
    )
    first = FermionSampler(hilbert, model, **options)
    second = FermionSampler(hilbert, model, **options)
    first_batch = first.sample(discard=True)
    second_batch = second.sample(discard=True)
    assert torch.equal(first_batch.raw_states, second_batch.raw_states)
    assert bool(torch.all(hilbert.contains(first_batch.raw_states)))
    assert first_batch.counts.sum().item() == 32
    assert first_batch.stats.valid == first_batch.stats.attempted
    assert first_batch.stats.accepted == first_batch.stats.valid


def test_point_group_rejection_never_changes_sector() -> None:
    hilbert = FermionHilbert(
        4,
        2,
        2,
        orbital_symmetries=(0, 1, 2, 3),
        target_symmetry=0,
    )
    sampler = FermionSampler(
        hilbert,
        ConstantWaveFunction(),
        n_samples=32,
        n_chains=8,
        seed=31,
        device="cpu",
        burn_in_sweeps=2,
        sweep_size=4,
    )
    batch = sampler.sample(discard=True)
    assert bool(torch.all(hilbert.contains(batch.raw_states)))
    assert 0 < batch.stats.valid <= batch.stats.attempted
    assert batch.stats.accepted == batch.stats.valid


def test_exact_sampler_draws_from_finite_sector() -> None:
    hilbert = FermionHilbert(3, 1, 1)
    sampler = FermionSampler(
        hilbert,
        ConstantWaveFunction(),
        n_samples=900,
        n_chains=9,
        seed=37,
        device="cpu",
        mode="exact",
    )
    batch = sampler.sample()
    assert bool(torch.all(hilbert.contains(batch.raw_states)))
    assert batch.counts.sum().item() == hilbert.sector_size()
    assert batch.states.shape[0] == hilbert.sector_size()
    expected = torch.full_like(batch.probabilities, 1.0 / hilbert.sector_size())
    torch.testing.assert_close(batch.probabilities, expected)
    assert batch.exact_probabilities
