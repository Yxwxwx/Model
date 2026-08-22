from __future__ import annotations

import torch

from qcnqs.geometry import SampledQGT, energy_sr_direction


def test_dense_and_minsr_updates_are_equivalent() -> None:
    generator = torch.Generator().manual_seed(17)
    derivatives = torch.randn((8, 11), generator=generator, dtype=torch.float64)
    probabilities = torch.rand(8, generator=generator, dtype=torch.float64)
    target = torch.randn(8, generator=generator, dtype=torch.float64)
    qgt = SampledQGT(derivatives, probabilities)
    dense = qgt.solve(target, diagonal_shift=0.07, method="dense")
    minsr = qgt.solve(target, diagonal_shift=0.07, method="minsr")
    torch.testing.assert_close(dense.update, minsr.update, atol=1.0e-10, rtol=1.0e-10)


def test_energy_force_contains_required_factor_two() -> None:
    derivatives = torch.tensor(
        [[0.2, 0.5], [0.7, -0.3], [-0.4, 0.1]],
        dtype=torch.float64,
    )
    probabilities = torch.tensor([0.2, 0.3, 0.5], dtype=torch.float64)
    energies = torch.tensor([-1.0, -0.4, 0.2], dtype=torch.float64)
    qgt = SampledQGT(derivatives, probabilities)
    result = energy_sr_direction(qgt, energies, diagonal_shift=0.1)
    mean_energy = (qgt.probabilities * energies).sum()
    expected = 2.0 * qgt.centered.T @ (qgt.probabilities * (energies - mean_energy))
    torch.testing.assert_close(result.force, expected)


def test_singular_qgt_increases_zero_shift() -> None:
    derivatives = torch.ones((2, 3), dtype=torch.float64)
    probabilities = torch.ones(2, dtype=torch.float64)
    qgt = SampledQGT(derivatives, probabilities)
    result = qgt.solve(
        torch.tensor([1.0, -1.0], dtype=torch.float64),
        diagonal_shift=0.0,
        method="dense",
    )
    assert result.diagonal_shift > 0.0
    assert result.solve_attempts > 1
    assert bool(torch.all(torch.isfinite(result.update)))
