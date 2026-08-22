from __future__ import annotations

import math

import torch

from qcnqs.hilbert import SectorBasis, encode_states
from qcnqs.hubbard import (
    Hubbard1D,
    Hubbard2D,
    HubbardHamiltonian,
    nearest_neighbor_bonds_1d,
    nearest_neighbor_bonds_2d,
)
from qcnqs.local_energy import local_energy
from qcnqs.models import DeterminantBackflow
from qcnqs.sampler import FermionSampler
from qcnqs.vmc import VMCDriver
from qcnqs.wavefunction import LogAmplitude


def _operate(state: torch.Tensor, orbital: int, create: bool) -> tuple[torch.Tensor, int] | None:
    if bool(state[orbital]) == create:
        return None
    sign = -1 if int(state[:orbital].sum()) % 2 else 1
    result = state.clone()
    result[orbital] = create
    return result, sign


def _explicit_hubbard_matrix(
    hamiltonian: HubbardHamiltonian,
    states: torch.Tensor,
) -> torch.Tensor:
    size = states.shape[0]
    index = {tuple(bool(value) for value in state): row for row, state in enumerate(states)}
    matrix = torch.zeros((size, size), dtype=hamiltonian.dtype)
    for column, state in enumerate(states):
        double_occupancy = (state[0::2] & state[1::2]).sum().to(hamiltonian.dtype)
        matrix[column, column] += hamiltonian.interaction * double_occupancy
        for left, right in hamiltonian.bonds:
            for spin in range(2):
                for source, target in ((left, right), (right, left)):
                    annihilated = _operate(state, 2 * source + spin, False)
                    if annihilated is None:
                        continue
                    intermediate, annihilation_sign = annihilated
                    created = _operate(intermediate, 2 * target + spin, True)
                    if created is None:
                        continue
                    connected, creation_sign = created
                    row = index[tuple(bool(value) for value in connected)]
                    matrix[row, column] += -hamiltonian.hopping * annihilation_sign * creation_sign
    return matrix


class _TableWaveFunction:
    def __init__(self, basis: SectorBasis, amplitudes: torch.Tensor) -> None:
        self.keys = basis.keys
        self.amplitudes = amplitudes

    def log_amplitude(self, states: torch.Tensor) -> LogAmplitude:
        indices = torch.searchsorted(self.keys, encode_states(states))
        values = self.amplitudes[indices]
        return LogAmplitude(torch.log(values.abs()), torch.sign(values))


def test_nearest_neighbor_lattice_bonds() -> None:
    assert nearest_neighbor_bonds_1d(4) == ((0, 1), (1, 2), (2, 3))
    assert nearest_neighbor_bonds_1d(4, periodic=True) == (
        (0, 1),
        (0, 3),
        (1, 2),
        (2, 3),
    )
    assert nearest_neighbor_bonds_2d(2, 3) == (
        (0, 1),
        (0, 3),
        (1, 2),
        (1, 4),
        (2, 5),
        (3, 4),
        (4, 5),
    )
    assert len(nearest_neighbor_bonds_2d(2, 3, periodic=True)) == 9


def test_1d_matrix_matches_explicit_second_quantization() -> None:
    hamiltonian = Hubbard1D(
        4,
        2,
        1,
        hopping=0.7,
        interaction=3.2,
        periodic=True,
    )
    basis = hamiltonian.space.enumerate()
    actual = hamiltonian.dense_matrix(basis)
    expected = _explicit_hubbard_matrix(hamiltonian, basis.states)
    torch.testing.assert_close(actual, expected, atol=1.0e-12, rtol=1.0e-12)
    torch.testing.assert_close(actual, actual.T, atol=1.0e-12, rtol=1.0e-12)


def test_2d_matrix_matches_explicit_second_quantization() -> None:
    hamiltonian = Hubbard2D(
        2,
        2,
        1,
        1,
        hopping=1.3,
        interaction=-0.4,
        periodic=False,
    )
    basis = hamiltonian.space.enumerate()
    actual = hamiltonian.dense_matrix(basis)
    expected = _explicit_hubbard_matrix(hamiltonian, basis.states)
    torch.testing.assert_close(actual, expected, atol=1.0e-12, rtol=1.0e-12)
    assert hamiltonian.n_connections == 1 + 4 * len(hamiltonian.bonds)


def test_two_site_ground_energy_has_closed_form() -> None:
    hopping = 1.0
    interaction = 4.0
    hamiltonian = Hubbard1D(2, 1, 1, hopping=hopping, interaction=interaction)
    matrix = hamiltonian.dense_matrix(hamiltonian.space.enumerate())
    actual = torch.linalg.eigvalsh(matrix)[0].item()
    expected = 0.5 * (interaction - math.sqrt(interaction**2 + 16.0 * hopping**2))
    assert math.isclose(actual, expected, abs_tol=1.0e-12)


def test_connections_and_local_energy_stay_in_the_fixed_sector() -> None:
    hamiltonian = Hubbard2D(2, 3, 2, 1, hopping=0.8, interaction=2.1)
    basis = hamiltonian.space.enumerate()
    connections = hamiltonian.connections(basis.states)
    assert bool(torch.all(hamiltonian.space.contains(connections.states)))

    generator = torch.Generator().manual_seed(47)
    amplitudes = torch.randn(basis.states.shape[0], generator=generator, dtype=torch.float64)
    amplitudes += torch.where(amplitudes >= 0, 0.4, -0.4)
    wave_function = _TableWaveFunction(basis, amplitudes)
    result = local_energy(
        hamiltonian,
        wave_function,
        basis.states,
        space=hamiltonian.space,
    )
    expected = hamiltonian.apply(basis, amplitudes) / amplitudes
    torch.testing.assert_close(result.values, expected, atol=1.0e-11, rtol=1.0e-11)


def test_hubbard_runs_through_exact_vmc_pipeline() -> None:
    torch.manual_seed(53)
    hamiltonian = Hubbard1D(3, 1, 1, hopping=1.0, interaction=2.0)
    model = DeterminantBackflow(6, 2, 3)
    sampler = FermionSampler(
        hamiltonian.space,
        model,
        n_samples=18,
        n_chains=3,
        seed=59,
        device="cpu",
        mode="exact",
    )
    driver = VMCDriver(
        hamiltonian,
        hamiltonian.space,
        model,
        sampler,
        learning_rate=0.01,
        diagonal_shift=0.1,
    )
    step = driver.step()
    assert math.isfinite(step.energy)
    assert math.isfinite(step.update_norm)
