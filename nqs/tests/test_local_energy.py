from __future__ import annotations

import torch

from qcnqs.assembly import hilbert_from_problem
from qcnqs.chemistry.problem import MolecularProblem
from qcnqs.hamiltonian import QCHamiltonian
from qcnqs.hilbert import SectorBasis, encode_states
from qcnqs.local_energy import local_energy
from qcnqs.wavefunction import LogAmplitude


class TableWaveFunction:
    def __init__(self, basis: SectorBasis, amplitudes: torch.Tensor) -> None:
        self.keys = basis.keys
        self.amplitudes = amplitudes

    def log_amplitude(self, states: torch.Tensor) -> LogAmplitude:
        keys = encode_states(states)
        indices = torch.searchsorted(self.keys, keys)
        if not bool(torch.all(self.keys[indices] == keys)):
            raise ValueError("state is outside the table")
        values = self.amplitudes[indices]
        return LogAmplitude(torch.log(values.abs()), torch.sign(values))


def _problem() -> MolecularProblem:
    generator = torch.Generator().manual_seed(19)
    h1e = torch.randn((3, 3), generator=generator, dtype=torch.float64)
    h1e = 0.5 * (h1e + h1e.T)
    eri = torch.randn((3, 3, 3, 3), generator=generator, dtype=torch.float64)
    eri = 0.5 * (eri + eri.transpose(0, 1))
    eri = 0.5 * (eri + eri.transpose(2, 3))
    eri = 0.5 * (eri + eri.permute(2, 3, 0, 1))
    return MolecularProblem(h1e, eri, 0.23, 1, 1, (0, 0, 0))


def test_local_energy_equals_explicit_h_psi_over_psi() -> None:
    problem = _problem()
    hilbert = hilbert_from_problem(problem, point_group=False)
    basis = hilbert.enumerate()
    hamiltonian = QCHamiltonian(problem)
    generator = torch.Generator().manual_seed(23)
    amplitudes = torch.randn(basis.states.shape[0], generator=generator, dtype=torch.float64)
    amplitudes += torch.where(amplitudes >= 0, 0.3, -0.3)
    model = TableWaveFunction(basis, amplitudes)
    result = local_energy(hamiltonian, model, basis.states, space=hilbert)
    expected = hamiltonian.apply(basis, amplitudes) / amplitudes
    torch.testing.assert_close(result.values, expected, atol=1.0e-10, rtol=1.0e-10)
    assert result.active_connections > 0
    assert result.unique_connected_states == basis.states.shape[0]
