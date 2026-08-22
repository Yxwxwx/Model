from __future__ import annotations

import math

import torch
from torch import nn

from qcnqs.assembly import hilbert_from_problem
from qcnqs.chemistry.problem import MolecularProblem
from qcnqs.hamiltonian import QCHamiltonian
from qcnqs.interfaces import Connections, SampleBatch, SamplerStats
from qcnqs.models import DeterminantBackflow
from qcnqs.sampler import FermionSampler
from qcnqs.vmc import VMCDriver, exact_model_expectation
from qcnqs.wavefunction import LogAmplitude


def _small_problem() -> MolecularProblem:
    h1e = torch.tensor([[-1.1, -0.2], [-0.2, -0.4]], dtype=torch.float64)
    eri = torch.zeros((2, 2, 2, 2), dtype=torch.float64)
    eri[0, 0, 0, 0] = 0.7
    eri[1, 1, 1, 1] = 0.5
    eri[0, 0, 1, 1] = eri[1, 1, 0, 0] = 0.3
    eri[0, 1, 0, 1] = eri[1, 0, 1, 0] = 0.12
    eri[1, 0, 0, 1] = eri[0, 1, 1, 0] = 0.12
    return MolecularProblem(h1e, eri, 0.2, 1, 1, (0, 0), name="two-orbital")


def test_exact_vmc_estimator_and_update() -> None:
    torch.manual_seed(41)
    problem = _small_problem()
    hilbert = hilbert_from_problem(problem, point_group=False)
    hamiltonian = QCHamiltonian(problem)
    model = DeterminantBackflow(4, 2, 4)
    sampler = FermionSampler(
        hilbert,
        model,
        n_samples=16,
        n_chains=4,
        seed=43,
        device="cpu",
        mode="exact",
    )
    driver = VMCDriver(
        hamiltonian,
        hilbert,
        model,
        sampler,
        learning_rate=0.02,
        diagonal_shift=0.1,
    )
    basis = hilbert.enumerate()
    explicit = exact_model_expectation(hamiltonian, model, basis)
    estimate = driver.estimate()
    assert estimate.standard_error == 0.0
    assert math.isclose(estimate.mean, explicit, abs_tol=1.0e-11)
    before = torch.cat([parameter.detach().reshape(-1) for parameter in model.parameters()])
    step = driver.step()
    after = torch.cat([parameter.detach().reshape(-1) for parameter in model.parameters()])
    assert not torch.equal(before, after)
    assert math.isfinite(step.energy)
    assert math.isfinite(step.update_norm)
    assert step.qgt_method == "minsr"
    assert driver.iteration == 1


class _TwoStateSpace:
    n_spin_orbitals = 2
    requires_projection = False

    def contains(self, states: torch.Tensor) -> torch.Tensor:
        return states.sum(dim=-1) == 1


class _DiagonalOperator:
    device = torch.device("cpu")
    n_spin_orbitals = 2

    def connections(self, states: torch.Tensor, *, validate: bool = True) -> Connections:
        if validate and not bool(torch.all(states.sum(dim=-1) == 1)):
            raise ValueError("state is outside the test space")
        energies = torch.where(states[:, 0], -1.0, 0.5).to(torch.float64)
        return Connections(states[:, None, :], energies[:, None])


class _LinearLogWaveFunction(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([0.1, -0.2], dtype=torch.float64))

    def log_amplitude(self, states: torch.Tensor) -> LogAmplitude:
        logabs = states.to(self.weight) @ self.weight
        return LogAmplitude(logabs, torch.ones_like(logabs))

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.log_amplitude(states).logabs


class _FixedSampler:
    def __init__(self, space: _TwoStateSpace, model: _LinearLogWaveFunction) -> None:
        self.space = space
        self.device = torch.device("cpu")
        self.model = model

    def sample(self, *, discard: bool = False) -> SampleBatch:
        del discard
        states = torch.tensor([[True, False], [False, True]])
        amplitudes = self.model.log_amplitude(states)
        return SampleBatch(
            states=states,
            counts=torch.ones(2, dtype=torch.int64),
            probabilities=torch.full((2,), 0.5, dtype=torch.float64),
            inverse=torch.arange(2),
            raw_states=states,
            amplitudes=amplitudes,
            stats=SamplerStats(0, 0, 0),
            exact_probabilities=True,
        )

    def refresh(self) -> None:
        return None


def test_vmc_driver_accepts_structural_components() -> None:
    """The driver must not require QC Hamiltonian or fermion-sampler classes."""

    space = _TwoStateSpace()
    model = _LinearLogWaveFunction()
    sampler = _FixedSampler(space, model)
    driver = VMCDriver(
        _DiagonalOperator(),
        space,
        model,
        sampler,
        learning_rate=0.01,
        diagonal_shift=0.1,
    )
    estimate = driver.estimate()
    assert math.isclose(estimate.mean, -0.25, abs_tol=1.0e-12)
    step = driver.step()
    assert math.isfinite(step.update_norm)
