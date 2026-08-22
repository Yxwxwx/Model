from __future__ import annotations

from itertools import product

import numpy as np
import torch
from pyscf import gto, scf

from qcnqs.assembly import hilbert_from_problem
from qcnqs.chemistry.problem import MolecularProblem, fci_reference_energy
from qcnqs.hamiltonian import QCHamiltonian


def _symmetric_eri(n_orbitals: int, seed: int = 5) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    eri = torch.randn(
        (n_orbitals,) * 4,
        generator=generator,
        dtype=torch.float64,
    )
    eri = 0.5 * (eri + eri.transpose(0, 1))
    eri = 0.5 * (eri + eri.transpose(2, 3))
    eri = 0.5 * (eri + eri.permute(2, 3, 0, 1))
    return eri


def _random_problem(n_orbitals: int = 3) -> MolecularProblem:
    generator = torch.Generator().manual_seed(7)
    h1e = torch.randn((n_orbitals, n_orbitals), generator=generator, dtype=torch.float64)
    h1e = 0.5 * (h1e + h1e.T)
    return MolecularProblem(
        h1e=h1e,
        eri=_symmetric_eri(n_orbitals),
        core_energy=0.37,
        n_alpha=1,
        n_beta=1,
        orbital_symmetries=(0,) * n_orbitals,
    )


def _operate(state: torch.Tensor, orbital: int, create: bool) -> tuple[torch.Tensor, int] | None:
    occupied = bool(state[orbital])
    if occupied == create:
        return None
    sign = -1 if int(state[:orbital].sum()) % 2 else 1
    result = state.clone()
    result[orbital] = create
    return result, sign


def _explicit_second_quantized_matrix(
    hamiltonian: QCHamiltonian,
    states: torch.Tensor,
) -> torch.Tensor:
    size = states.shape[0]
    state_to_index = {tuple(bool(x) for x in state): i for i, state in enumerate(states)}
    result = torch.eye(size, dtype=torch.float64) * hamiltonian.problem.core_energy
    spin_orbitals = hamiltonian.n_spin_orbitals
    for column, state in enumerate(states):
        for p, q in product(range(spin_orbitals), repeat=2):
            first = _operate(state, q, False)
            if first is None:
                continue
            intermediate, sign = first
            second = _operate(intermediate, p, True)
            if second is None:
                continue
            target, create_sign = second
            row = state_to_index.get(tuple(bool(x) for x in target))
            if row is not None:
                result[row, column] += sign * create_sign * hamiltonian.h1e[p, q]
        for p, q, r, s in product(range(spin_orbitals), repeat=4):
            coefficient = 0.25 * hamiltonian.g2e[p, q, r, s]
            if not coefficient:
                continue
            working = state
            sign = 1
            for orbital, create in ((r, False), (s, False), (q, True), (p, True)):
                applied = _operate(working, orbital, create)
                if applied is None:
                    break
                working, parity = applied
                sign *= parity
            else:
                row = state_to_index.get(tuple(bool(x) for x in working))
                if row is not None:
                    result[row, column] += sign * coefficient
    return result


def test_connections_match_explicit_second_quantization() -> None:
    problem = _random_problem()
    hilbert = hilbert_from_problem(problem, point_group=False)
    basis = hilbert.enumerate()
    hamiltonian = QCHamiltonian(problem)
    actual = hamiltonian.dense_matrix(basis)
    expected = _explicit_second_quantized_matrix(hamiltonian, basis.states)
    torch.testing.assert_close(actual, expected, atol=1.0e-11, rtol=1.0e-11)
    torch.testing.assert_close(actual, actual.T, atol=1.0e-12, rtol=1.0e-12)


def test_trusted_connection_path_is_identical_to_validated_path() -> None:
    problem = _random_problem()
    basis = hilbert_from_problem(problem, point_group=False).enumerate()
    hamiltonian = QCHamiltonian(problem)
    validated = hamiltonian.connections(basis.states, validate=True)
    trusted = hamiltonian.connections(basis.states, validate=False)
    assert torch.equal(validated.states, trusted.states)
    torch.testing.assert_close(validated.matrix_elements, trusted.matrix_elements)


def test_n2_connection_count() -> None:
    problem = MolecularProblem(
        h1e=torch.zeros((10, 10), dtype=torch.float64),
        eri=torch.zeros((10, 10, 10, 10), dtype=torch.float64),
        core_energy=0.0,
        n_alpha=7,
        n_beta=7,
        orbital_symmetries=(0, 4, 0, 4, 2, 1, 0, 5, 6, 4),
    )
    assert QCHamiltonian(problem).n_connections == 610


def test_h2_ground_energy_matches_pyscf_fci() -> None:
    molecule = gto.M(
        atom="H 0 0 0; H 0 0 1.4",
        unit="Bohr",
        basis="sto-3g",
        symmetry=False,
        verbose=0,
    )
    mean_field = scf.RHF(molecule).run(conv_tol=1.0e-14)
    problem = MolecularProblem.from_pyscf_rhf(mean_field, name="H2")
    hilbert = hilbert_from_problem(problem, point_group=False)
    matrix = QCHamiltonian(problem).dense_matrix(hilbert.enumerate())
    actual = torch.linalg.eigvalsh(matrix)[0].item()
    expected = fci_reference_energy(problem)
    np.testing.assert_allclose(actual, expected, atol=1.0e-10, rtol=0.0)
