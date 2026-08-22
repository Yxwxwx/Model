"""Composition helpers that connect chemistry input to execution components.

Factories belong at the composition boundary because they necessarily know
about more than one otherwise independent layer.  In particular, neither the
Hilbert-space module nor the Hamiltonian module should import the other merely
to construct a calculation from :class:`MolecularProblem` metadata.
"""

from __future__ import annotations

from qcnqs.chemistry.problem import MolecularProblem
from qcnqs.hilbert import FermionHilbert


def hilbert_from_problem(problem: MolecularProblem, *, point_group: bool) -> FermionHilbert:
    """Build the fixed-spin Hilbert sector requested for a molecular problem."""

    return FermionHilbert(
        n_orbitals=problem.n_orbitals,
        n_alpha=problem.n_alpha,
        n_beta=problem.n_beta,
        orbital_symmetries=problem.orbital_symmetries if point_group else None,
        target_symmetry=problem.target_symmetry if point_group else None,
    )
