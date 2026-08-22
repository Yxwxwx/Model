"""Exact finite-basis utilities for any fixed-width connected operator.

These helpers depend only on the structural ``ConnectedOperator`` contract and
the compact ``SectorBasis`` index.  Quantum-chemistry and Hubbard operators can
therefore share exact application/validation code without inheriting from a
common implementation class.
"""

from __future__ import annotations

import torch

from qcnqs.hilbert import SectorBasis
from qcnqs.interfaces import ConnectedOperator


def apply_in_basis(
    operator: ConnectedOperator,
    basis: SectorBasis,
    vector: torch.Tensor,
    *,
    missing_tolerance: float = 1.0e-10,
) -> torch.Tensor:
    """Apply ``operator`` to a vector represented in sorted ``basis`` order."""

    if basis.states.device != operator.device or vector.device != operator.device:
        raise ValueError("basis, vector, and operator must share a device")
    if vector.shape != (basis.states.shape[0],):
        raise ValueError("vector has the wrong sector dimension")
    connections = operator.connections(basis.states, validate=False)
    columns, valid = basis.index(connections.states)
    missing = ~valid & (connections.matrix_elements.abs() > missing_tolerance)
    if bool(torch.any(missing)):
        largest = connections.matrix_elements[missing].abs().max().item()
        raise ValueError(f"basis omits a connected state with |H|={largest:.3e}")
    coefficients = torch.where(valid, connections.matrix_elements, 0.0)
    return (coefficients * vector[columns]).sum(dim=1)


def dense_operator_matrix(
    operator: ConnectedOperator,
    basis: SectorBasis,
    *,
    max_states: int = 4096,
    missing_tolerance: float = 1.0e-10,
) -> torch.Tensor:
    """Materialize a small closed-sector matrix for tests/exact diagnostics."""

    size = basis.states.shape[0]
    if size > max_states:
        raise ValueError(f"refusing to build a dense {size} x {size} operator")
    if basis.states.device != operator.device:
        raise ValueError("basis and operator must share a device")
    connections = operator.connections(basis.states, validate=False)
    columns, valid = basis.index(connections.states)
    missing = ~valid & (connections.matrix_elements.abs() > missing_tolerance)
    if bool(torch.any(missing)):
        largest = connections.matrix_elements[missing].abs().max().item()
        raise ValueError(f"basis omits a connected state with |H|={largest:.3e}")
    rows = torch.arange(size, device=operator.device)[:, None].expand_as(columns)
    matrix = torch.zeros(
        (size, size),
        dtype=connections.matrix_elements.dtype,
        device=operator.device,
    )
    matrix.index_put_(
        (rows[valid], columns[valid]),
        connections.matrix_elements[valid],
        accumulate=True,
    )
    return matrix
