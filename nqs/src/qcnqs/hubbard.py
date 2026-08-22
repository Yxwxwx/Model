"""Tensorized one- and two-dimensional spinful Hubbard Hamiltonians.

The implemented convention is

``H = -t sum[<ij>,sigma] (c_i,sigma^dag c_j,sigma + h.c.)``
``    + U sum[i] n_i,up n_i,down``.

Each undirected lattice bond appears once; both hopping orientations and both
spins are generated internally.  Connections have a fixed width of
``1 + 4 * n_bonds`` (one diagonal slot plus four directed spin hops per bond),
which makes the operator compatible with the same local-energy, sampler, VMC,
and exact-diagnostic code used by the quantum-chemistry Hamiltonian.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from qcnqs.determinant import single_fermion_hop_sign
from qcnqs.hilbert import FermionHilbert, SectorBasis
from qcnqs.interfaces import Connections
from qcnqs.operator_tools import apply_in_basis, dense_operator_matrix

Bond = tuple[int, int]


def _add_simple_bond(bonds: set[Bond], left: int, right: int) -> None:
    if left != right:
        bonds.add((min(left, right), max(left, right)))


def nearest_neighbor_bonds_1d(n_sites: int, *, periodic: bool = False) -> tuple[Bond, ...]:
    """Return unique undirected nearest-neighbor bonds for a one-dimensional chain."""

    if n_sites <= 0:
        raise ValueError("n_sites must be positive")
    bonds: set[Bond] = set()
    for site in range(n_sites - 1):
        _add_simple_bond(bonds, site, site + 1)
    if periodic:
        _add_simple_bond(bonds, n_sites - 1, 0)
    return tuple(sorted(bonds))


def nearest_neighbor_bonds_2d(
    rows: int,
    columns: int,
    *,
    periodic: bool = False,
) -> tuple[Bond, ...]:
    """Return row-major nearest-neighbor bonds for a rectangular square lattice.

    Periodic boundaries wrap independently along both axes.  The result is a
    simple graph: small dimensions such as length two do not create duplicate
    parallel bonds when forward and wrapped neighbors coincide.
    """

    if rows <= 0 or columns <= 0:
        raise ValueError("rows and columns must be positive")

    def site(row: int, column: int) -> int:
        return row * columns + column

    bonds: set[Bond] = set()
    for row in range(rows):
        for column in range(columns):
            current = site(row, column)
            if column + 1 < columns:
                _add_simple_bond(bonds, current, site(row, column + 1))
            elif periodic:
                _add_simple_bond(bonds, current, site(row, 0))
            if row + 1 < rows:
                _add_simple_bond(bonds, current, site(row + 1, column))
            elif periodic:
                _add_simple_bond(bonds, current, site(0, column))
    return tuple(sorted(bonds))


class HubbardHamiltonian:
    """Spinful Hubbard operator on an arbitrary simple undirected lattice.

    Args:
        space: Fixed ``(N_alpha, N_beta)`` fermion sector.  Spatial orbitals are
            interpreted as lattice sites.
        bonds: Unique undirected pairs of lattice-site indices.
        hopping: Real nearest-neighbor hopping amplitude ``t``.
        interaction: Real on-site interaction ``U``.
        device: PyTorch execution device.
        dtype: Floating-point dtype for matrix elements.
    """

    def __init__(
        self,
        space: FermionHilbert,
        bonds: Sequence[Bond],
        *,
        hopping: float = 1.0,
        interaction: float = 4.0,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float64,
    ) -> None:
        probe = torch.empty((), dtype=dtype)
        if not probe.is_floating_point():
            raise ValueError("Hubbard matrix elements require a floating-point dtype")
        normalized = []
        seen = set()
        for raw_left, raw_right in bonds:
            left, right = int(raw_left), int(raw_right)
            if left == right:
                raise ValueError("self bonds are not allowed")
            if not 0 <= left < space.n_orbitals or not 0 <= right < space.n_orbitals:
                raise ValueError("bond endpoint lies outside the lattice")
            bond = (min(left, right), max(left, right))
            if bond in seen:
                raise ValueError(f"duplicate undirected bond: {bond}")
            seen.add(bond)
            normalized.append(bond)

        self.space = space
        self.bonds = tuple(sorted(normalized))
        self.hopping = float(hopping)
        self.interaction = float(interaction)
        self.dtype = dtype
        self.device = torch.empty(0, device=torch.device(device)).device
        self._sector_filter = space.device_filter(self.device)

        sources = []
        targets = []
        for left, right in self.bonds:
            for spin in range(2):
                left_spin_orbital = 2 * left + spin
                right_spin_orbital = 2 * right + spin
                sources.extend((left_spin_orbital, right_spin_orbital))
                targets.extend((right_spin_orbital, left_spin_orbital))
        self._sources = torch.tensor(sources, dtype=torch.int64, device=self.device)
        self._targets = torch.tensor(targets, dtype=torch.int64, device=self.device)

    @property
    def n_sites(self) -> int:
        return self.space.n_orbitals

    @property
    def n_spin_orbitals(self) -> int:
        return self.space.n_spin_orbitals

    @property
    def n_electrons(self) -> int:
        return self.space.n_electrons

    @property
    def n_connections(self) -> int:
        return 1 + self._sources.numel()

    def to(self, device: torch.device | str) -> HubbardHamiltonian:
        """Rebuild immutable lattice tensors on another execution device."""

        return HubbardHamiltonian(
            self.space,
            self.bonds,
            hopping=self.hopping,
            interaction=self.interaction,
            device=device,
            dtype=self.dtype,
        )

    def _prepare_states(self, states: torch.Tensor) -> torch.Tensor:
        if states.ndim == 1:
            states = states.unsqueeze(0)
        if states.ndim != 2 or states.shape[1] != self.n_spin_orbitals:
            raise ValueError("states must have shape [batch, n_spin_orbitals]")
        if states.device != self.device:
            raise ValueError(f"states are on {states.device}, operator is on {self.device}")
        return states.bool()

    def _validate_states(self, states: torch.Tensor) -> torch.Tensor:
        states = self._prepare_states(states)
        if not bool(torch.all(self._sector_filter.contains(states))):
            raise ValueError("state batch leaves the configured Hubbard sector")
        return states

    def _diagonal_unchecked(self, states: torch.Tensor) -> torch.Tensor:
        double_occupancy = states[:, 0::2] & states[:, 1::2]
        return self.interaction * double_occupancy.sum(dim=-1).to(self.dtype)

    def diagonal(self, states: torch.Tensor, *, validate: bool = True) -> torch.Tensor:
        """Return ``U`` times the number of doubly occupied lattice sites."""

        prepared = self._validate_states(states) if validate else self._prepare_states(states)
        return self._diagonal_unchecked(prepared)

    def connections(
        self,
        states: torch.Tensor,
        *,
        validate: bool = True,
    ) -> Connections:
        """Return the diagonal row and every directed nearest-neighbor hop."""

        states = self._validate_states(states) if validate else self._prepare_states(states)
        batch = states.shape[0]
        move_count = self._sources.numel()
        if not move_count:
            return Connections(states[:, None, :], self._diagonal_unchecked(states)[:, None])

        holes = self._sources[None, :].expand(batch, -1)
        particles = self._targets[None, :].expand(batch, -1)
        active = states[:, self._sources] & ~states[:, self._targets]
        proposed = states[:, None, :].expand(-1, move_count, -1).clone()
        proposed.scatter_(2, holes.unsqueeze(-1), False)
        proposed.scatter_(2, particles.unsqueeze(-1), True)
        # Zero-valued slots point back to the input determinant.  Downstream
        # code filters by matrix element, and every returned state remains a
        # valid member of the fixed particle sector.
        connected = torch.where(active[..., None], proposed, states[:, None, :])
        signs = single_fermion_hop_sign(states, holes, particles).to(self.dtype)
        hopping_values = -self.hopping * signs * active.to(self.dtype)
        values = torch.cat((self._diagonal_unchecked(states)[:, None], hopping_values), dim=1)
        return Connections(torch.cat((states[:, None, :], connected), dim=1), values)

    def apply(
        self,
        basis: SectorBasis,
        vector: torch.Tensor,
        *,
        missing_tolerance: float = 1.0e-10,
    ) -> torch.Tensor:
        """Apply the Hubbard operator to a finite-sector vector."""

        return apply_in_basis(
            self,
            basis,
            vector,
            missing_tolerance=missing_tolerance,
        )

    def dense_matrix(
        self,
        basis: SectorBasis,
        *,
        max_states: int = 4096,
        missing_tolerance: float = 1.0e-10,
    ) -> torch.Tensor:
        """Materialize a small Hubbard matrix for exact tests or references."""

        return dense_operator_matrix(
            self,
            basis,
            max_states=max_states,
            missing_tolerance=missing_tolerance,
        )


class Hubbard1D(HubbardHamiltonian):
    """Nearest-neighbor Hubbard Hamiltonian on a one-dimensional chain."""

    def __init__(
        self,
        n_sites: int,
        n_alpha: int,
        n_beta: int,
        *,
        hopping: float = 1.0,
        interaction: float = 4.0,
        periodic: bool = False,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float64,
    ) -> None:
        self.length = int(n_sites)
        self.periodic = bool(periodic)
        super().__init__(
            FermionHilbert(self.length, n_alpha, n_beta),
            nearest_neighbor_bonds_1d(self.length, periodic=self.periodic),
            hopping=hopping,
            interaction=interaction,
            device=device,
            dtype=dtype,
        )


class Hubbard2D(HubbardHamiltonian):
    """Nearest-neighbor Hubbard Hamiltonian on a rectangular square lattice."""

    def __init__(
        self,
        rows: int,
        columns: int,
        n_alpha: int,
        n_beta: int,
        *,
        hopping: float = 1.0,
        interaction: float = 4.0,
        periodic: bool = False,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float64,
    ) -> None:
        self.rows = int(rows)
        self.columns = int(columns)
        self.periodic = bool(periodic)
        super().__init__(
            FermionHilbert(self.rows * self.columns, n_alpha, n_beta),
            nearest_neighbor_bonds_2d(self.rows, self.columns, periodic=self.periodic),
            hopping=hopping,
            interaction=interaction,
            device=device,
            dtype=dtype,
        )
