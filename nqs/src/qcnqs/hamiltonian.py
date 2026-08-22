"""Tensorized Slater--Condon operator for non-relativistic QC Hamiltonians.

The public operator exposes fixed-width connection rows through the neutral
``ConnectedOperator`` contract.  Internally, all determinants in a batch share
the same excitation templates, which turns the diagonal, single, and double
Slater--Condon rules into PyTorch gather/scatter operations on CPU or CUDA.

No persistent local-energy cache, stochastic connection truncation, or manual
GPU block scheduler is used.  The only sparsity is the exact Slater--Condon
selection rule.
"""

from __future__ import annotations

import torch

from qcnqs.chemistry.problem import MolecularProblem
from qcnqs.determinant import (
    double_fermion_replacement_sign,
    make_transition_templates,
    single_fermion_hop_sign,
)
from qcnqs.hilbert import SectorBasis
from qcnqs.interfaces import Connections
from qcnqs.operator_tools import apply_in_basis, dense_operator_matrix


def spin_orbital_integrals(
    h1e: torch.Tensor,
    eri: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Expand spatial integrals into ``h[p,q]`` and ``<pq||rs>``.

    Spatial ERIs arrive in PySCF chemist notation ``(pq|rs)``.  Spin orbitals
    are interleaved ``alpha, beta, alpha, beta, ...``.  The returned rank-four
    tensor is antisymmetrized and therefore enters the Hamiltonian as
    ``1/4 sum[pqrs] <pq||rs> a_p^dag a_q^dag a_s a_r``.

    Advanced indexing creates the entire tensor with PyTorch operations on the
    target device.  The dense representation is intentionally simple and is
    appropriate for the small/medium orbital spaces targeted by this readable
    implementation; replacing it does not affect the operator interface.
    """

    n_orbitals = h1e.shape[0]
    n_spin_orbitals = 2 * n_orbitals
    index = torch.arange(n_spin_orbitals, device=h1e.device)
    spatial = torch.div(index, 2, rounding_mode="floor")
    spin = torch.remainder(index, 2)
    h1_spin = h1e[spatial[:, None], spatial[None, :]]
    h1_spin = h1_spin * (spin[:, None] == spin[None, :])

    p = index[:, None, None, None]
    q = index[None, :, None, None]
    r = index[None, None, :, None]
    s = index[None, None, None, :]
    ps, qs, rs, ss = spatial[p], spatial[q], spatial[r], spatial[s]
    direct = eri[ps, rs, qs, ss]
    exchange = eri[ps, ss, qs, rs]
    direct_spin = (spin[p] == spin[r]) & (spin[q] == spin[s])
    exchange_spin = (spin[p] == spin[s]) & (spin[q] == spin[r])
    antisymmetrized = direct * direct_spin - exchange * exchange_spin
    return h1_spin.contiguous(), antisymmetrized.contiguous()


class QCHamiltonian:
    """Exact electronic Hamiltonian in a fixed ``(N_alpha, N_beta)`` sector.

    Connection slot zero is always the diagonal determinant.  Remaining slots
    contain every spin-conserving single and double excitation, including zero
    matrix elements.  A fixed slot count avoids ragged host data structures and
    lets downstream local-energy code filter/deduplicate entirely in PyTorch.
    """

    def __init__(
        self,
        problem: MolecularProblem,
        *,
        device: torch.device | str = "cpu",
        drop_tolerance: float = 0.0,
    ) -> None:
        if drop_tolerance < 0:
            raise ValueError("drop_tolerance must be non-negative")
        self.problem = problem
        requested_device = torch.device(device)
        self.drop_tolerance = float(drop_tolerance)
        h1e = problem.h1e.to(requested_device)
        eri = problem.eri.to(requested_device)
        self.device = h1e.device
        self.h1e, self.g2e = spin_orbital_integrals(h1e, eri)
        # Reuse immutable device indices in every connection evaluation.  They
        # are execution metadata, not a cache of model- or sample-dependent data.
        self._spatial_indices = torch.arange(problem.n_orbitals, device=self.device)
        self._spin_indices = torch.arange(problem.n_spin_orbitals, device=self.device)
        self.templates = make_transition_templates(
            problem.n_orbitals,
            problem.n_alpha,
            problem.n_beta,
        ).to(self.device)

    @property
    def n_spin_orbitals(self) -> int:
        return self.problem.n_spin_orbitals

    @property
    def n_electrons(self) -> int:
        return self.problem.n_electrons

    @property
    def n_connections(self) -> int:
        return 1 + self.templates.singles.shape[0] + self.templates.doubles.shape[0]

    def to(self, device: torch.device | str) -> QCHamiltonian:
        """Rebuild immutable integral/template tensors on another device."""

        return QCHamiltonian(
            self.problem,
            device=device,
            drop_tolerance=self.drop_tolerance,
        )

    def _prepare_states(self, states: torch.Tensor) -> torch.Tensor:
        """Normalize shape/dtype using metadata checks that never sync CUDA."""

        if states.ndim == 1:
            states = states.unsqueeze(0)
        if states.ndim != 2 or states.shape[1] != self.n_spin_orbitals:
            raise ValueError("states must have shape [batch, n_spin_orbitals]")
        if states.device != self.device:
            raise ValueError(f"states are on {states.device}, Hamiltonian is on {self.device}")
        return states.bool()

    def _validate_states(self, states: torch.Tensor) -> torch.Tensor:
        """Validate the conserved sector at a public API boundary.

        The two reductions below intentionally synchronize a CUDA device so a
        malformed external input raises immediately.  Trusted internal paths
        call ``connections(validate=False)`` and avoid these synchronizations.
        """

        states = self._prepare_states(states)
        if not bool(torch.all(states[:, 0::2].sum(dim=1) == self.problem.n_alpha)):
            raise ValueError("state batch leaves the fixed alpha-electron sector")
        if not bool(torch.all(states[:, 1::2].sum(dim=1) == self.problem.n_beta)):
            raise ValueError("state batch leaves the fixed beta-electron sector")
        return states

    def _occupied_virtual_tables(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return sorted spatial-orbital slots with fixed padded widths."""

        n_orbitals = self.problem.n_orbitals
        orbital = self._spatial_indices
        spin_occupation = torch.stack((states[:, 0::2], states[:, 1::2]), dim=1)
        occupied = torch.where(spin_occupation, orbital, n_orbitals).sort(dim=-1).values
        virtual = torch.where(~spin_occupation, orbital, n_orbitals).sort(dim=-1).values
        return occupied, virtual

    @staticmethod
    def _lookup_slots(table: torch.Tensor, spin: torch.Tensor, slot: torch.Tensor) -> torch.Tensor:
        return table[:, spin, slot]

    @staticmethod
    def _toggle_states(
        states: torch.Tensor,
        holes: tuple[torch.Tensor, ...],
        particles: tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        """Broadcast each state over excitation slots and toggle occupations."""

        count = holes[0].shape[1]
        connected = states[:, None, :].expand(-1, count, -1).clone()
        for index in holes:
            connected.scatter_(2, index.unsqueeze(-1), False)
        for index in particles:
            connected.scatter_(2, index.unsqueeze(-1), True)
        return connected

    def _diagonal_unchecked(self, states: torch.Tensor) -> torch.Tensor:
        """Evaluate diagonal Slater--Condon rows for prepared fixed-spin states."""

        orbital = self._spin_indices
        occupied = torch.where(states, orbital, self.n_spin_orbitals).sort(dim=-1).values
        occupied = occupied[:, : self.n_electrons]
        one_body = self.h1e[occupied, occupied].sum(dim=1)
        left = occupied[:, :, None]
        right = occupied[:, None, :]
        two_body = 0.5 * self.g2e[left, right, left, right].sum(dim=(1, 2))
        return one_body + two_body + self.problem.core_energy

    def diagonal(self, states: torch.Tensor, *, validate: bool = True) -> torch.Tensor:
        """Return diagonal matrix elements for a determinant batch."""

        prepared = self._validate_states(states) if validate else self._prepare_states(states)
        return self._diagonal_unchecked(prepared)

    def connections(
        self,
        states: torch.Tensor,
        *,
        validate: bool = True,
    ) -> Connections:
        """Evaluate all exact Slater--Condon connections for ``states``.

        Set ``validate=False`` only when the caller already obtained states
        from the matching fixed-sector sampler/basis.  This hot-path option
        removes device-to-host validation synchronizations; shape, device, and
        boolean dtype normalization are still checked.
        """

        states = self._validate_states(states) if validate else self._prepare_states(states)
        occupied, virtual = self._occupied_virtual_tables(states)
        batch = states.shape[0]

        single_template = self.templates.singles
        if single_template.numel():
            hole_spin, hole_slot, particle_spin, particle_slot = single_template.T
            hole_spatial = self._lookup_slots(occupied, hole_spin, hole_slot)
            particle_spatial = self._lookup_slots(virtual, particle_spin, particle_slot)
            holes = 2 * hole_spatial + hole_spin
            particles = 2 * particle_spatial + particle_spin
            single_states = self._toggle_states(states, (holes,), (particles,))
            sign = single_fermion_hop_sign(states, holes, particles)
            all_occupied = (
                torch.where(
                    states,
                    self._spin_indices,
                    self.n_spin_orbitals,
                )
                .sort(dim=-1)
                .values[:, : self.n_electrons]
            )
            # Slater--Condon singles contain the one-electron term plus a
            # contraction of <p j || h j> over every occupied spin orbital j.
            interaction = self.g2e[
                particles[:, :, None],
                all_occupied[:, None, :],
                holes[:, :, None],
                all_occupied[:, None, :],
            ].sum(dim=-1)
            single_values = sign * (self.h1e[particles, holes] + interaction)
        else:
            single_states = states.new_empty((batch, 0, self.n_spin_orbitals))
            single_values = self.h1e.new_empty((batch, 0))

        double_template = self.templates.doubles
        if double_template.numel():
            hs0, hi0, hs1, hi1, ps0, pi0, ps1, pi1 = double_template.T
            hole_a = 2 * self._lookup_slots(occupied, hs0, hi0) + hs0
            hole_b = 2 * self._lookup_slots(occupied, hs1, hi1) + hs1
            particle_a = 2 * self._lookup_slots(virtual, ps0, pi0) + ps0
            particle_b = 2 * self._lookup_slots(virtual, ps1, pi1) + ps1
            double_states = self._toggle_states(
                states,
                (hole_a, hole_b),
                (particle_a, particle_b),
            )
            sign, h_high, h_low, p_high, p_low = double_fermion_replacement_sign(
                states,
                hole_a,
                hole_b,
                particle_a,
                particle_b,
            )
            # Antisymmetrization has already combined direct and exchange
            # contributions, leaving one rank-four lookup per double slot.
            double_values = sign * self.g2e[p_high, p_low, h_high, h_low]
        else:
            double_states = states.new_empty((batch, 0, self.n_spin_orbitals))
            double_values = self.h1e.new_empty((batch, 0))

        values = torch.cat(
            (self._diagonal_unchecked(states)[:, None], single_values, double_values),
            dim=1,
        )
        if self.drop_tolerance:
            values = torch.where(values.abs() > self.drop_tolerance, values, 0.0)
        connected = torch.cat((states[:, None, :], single_states, double_states), dim=1)
        return Connections(connected, values)

    def apply(
        self,
        basis: SectorBasis,
        vector: torch.Tensor,
        *,
        missing_tolerance: float = 1.0e-10,
    ) -> torch.Tensor:
        """Apply the operator to a vector represented in ``basis`` order.

        Connected-state keys are located with ``searchsorted`` and accumulated
        without constructing a sparse or dense matrix.  Missing non-negligible
        connections indicate that the supplied basis is not a closed sector.
        """

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
        """Materialize a small closed-sector matrix for exact validation only."""

        return dense_operator_matrix(
            self,
            basis,
            max_states=max_states,
            missing_tolerance=missing_tolerance,
        )
