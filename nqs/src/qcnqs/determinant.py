"""Determinant transition primitives for a fixed-spin Fock sector.

The name deliberately refers to transitions *between occupation states*, not
to physical excited-state calculations.  Both Slater--Condon operators and
Metropolis proposals need state-independent occupied/virtual slot layouts and
fermionic parity factors, so those neutral primitives live here.

No Hamiltonian coefficients are stored in this module.  A consumer supplies
the one-/two-body matrix element or proposal probability associated with each
transition.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import torch


@dataclass(frozen=True)
class DeterminantTransitionTemplates:
    """Packed spin/slot layouts for all fixed-sector single and double moves.

    ``singles`` columns are ``(hole_spin, hole_slot, particle_spin,
    particle_slot)``.  ``doubles`` contains four corresponding spin/slot pairs.
    Slots index sorted occupied and virtual tables, so the same tensors apply
    to every determinant with the configured alpha/beta particle counts.
    """

    singles: torch.Tensor
    doubles: torch.Tensor

    def to(self, device: torch.device | str) -> DeterminantTransitionTemplates:
        """Copy both immutable layout tensors to an execution device."""

        return DeterminantTransitionTemplates(self.singles.to(device), self.doubles.to(device))


def make_transition_templates(
    n_orbitals: int,
    n_alpha: int,
    n_beta: int,
) -> DeterminantTransitionTemplates:
    """Enumerate every spin-conserving single and double replacement slot."""

    occupied = (n_alpha, n_beta)
    virtual = (n_orbitals - n_alpha, n_orbitals - n_beta)
    singles: list[tuple[int, ...]] = []
    for spin in range(2):
        singles.extend(
            (spin, hole, spin, particle)
            for hole in range(occupied[spin])
            for particle in range(virtual[spin])
        )

    doubles: list[tuple[int, ...]] = []
    # Same-spin doubles use unordered hole and particle pairs.  Opposite-spin
    # doubles use one alpha and one beta replacement, so their Cartesian
    # product contains no equivalent pair permutations.
    for spin in range(2):
        doubles.extend(
            (spin, h0, spin, h1, spin, p0, spin, p1)
            for h0, h1 in combinations(range(occupied[spin]), 2)
            for p0, p1 in combinations(range(virtual[spin]), 2)
        )
    doubles.extend(
        (0, h_alpha, 1, h_beta, 0, p_alpha, 1, p_beta)
        for h_alpha in range(occupied[0])
        for p_alpha in range(virtual[0])
        for h_beta in range(occupied[1])
        for p_beta in range(virtual[1])
    )
    return DeterminantTransitionTemplates(
        singles=torch.tensor(singles, dtype=torch.int64).reshape(-1, 4),
        doubles=torch.tensor(doubles, dtype=torch.int64).reshape(-1, 8),
    )


def single_fermion_hop_sign(
    states: torch.Tensor,
    holes: torch.Tensor,
    particles: torch.Tensor,
) -> torch.Tensor:
    """Return the parity of ``a_particle^dagger a_hole`` for batched moves.

    ``holes`` and ``particles`` have shape ``[batch, moves]``.  Prefix sums
    count occupied spin orbitals below each operator index.  If the removed
    orbital lies below the created orbital, its removal changes the creation
    prefix by one, which gives the final correction term.
    """

    integer_states = states.to(torch.int64)
    prefix = integer_states.cumsum(dim=-1) - integer_states
    exponent = prefix.gather(1, holes) + prefix.gather(1, particles)
    exponent = exponent - (holes < particles).to(torch.int64)
    return torch.where(exponent.remainder(2) == 0, 1.0, -1.0)


def double_fermion_replacement_sign(
    states: torch.Tensor,
    hole_a: torch.Tensor,
    hole_b: torch.Tensor,
    particle_a: torch.Tensor,
    particle_b: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return parity and canonical indices for a two-particle replacement.

    Holes and particles are sorted so same- and opposite-spin transitions use
    one operator ordering.  Prefix counts are corrected after each conceptual
    annihilation/creation instead of mutating one state per transition slot.
    """

    hole_high = torch.maximum(hole_a, hole_b)
    hole_low = torch.minimum(hole_a, hole_b)
    particle_high = torch.maximum(particle_a, particle_b)
    particle_low = torch.minimum(particle_a, particle_b)
    integer_states = states.to(torch.int64)
    prefix = integer_states.cumsum(dim=-1) - integer_states
    hole_parity = prefix.gather(1, hole_high) + prefix.gather(1, hole_low)
    low_prefix = prefix.gather(1, particle_low)
    low_prefix = low_prefix - (hole_low < particle_low).to(torch.int64)
    low_prefix = low_prefix - (hole_high < particle_low).to(torch.int64)
    high_prefix = prefix.gather(1, particle_high)
    high_prefix = high_prefix - (hole_low < particle_high).to(torch.int64)
    high_prefix = high_prefix - (hole_high < particle_high).to(torch.int64)
    high_prefix = high_prefix + (particle_low < particle_high).to(torch.int64)
    exponent = hole_parity + low_prefix + high_prefix
    sign = torch.where(exponent.remainder(2) == 0, 1.0, -1.0)
    return sign, hole_high, hole_low, particle_high, particle_low
