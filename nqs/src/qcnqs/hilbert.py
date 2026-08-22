"""Fixed-particle fermionic Hilbert spaces and compact determinant indexing.

The module owns state constraints and finite-sector enumeration only; it knows
nothing about molecular integrals, neural models, sampling, or optimization.
Spin orbitals are interleaved alpha/beta throughout the project.  Determinants
with at most 62 spin orbitals are packed into signed ``int64`` keys so lookup and
batch-local deduplication remain native PyTorch operations on CPU or CUDA.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import comb

import torch


def encode_states(states: torch.Tensor) -> torch.Tensor:
    """Pack the last occupation axis into a signed ``int64`` bit key.

    Bit packing is collision-free up to 62 orbitals and works for arbitrary
    leading batch dimensions.  Keeping keys on the execution device allows
    ``torch.unique`` and ``torch.searchsorted`` to replace Python dictionaries
    in local-energy and exact-basis paths.
    """

    if states.ndim < 1:
        raise ValueError("states must have an orbital dimension")
    n_spin_orbitals = states.shape[-1]
    if n_spin_orbitals > 62:
        raise ValueError("int64 state keys support at most 62 spin orbitals")
    powers = torch.bitwise_left_shift(
        torch.ones(n_spin_orbitals, dtype=torch.int64, device=states.device),
        torch.arange(n_spin_orbitals, dtype=torch.int64, device=states.device),
    )
    return (states.to(torch.int64) * powers).sum(dim=-1)


def decode_states(keys: torch.Tensor, n_spin_orbitals: int) -> torch.Tensor:
    """Unpack integer determinant keys into a boolean occupation tensor."""

    if n_spin_orbitals > 62:
        raise ValueError("int64 state keys support at most 62 spin orbitals")
    shifts = torch.arange(n_spin_orbitals, dtype=torch.int64, device=keys.device)
    return torch.bitwise_and(torch.bitwise_right_shift(keys[..., None], shifts), 1).bool()


@dataclass(frozen=True)
class SectorBasis:
    """Sorted finite basis with vectorized state-to-row lookup."""

    states: torch.Tensor
    keys: torch.Tensor

    def __post_init__(self) -> None:
        if self.states.ndim != 2 or self.states.dtype != torch.bool:
            raise ValueError("basis states must be a bool matrix")
        if self.keys.shape != (self.states.shape[0],):
            raise ValueError("basis keys must have one entry per state")
        if self.keys.numel() > 1 and not bool(torch.all(self.keys[1:] > self.keys[:-1])):
            raise ValueError("basis keys must be unique and sorted")

    def to(self, device: torch.device | str) -> SectorBasis:
        return SectorBasis(self.states.to(device), self.keys.to(device))

    def index(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Map states to basis rows and separately flag missing determinants."""

        keys = encode_states(states)
        indices = torch.searchsorted(self.keys, keys)
        safe = indices.clamp_max(max(self.keys.numel() - 1, 0))
        valid = indices < self.keys.numel()
        if self.keys.numel():
            valid = valid & (self.keys[safe] == keys)
        return safe, valid


@dataclass(frozen=True)
class FermionSectorFilter:
    """Device-resident validator compiled from immutable sector metadata.

    Abelian point-group irreps are represented as XOR labels.  XOR reduction
    is vectorized by expanding labels into bits and computing the parity of
    each bit across occupied spatial orbitals.  This replaces one Python/GPU
    launch per orbital with a small, fixed collection of tensor operations.
    """

    n_orbitals: int
    n_alpha: int
    n_beta: int
    target_symmetry: int | None
    symmetry_bits: torch.Tensor
    bit_values: torch.Tensor

    @property
    def n_spin_orbitals(self) -> int:
        return 2 * self.n_orbitals

    def _check_shape(self, states: torch.Tensor) -> None:
        if states.shape[-1] != self.n_spin_orbitals:
            raise ValueError("state has the wrong number of spin orbitals")

    def state_symmetry(self, states: torch.Tensor) -> torch.Tensor:
        """Return XOR irrep labels for an arbitrary state batch."""

        self._check_shape(states)
        if not self.symmetry_bits.numel():
            return torch.zeros(states.shape[:-1], dtype=torch.int64, device=states.device)
        occupied_parity = torch.logical_xor(states[..., 0::2], states[..., 1::2])
        contributions = occupied_parity.to(torch.int64)[..., :, None] * self.symmetry_bits
        bit_parity = contributions.sum(dim=-2).remainder(2)
        return (bit_parity * self.bit_values).sum(dim=-1)

    def symmetry_mask(self, states: torch.Tensor) -> torch.Tensor:
        """Check only point-group symmetry for moves known to conserve spin."""

        self._check_shape(states)
        if self.target_symmetry is None:
            return torch.ones(states.shape[:-1], dtype=torch.bool, device=states.device)
        return self.state_symmetry(states) == self.target_symmetry

    def contains(self, states: torch.Tensor) -> torch.Tensor:
        """Check particle counts and, when requested, the target irrep."""

        self._check_shape(states)
        alpha_ok = states[..., 0::2].sum(dim=-1) == self.n_alpha
        beta_ok = states[..., 1::2].sum(dim=-1) == self.n_beta
        return alpha_ok & beta_ok & self.symmetry_mask(states)


@dataclass(frozen=True)
class FermionHilbert:
    """Immutable metadata for a fixed-spin, optional point-group sector.

    The object is device independent and safe to compare or serialize.  Hot
    execution paths should call :meth:`device_filter` once and reuse the
    returned device tensors rather than rebuilding symmetry labels per move.
    """

    n_orbitals: int
    n_alpha: int
    n_beta: int
    orbital_symmetries: tuple[int, ...] | None = None
    target_symmetry: int | None = None

    def __post_init__(self) -> None:
        if self.n_orbitals <= 0:
            raise ValueError("n_orbitals must be positive")
        if not 0 <= self.n_alpha <= self.n_orbitals:
            raise ValueError("invalid alpha-electron count")
        if not 0 <= self.n_beta <= self.n_orbitals:
            raise ValueError("invalid beta-electron count")
        if self.target_symmetry is not None:
            if self.orbital_symmetries is None:
                raise ValueError("point-group mode requires orbital symmetry labels")
            if len(self.orbital_symmetries) != self.n_orbitals:
                raise ValueError("wrong number of orbital symmetry labels")

    @property
    def n_spin_orbitals(self) -> int:
        return 2 * self.n_orbitals

    @property
    def n_electrons(self) -> int:
        return self.n_alpha + self.n_beta

    @property
    def uses_point_group(self) -> bool:
        return self.target_symmetry is not None

    @property
    def requires_projection(self) -> bool:
        """Point-group sectors must filter symmetry-breaking connections."""

        return self.uses_point_group

    def without_point_group(self) -> FermionHilbert:
        """Return the containing fixed-spin sector without an irrep projection."""

        return FermionHilbert(self.n_orbitals, self.n_alpha, self.n_beta)

    def sector_size(self) -> int:
        """Return the exact number of determinants without enumerating them.

        Point-group dimensions are counted by dynamic programming over spatial
        orbitals, alpha/beta particle numbers, and XOR irreps.  This is a small
        host-side setup calculation and never appears in a VMC hot path.
        """

        if not self.uses_point_group:
            return comb(self.n_orbitals, self.n_alpha) * comb(self.n_orbitals, self.n_beta)
        assert self.orbital_symmetries is not None
        labels = self.orbital_symmetries
        symmetry_count = 1
        largest = max((*labels, int(self.target_symmetry or 0)))
        while symmetry_count <= largest:
            symmetry_count *= 2
        current = [
            [[0] * symmetry_count for _ in range(self.n_beta + 1)] for _ in range(self.n_alpha + 1)
        ]
        current[0][0][0] = 1
        for label in labels:
            following = [
                [[0] * symmetry_count for _ in range(self.n_beta + 1)]
                for _ in range(self.n_alpha + 1)
            ]
            for alpha in range(self.n_alpha + 1):
                for beta in range(self.n_beta + 1):
                    for irrep, count in enumerate(current[alpha][beta]):
                        if not count:
                            continue
                        following[alpha][beta][irrep] += count
                        if alpha < self.n_alpha:
                            following[alpha + 1][beta][irrep ^ label] += count
                        if beta < self.n_beta:
                            following[alpha][beta + 1][irrep ^ label] += count
                        if alpha < self.n_alpha and beta < self.n_beta:
                            following[alpha + 1][beta + 1][irrep] += count
            current = following
        return current[self.n_alpha][self.n_beta][int(self.target_symmetry)]

    def device_filter(self, device: torch.device | str) -> FermionSectorFilter:
        """Materialize a reusable point-group validator on ``device``."""

        requested_device = torch.device(device)
        labels = self.orbital_symmetries
        if labels is None:
            symmetry_bits = torch.empty(
                (self.n_orbitals, 0),
                dtype=torch.int64,
                device=requested_device,
            )
            bit_values = torch.empty(0, dtype=torch.int64, device=requested_device)
        else:
            largest = max((*labels, int(self.target_symmetry or 0)))
            bit_count = max(1, largest.bit_length())
            shifts = torch.arange(bit_count, dtype=torch.int64, device=requested_device)
            label_tensor = torch.tensor(labels, dtype=torch.int64, device=requested_device)
            symmetry_bits = torch.bitwise_and(
                torch.bitwise_right_shift(label_tensor[:, None], shifts[None, :]),
                1,
            )
            bit_values = torch.bitwise_left_shift(torch.ones_like(shifts), shifts)
        return FermionSectorFilter(
            n_orbitals=self.n_orbitals,
            n_alpha=self.n_alpha,
            n_beta=self.n_beta,
            target_symmetry=self.target_symmetry,
            symmetry_bits=symmetry_bits,
            bit_values=bit_values,
        )

    def state_symmetry(self, states: torch.Tensor) -> torch.Tensor:
        """Return XOR irrep labels using a transient device filter."""

        return self.device_filter(states.device).state_symmetry(states)

    def contains(self, states: torch.Tensor) -> torch.Tensor:
        """Return membership flags for an arbitrary batch of determinants."""

        return self.device_filter(states.device).contains(states)

    def enumerate(self, device: torch.device | str = "cpu") -> SectorBasis:
        """Enumerate the finite sector in monotonically increasing key order."""

        if self.n_spin_orbitals > 62:
            raise ValueError("exact enumeration currently supports at most 62 spin orbitals")
        labels = self.orbital_symmetries or (0,) * self.n_orbitals
        alpha = []
        for occupied in combinations(range(self.n_orbitals), self.n_alpha):
            key = sum(1 << (2 * orbital) for orbital in occupied)
            irrep = 0
            for orbital in occupied:
                irrep ^= labels[orbital]
            alpha.append((key, irrep))
        beta = []
        for occupied in combinations(range(self.n_orbitals), self.n_beta):
            key = sum(1 << (2 * orbital + 1) for orbital in occupied)
            irrep = 0
            for orbital in occupied:
                irrep ^= labels[orbital]
            beta.append((key, irrep))
        keys = [
            alpha_key | beta_key
            for alpha_key, alpha_irrep in alpha
            for beta_key, beta_irrep in beta
            if not self.uses_point_group or (alpha_irrep ^ beta_irrep) == self.target_symmetry
        ]
        key_tensor = torch.tensor(sorted(keys), dtype=torch.int64, device=device)
        states = decode_states(key_tensor, self.n_spin_orbitals)
        return SectorBasis(states, key_tensor)

    def reference_state(self, device: torch.device | str = "cpu") -> torch.Tensor:
        """Return an Aufbau determinant, or the first valid symmetry state."""

        state = torch.zeros(self.n_spin_orbitals, dtype=torch.bool, device=device)
        state[2 * torch.arange(self.n_alpha, device=device)] = True
        state[2 * torch.arange(self.n_beta, device=device) + 1] = True
        if bool(self.contains(state)):
            return state
        return self.enumerate(device).states[0]
