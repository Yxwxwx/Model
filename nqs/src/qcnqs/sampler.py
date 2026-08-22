"""Persistent Metropolis sampling for fixed-spin determinant spaces.

The sampler depends only on Hilbert-space constraints, excitation layouts, and
the narrow log-wave-function interface.  It does not import a Hamiltonian or a
VMC driver.  Every chain operation is batched across ``n_chains`` in PyTorch;
Python controls only the inherently sequential Markov transition order.

Random choices for an entire advance are generated in bulk.  This preserves
the exact Markov process while reducing CUDA random-number kernel launches and
Python dispatch relative to drawing three tensors for every transition.
"""

from __future__ import annotations

from typing import Literal

import torch

from qcnqs.determinant import (
    DeterminantTransitionTemplates,
    make_transition_templates,
)
from qcnqs.hilbert import FermionHilbert, decode_states, encode_states
from qcnqs.interfaces import SampleBatch, SamplerStats
from qcnqs.wavefunction import LogAmplitude, LogWaveFunction

SamplingMode = Literal["metropolis", "exact"]


class FermionSampler:
    """Persistent fixed-spin Metropolis or exact finite-sector sampler.

    Single/double templates are symmetric in the forward and reverse
    directions, so the Metropolis ratio contains only ``|psi(new)/psi(old)|``
    raised to ``sampling_power``.  Point-group-breaking moves become explicit
    self loops.  Proposal construction itself preserves alpha/beta counts, so
    the hot transition kernel checks only the optional point-group constraint.
    """

    def __init__(
        self,
        hilbert: FermionHilbert,
        wave_function: LogWaveFunction,
        *,
        n_samples: int,
        n_chains: int,
        seed: int,
        device: torch.device | str,
        mode: SamplingMode = "metropolis",
        burn_in_sweeps: int = 16,
        sweep_size: int | None = None,
        sample_interval: int = 1,
        single_probability: float = 0.05,
        sampling_power: float = 2.0,
        exact_max_states: int = 100_000,
        compile_transitions: bool = False,
    ) -> None:
        if n_samples <= 0 or n_chains <= 0:
            raise ValueError("sample and chain counts must be positive")
        if n_samples % n_chains:
            raise ValueError("n_samples must be divisible by n_chains")
        if mode not in {"metropolis", "exact"}:
            raise ValueError(f"unknown sampling mode: {mode}")
        if burn_in_sweeps < 0 or sample_interval <= 0:
            raise ValueError("invalid burn-in or sampling interval")
        if sweep_size is not None and sweep_size <= 0:
            raise ValueError("sweep_size must be positive")
        if not 0.0 <= single_probability <= 1.0:
            raise ValueError("single_probability must lie in [0, 1]")
        if sampling_power <= 0:
            raise ValueError("sampling_power must be positive")
        if mode == "exact" and hilbert.sector_size() > exact_max_states:
            raise ValueError("exact sampling sector exceeds exact_max_states")
        self.space = hilbert
        self.wave_function = wave_function
        self.n_samples = int(n_samples)
        self.n_chains = int(n_chains)
        requested_device = torch.device(device)
        self.device = torch.empty(0, device=requested_device).device
        self.mode = mode
        self.burn_in_sweeps = int(burn_in_sweeps)
        self.sweep_size = sweep_size or hilbert.n_spin_orbitals
        self.sample_interval = int(sample_interval)
        self.single_probability = float(single_probability)
        self.sampling_power = float(sampling_power)
        self.generator = torch.Generator(device=self.device).manual_seed(seed)
        self._sector_filter = hilbert.device_filter(self.device)
        self._spatial_indices = torch.arange(hilbert.n_orbitals, device=self.device)
        self._chain_indices = torch.arange(self.n_chains, device=self.device)
        self.templates: DeterminantTransitionTemplates = make_transition_templates(
            hilbert.n_orbitals,
            hilbert.n_alpha,
            hilbert.n_beta,
        ).to(self.device)
        if (
            mode == "metropolis"
            and not self.templates.singles.shape[0]
            and not self.templates.doubles.shape[0]
        ):
            raise ValueError("the selected sector has no Metropolis moves")
        self._exact_basis = hilbert.enumerate(self.device) if mode == "exact" else None
        self.chain_states = self._initial_chain_states()
        self.chain_logabs = self._evaluate_logabs(self.chain_states)
        self._attempted = 0
        self._valid = torch.zeros((), dtype=torch.int64, device=self.device)
        self._accepted = torch.zeros((), dtype=torch.int64, device=self.device)
        self._transition_call = (
            torch.compile(self._transition_kernel, fullgraph=True, dynamic=False)
            if compile_transitions
            else self._transition_kernel
        )

    @property
    def hilbert(self) -> FermionHilbert:
        """Backward-compatible name for the sampler's immutable state space."""

        return self.space

    @torch.no_grad()
    def _evaluate_logabs(self, states: torch.Tensor) -> torch.Tensor:
        return self.wave_function.log_amplitude(states).logabs.detach()

    def _random_fixed_spin_states(self, count: int) -> torch.Tensor:
        states = torch.zeros(
            (count, self.space.n_spin_orbitals),
            dtype=torch.bool,
            device=self.device,
        )
        for spin, electrons in enumerate((self.space.n_alpha, self.space.n_beta)):
            scores = torch.rand(
                (count, self.space.n_orbitals),
                generator=self.generator,
                device=self.device,
            )
            occupied = scores.argsort(dim=1)[:, :electrons]
            states[:, spin::2].scatter_(1, occupied, True)
        return states

    def _initial_chain_states(self) -> torch.Tensor:
        if self.space.sector_size() <= 100_000:
            basis = self.space.enumerate(self.device)
            indices = torch.randint(
                basis.states.shape[0],
                (self.n_chains,),
                generator=self.generator,
                device=self.device,
            )
            return basis.states[indices].clone()
        states = self._random_fixed_spin_states(self.n_chains)
        if not self.space.uses_point_group:
            return states
        valid = self._sector_filter.contains(states)
        for _ in range(1_000):
            if bool(torch.all(valid)):
                return states
            replacements = self._random_fixed_spin_states(int((~valid).sum().item()))
            states[~valid] = replacements
            valid = self._sector_filter.contains(states)
        raise RuntimeError("failed to initialize chains in the target symmetry sector")

    def _occupied_virtual_tables(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        orbital = self._spatial_indices
        occupation = torch.stack((states[:, 0::2], states[:, 1::2]), dim=1)
        occupied = torch.where(occupation, orbital, self.space.n_orbitals).sort(dim=-1).values
        virtual = torch.where(~occupation, orbital, self.space.n_orbitals).sort(dim=-1).values
        return occupied, virtual

    def _lookup(self, table: torch.Tensor, spin: torch.Tensor, slot: torch.Tensor) -> torch.Tensor:
        return table[self._chain_indices, spin, slot]

    def _single_proposals(
        self,
        states: torch.Tensor,
        occupied: torch.Tensor,
        virtual: torch.Tensor,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        if not self.templates.singles.shape[0]:
            return states
        selected = self.templates.singles[indices]
        hole_spin, hole_slot, particle_spin, particle_slot = selected.T
        holes = 2 * self._lookup(occupied, hole_spin, hole_slot) + hole_spin
        particles = 2 * self._lookup(virtual, particle_spin, particle_slot) + particle_spin
        proposed = states.clone()
        proposed.scatter_(1, holes[:, None], False)
        proposed.scatter_(1, particles[:, None], True)
        return proposed

    def _double_proposals(
        self,
        states: torch.Tensor,
        occupied: torch.Tensor,
        virtual: torch.Tensor,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        if not self.templates.doubles.shape[0]:
            return states
        selected = self.templates.doubles[indices]
        hs0, hi0, hs1, hi1, ps0, pi0, ps1, pi1 = selected.T
        hole_a = 2 * self._lookup(occupied, hs0, hi0) + hs0
        hole_b = 2 * self._lookup(occupied, hs1, hi1) + hs1
        particle_a = 2 * self._lookup(virtual, ps0, pi0) + ps0
        particle_b = 2 * self._lookup(virtual, ps1, pi1) + ps1
        proposed = states.clone()
        proposed.scatter_(1, hole_a[:, None], False)
        proposed.scatter_(1, hole_b[:, None], False)
        proposed.scatter_(1, particle_a[:, None], True)
        proposed.scatter_(1, particle_b[:, None], True)
        return proposed

    def _transition_kernel(
        self,
        chain_states: torch.Tensor,
        chain_logabs: torch.Tensor,
        single_indices: torch.Tensor,
        double_indices: torch.Tensor,
        proposal_uniform: torch.Tensor,
        acceptance_uniform: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        occupied, virtual = self._occupied_virtual_tables(chain_states)
        if (
            self.templates.singles.shape[0]
            and self.templates.doubles.shape[0]
            and 0.0 < self.single_probability < 1.0
        ):
            singles = self._single_proposals(chain_states, occupied, virtual, single_indices)
            doubles = self._double_proposals(chain_states, occupied, virtual, double_indices)
            choose_single = proposal_uniform < self.single_probability
            proposed = torch.where(choose_single[:, None], singles, doubles)
        elif self.templates.singles.shape[0] and (
            not self.templates.doubles.shape[0] or self.single_probability > 0.0
        ):
            proposed = self._single_proposals(chain_states, occupied, virtual, single_indices)
        else:
            proposed = self._double_proposals(chain_states, occupied, virtual, double_indices)
        # Excitation templates conserve alpha/beta counts by construction.  A
        # full particle-count reduction here would therefore be redundant on
        # every transition; only the optional point-group irrep is checked.
        if self.space.uses_point_group:
            valid = self._sector_filter.symmetry_mask(proposed)
        # The wave function is defined on the complete fixed-spin space.  It is
        # therefore safe to evaluate every proposed state in one dense batch;
        # invalid point-group moves are masked after the ratio is computed.
        proposed_logabs = self._evaluate_logabs(proposed)
        log_ratio = self.sampling_power * (proposed_logabs - chain_logabs)
        log_uniform = torch.log(acceptance_uniform)
        accepted = log_uniform < log_ratio.clamp_max(0.0)
        if self.space.uses_point_group:
            accepted = valid & accepted
        states = torch.where(accepted[:, None], proposed, chain_states)
        logabs = torch.where(accepted, proposed_logabs, chain_logabs)
        accepted_count = accepted.sum()
        valid_count = valid.sum() if self.space.uses_point_group else accepted_count
        return states, logabs, valid_count, accepted_count

    def _draw_randomness(
        self,
        count: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate all random tensors for ``count`` sequential transitions.

        Drawing a ``[count, n_chains]`` tensor is substantially cheaper on CUDA
        than launching a random-number kernel inside every Markov transition.
        In a pure-single or pure-double run, the unused index argument aliases
        the selected tensor because its static kernel branch never reads it.
        """

        shape = (count, self.n_chains)
        has_singles = bool(self.templates.singles.shape[0])
        has_doubles = bool(self.templates.doubles.shape[0])
        mixture = has_singles and has_doubles and 0.0 < self.single_probability < 1.0
        uses_singles = mixture or (has_singles and (not has_doubles or self.single_probability > 0))
        if mixture:
            single_indices = torch.randint(
                self.templates.singles.shape[0],
                shape,
                generator=self.generator,
                device=self.device,
            )
            double_indices = torch.randint(
                self.templates.doubles.shape[0],
                shape,
                generator=self.generator,
                device=self.device,
            )
        elif uses_singles:
            single_indices = torch.randint(
                self.templates.singles.shape[0],
                shape,
                generator=self.generator,
                device=self.device,
            )
            double_indices = single_indices
        else:
            double_indices = torch.randint(
                self.templates.doubles.shape[0],
                shape,
                generator=self.generator,
                device=self.device,
            )
            single_indices = double_indices
        if mixture:
            uniform = torch.rand(
                (2, *shape),
                generator=self.generator,
                device=self.device,
            )
            proposal_uniform, acceptance_uniform = uniform.unbind(0)
        else:
            acceptance_uniform = torch.rand(
                shape,
                generator=self.generator,
                device=self.device,
            )
            # The pure-single and pure-double branches do not read this input.
            proposal_uniform = acceptance_uniform
        return single_indices, double_indices, proposal_uniform, acceptance_uniform

    @torch.no_grad()
    def _run_transitions(
        self,
        count: int,
        *,
        collect_every: int | None = None,
    ) -> list[torch.Tensor]:
        """Advance all chains sequentially while batching RNG and statistics.

        Markov states remain causally dependent, so the transition loop cannot
        be vectorized over time.  All work *within* a transition is vectorized
        over chains, and proposal counters are reduced on the device only once
        after the loop.  ``collect_every`` records chain snapshots without a
        device synchronization.
        """

        if count < 0:
            raise ValueError("transition count must be non-negative")
        if collect_every is not None and (collect_every <= 0 or count % collect_every):
            raise ValueError("collect_every must be positive and divide transition count")
        if count == 0:
            return []
        single_indices, double_indices, proposal_uniform, acceptance_uniform = (
            self._draw_randomness(count)
        )
        valid_counts = []
        accepted_counts = []
        collected = []
        for index in range(count):
            states, logabs, valid, accepted = self._transition_call(
                self.chain_states,
                self.chain_logabs,
                single_indices[index],
                double_indices[index],
                proposal_uniform[index],
                acceptance_uniform[index],
            )
            self.chain_states = states
            self.chain_logabs = logabs
            if self.space.uses_point_group:
                valid_counts.append(valid)
            accepted_counts.append(accepted)
            if collect_every is not None and (index + 1) % collect_every == 0:
                collected.append(states.clone())
        self._attempted += count * self.n_chains
        if self.space.uses_point_group:
            self._valid += torch.stack(valid_counts).sum()
        else:
            self._valid += count * self.n_chains
        self._accepted += torch.stack(accepted_counts).sum()
        return collected

    def transition(self) -> None:
        """Advance every persistent chain by one Metropolis proposal."""

        self._run_transitions(1)

    def sweep(self) -> None:
        """Advance every chain by the configured number of proposals."""

        self._run_transitions(self.sweep_size)

    def _reset_stats(self) -> None:
        self._attempted = 0
        self._valid.zero_()
        self._accepted.zero_()

    def _stats(self) -> SamplerStats:
        return SamplerStats(
            attempted=self._attempted,
            valid=int(self._valid.item()),
            accepted=int(self._accepted.item()),
        )

    @torch.no_grad()
    def refresh(self) -> None:
        """Refresh persistent-chain amplitudes after parameters change."""

        self.chain_logabs = self._evaluate_logabs(self.chain_states)

    def _make_batch(self, raw_states: torch.Tensor) -> SampleBatch:
        """Compress repeated samples while preserving their exact weights."""

        keys = encode_states(raw_states)
        unique_keys, inverse, counts = torch.unique(
            keys,
            sorted=True,
            return_inverse=True,
            return_counts=True,
        )
        states = decode_states(unique_keys, self.space.n_spin_orbitals)
        amplitudes = self.wave_function.log_amplitude(states)
        probabilities = counts.to(amplitudes.logabs.dtype) / raw_states.shape[0]
        return SampleBatch(
            states=states,
            counts=counts,
            probabilities=probabilities,
            inverse=inverse,
            raw_states=raw_states,
            amplitudes=LogAmplitude(amplitudes.logabs.detach(), amplitudes.sign.detach()),
            stats=self._stats(),
            exact_probabilities=False,
        )

    @torch.no_grad()
    def sample(self, *, discard: bool = False) -> SampleBatch:
        """Return exact weights or a compressed persistent-chain trace.

        ``discard=True`` advances each chain by ``burn_in_sweeps`` before
        recording states.  Subsequent parameter steps keep the chains
        persistent and refresh their amplitudes after every model update.
        """

        self._reset_stats()
        if self.mode == "exact":
            assert self._exact_basis is not None
            amplitudes = self.wave_function.log_amplitude(self._exact_basis.states)
            probabilities = torch.softmax(
                self.sampling_power * amplitudes.logabs,
                dim=0,
            )
            finite = (
                torch.isfinite(amplitudes.logabs) & (amplitudes.sign != 0) & (probabilities > 0)
            )
            states = self._exact_basis.states[finite]
            probabilities = probabilities[finite]
            probabilities = probabilities / probabilities.sum()
            selected = LogAmplitude(
                amplitudes.logabs[finite].detach(),
                amplitudes.sign[finite].detach(),
            )
            count = states.shape[0]
            return SampleBatch(
                states=states,
                counts=torch.ones(count, dtype=torch.int64, device=self.device),
                probabilities=probabilities,
                inverse=torch.empty(0, dtype=torch.int64, device=self.device),
                raw_states=states,
                amplitudes=selected,
                stats=self._stats(),
                exact_probabilities=True,
            )
        if discard:
            self._run_transitions(self.burn_in_sweeps * self.sweep_size)
        transitions_per_trace = self.sample_interval * self.sweep_size
        trace_count = self.n_samples // self.n_chains
        collected = self._run_transitions(
            trace_count * transitions_per_trace,
            collect_every=transitions_per_trace,
        )
        return self._make_batch(torch.cat(collected, dim=0))
