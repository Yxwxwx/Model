# qcnqs architecture

This document defines module ownership, dependency direction, tensor contracts, and the
performance rules used by qcnqs. The rules are enforced by
`tests/test_architecture.py`; changing an import edge requires an explicit architecture
decision rather than an accidental convenience import.

## Library boundary and dependency direction

`qcnqs` contains reusable numerical components only. The root `main.py` is an editable
user script and the composition root: it selects a Hamiltonian, state space, model, sampler,
and driver through one `PARAMETERS` dictionary. It is deliberately outside `src/qcnqs`.

```text
user main.py
  |-- chemistry -> assembly -> hilbert
  |-- QCHamiltonian ----\
  |-- Hubbard* ----------+-> interfaces + determinant
  |                      \-> operator_tools -> hilbert
  |-- FermionSampler ------> interfaces + determinant
  |                          + hilbert + wavefunction
  |-- DeterminantBackflow --> wavefunction
  `-- VMCDriver -----------> interfaces + local_energy + geometry
                                                |            |
                                           wavefunction   autodiff
```

Arrows point from a consumer to a dependency. `hamiltonian.py` and `hubbard.py` are peer
implementations of `ConnectedOperator`; neither the sampler nor VMC driver imports either
one. The driver imports structural protocols and generic algorithms, not
`QCHamiltonian`, `Hubbard1D`, `FermionSampler`, or `DeterminantBackflow`.
The package root exposes convenience symbols through PEP 562 lazy imports, so importing
`qcnqs` does not eagerly load PySCF or concrete execution layers.

| Module/layer | Owns | Must not own |
|---|---|---|
| `chemistry/` | Integral input, PySCF/FCIDUMP conversion, FCI reference | Hilbert or VMC choices |
| `assembly.py` | Conversion from molecular metadata to a selected Hilbert sector | Numerical kernels |
| `hilbert.py` | Fixed-spin constraints, XOR irreps, enumeration, state keys | Integrals or models |
| `determinant.py` | Occupied/virtual slot layouts and fermionic parity | Hamiltonian coefficients or excited-state methods |
| `interfaces.py` | Transport dataclasses and structural protocols | Concrete algorithms |
| `operator_tools.py` | Generic exact finite-basis application and dense diagnostics | Domain Hamiltonians |
| `hamiltonian.py` | Spin-integral expansion and exact Slater--Condon rows | Sampling or optimization |
| `hubbard.py` | Arbitrary-bond, 1D, and 2D spinful Hubbard rows | Sampling or optimization |
| `models/` | Differentiable log wave functions | Hamiltonian or sampler behavior |
| `sampler.py` | Persistent chains, proposals, exact finite-sector weights | Local energy or SR |
| `local_energy.py` | Generic connected-operator estimator | Concrete Hamiltonian classes |
| `geometry.py` | Centered Jacobians, dense QGT, MinSR, regularized solves | Energy-specific orchestration |
| `vmc.py` | Pipeline ordering, SR energy target, timings, parameter update | Input formats or concrete classes |

## Runtime data flow

1. A user script constructs a `FermionHilbert` and a compatible connected operator.
   Molecular calculations may first create an immutable CPU-FP64 `MolecularProblem` from
   PySCF or FCIDUMP; Hubbard constructors create their fixed-spin space directly.
2. The script independently creates the model and sampler on the requested device, then
   passes all components to `VMCDriver`.
3. The sampler advances persistent chains and compresses repeated determinants into unique
   states with exact count weights.
4. The local-energy estimator requests fixed-width connection rows from the operator,
   projects them into the requested state space, and evaluates each unique connected state
   once within the batch.
5. `torch.func.vmap(grad(...))` creates the per-sample log-derivative matrix. `SampledQGT`
   centers and weights it, then solves the smaller dense or sample-space system.
6. The driver applies the flat SR direction and tells the sampler to refresh chain
   amplitudes. No numerical component reaches into another component's private state.

## Tensor contracts

All state tensors are boolean and use interleaved spin orbitals:

```text
[spatial/site 0 alpha, spatial/site 0 beta, spatial/site 1 alpha, spatial/site 1 beta, ...]
```

| Value | Shape |
|---|---|
| Sample states | `[samples, spin_orbitals]` |
| Connected states | `[samples, connection_slots, spin_orbitals]` |
| Matrix elements | `[samples, connection_slots]` |
| Backflow orbital matrices | `[samples, spin_orbitals, electrons]` |
| Selected determinant matrices | `[samples, electrons, electrons]` |
| Log-derivative matrix | `[unique_samples, parameters]` |
| Weighted Jacobian factor | `[unique_samples, parameters]` |

Floating-point tensors follow the model/operator dtype and execution device. Python scalars
are created only for explicit validation, logging, or synchronized stage timings.

## PyTorch performance policy

The implementation remains readable Python while moving scale-dependent work into PyTorch:

- Slater--Condon singles/doubles and Hubbard hops use fixed templates plus `gather`,
  `scatter`, and advanced indexing; there is no Python loop over samples or connections.
- Metropolis moves are vectorized over all chains. Random tensors are generated for an
  entire advance in bulk; only the causally sequential time loop remains in Python.
- Static orbital indices, transition templates, bonds, and symmetry-bit tensors are
  allocated once per device and reused. These are immutable execution metadata, not
  model/sample caches.
- Point-group XOR reduction is expressed as parallel bit-parity tensor operations.
- Repeated sampled and connected determinants are compressed with `torch.unique`; exact
  count weights preserve the estimator.
- Per-sample derivatives use `torch.func.functional_call`, `grad`, and `vmap`.
- QGT solves use `torch.linalg.cholesky_ex` in FP64 and choose parameter-space or
  sample-space MinSR according to the smaller matrix dimension.
- Public boundaries validate particle counts. Trusted sampler/basis paths skip redundant
  CUDA reductions, eliminating device-to-host synchronization from the steady path.
- `torch.compile` remains opt-in for transition kernels because cold compilation dominates
  the verified N2/300-step workload.

Persistent local-energy caches, stochastic connection truncation, and manual GPU block
schedulers are intentionally excluded. They can be introduced behind the existing contracts
if a larger benchmark demonstrates that their complexity is necessary.

## Extension recipes

### New connected operator

Implement `ConnectedOperator.connections(states, validate=...)` and return a fixed-width
`Connections` object. `operator_tools.py` supplies optional finite-basis application and
dense-matrix diagnostics. Neither the sampler nor VMC driver needs to change.

### New sampler

Implement `VariationalSampler.sample()` and `refresh()`, expose its `space` and `device`,
and return a weighted `SampleBatch`. The driver does not require Markov-chain internals.

### New wave function

Subclass `torch.nn.Module`, implement `log_amplitude(states) -> LogAmplitude`, and make
`forward(states)` return differentiable `logabs`. The sampler, local energy,
`torch.func` derivative path, QGT, and exact diagnostics then remain reusable.

### TDVP

Reuse `SampledQGT.from_model` and call `SampledQGT.solve` with the centered real- or
imaginary-time target produced by a TDVP equation. A TDVP driver should own its integrator
and time state, just as `VMCDriver` owns the energy target and SGD step.
