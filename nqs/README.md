# qcnqs

`qcnqs` is a readable, pure-Python library for fermionic neural quantum-state VMC.
It currently provides exact connected operators for ab-initio quantum-chemistry
Hamiltonians and spinful 1D/2D Hubbard models. PySCF supplies molecular integrals and
PyTorch supplies batched CPU/CUDA kernels; no project-specific C++ or CUDA extension is
required.

The implementation keeps the useful numerical ideas from the optimized C++ reference:
fixed-shape determinant transitions, batch-local deduplication of connected determinants,
log-domain determinant amplitudes, and dense/MinSR formulations of the sampled QGT. It
deliberately omits persistent local-energy caches, manual block scheduling, and stochastic
Hamiltonian truncation so the mathematical path remains inspectable.

The verified molecular target is N2/STO-3G at 1.1 Angstrom:

- 20 spin orbitals with `N_alpha = N_beta = 7`;
- full fixed-spin space (14,400 determinants) and D2h/Ag space (1,824 determinants);
- one-hidden-layer, single-determinant Backflow with `H=20` (6,300 parameters);
- exact one- and two-electron Slater--Condon local energy;
- persistent Metropolis sampling and exact finite-sector expectation mode;
- materialized dense SR and sample-space MinSR in FP64;
- a parameter- and sampler-matched NetKet benchmark.

On an RTX 4090 D, the locked three-seed matched run takes `19.38 +/- 1.19 s` in
qcnqs and `339.94 +/- 24.03 s` in NetKet: a 17.54x end-to-end speedup. Their exact
post-training model errors are comparable (`33.33 +/- 5.51 mHa` versus
`32.17 +/- 5.56 mHa`). The recommended qcnqs point-group mode improves the exact model
error to `13.80 +/- 0.68 mHa` in `29.62 +/- 0.22 s`.

## Install and test

Python 3.12 and a CUDA 12.6 PyTorch wheel are locked by `uv.lock`:

```bash
uv sync
uv run pytest
uv run ruff check .
```

The reproducible NetKet comparison is isolated in an optional dependency group:

```bash
uv sync --group benchmark
```

## Run a calculation

qcnqs is a library and does not install a command-line program. The root
[main.py](main.py) is an ordinary example script. Edit its single nested `PARAMETERS`
dictionary and run:

```bash
uv run python main.py
```

The default selects the N2 FCIDUMP, its Ag point-group sector, the determinant Backflow
model, Metropolis sampling, and SR optimization. The same dictionary also contains inactive
`hubbard_1d` and `hubbard_2d` subsections; selecting either requires only changing
`PARAMETERS["system"]["kind"]`. Nothing in the library reads global configuration or
command-line flags, and applications may instead construct these objects directly.

For example, a minimal 1D Hubbard composition is:

```python
import torch

from qcnqs import DeterminantBackflow, FermionSampler, Hubbard1D, VMCDriver

operator = Hubbard1D(
    n_sites=10,
    n_alpha=5,
    n_beta=5,
    hopping=1.0,
    interaction=4.0,
    periodic=True,
    device="cuda",
)
space = operator.space
model = DeterminantBackflow(
    space.n_spin_orbitals,
    space.n_electrons,
    hidden_units=20,
    device="cuda",
    dtype=torch.float64,
)
sampler = FermionSampler(
    space,
    model,
    n_samples=4096,
    n_chains=1024,
    seed=3,
    device="cuda",
)
driver = VMCDriver(operator, space, model, sampler)
result = driver.run(300)
```

`Hubbard2D(rows, columns, n_alpha, n_beta, ...)` uses row-major square-lattice sites.
Both Hubbard constructors accept open or periodic boundaries. `HubbardHamiltonian`
accepts an arbitrary list of unique undirected bonds when a custom lattice is needed.

## N2 input and independent validation

Generate the FCIDUMP used by the default example:

```bash
uv run python benchmark/generate_n2.py
```

This independent validation constructs the complete Ag matrix, checks Hermiticity,
diagonalizes it, and compares its ground state with PySCF FCI:

```bash
uv run python benchmark/validate_n2.py \
  benchmark/data/N2.STO3G.RHF.FCIDUMP --device cpu
```

On the checked FCIDUMP, the 1,824-dimensional matrix has a maximum Hermiticity residual of
`2.22e-16`; its lowest eigenvalue differs from PySCF FCI by `9.95e-14 Ha`.

## Code map and extension boundary

- `interfaces.py` defines implementation-neutral operator, state-space, and sampler contracts.
- `determinant.py` owns fixed-sector move layouts and fermionic parity primitives; it does
  not implement excited-state calculations.
- `operator_tools.py` provides exact finite-basis apply/dense operations shared by all
  connected Hamiltonians.
- `chemistry/` owns PySCF and FCIDUMP conversion; spatial ERIs use chemist notation.
- `assembly.py` maps molecular metadata to a selected Hilbert sector.
- `hilbert.py` owns fixed-spin enumeration, compact determinant keys, and abelian
  point-group sectors.
- `hamiltonian.py` owns exact diagonal, single, and double Slater--Condon matrix elements.
- `hubbard.py` owns arbitrary-bond and nearest-neighbor 1D/2D Hubbard matrix elements.
- `models/` implements the current real determinant Backflow behind `LogWaveFunction`.
- `sampler.py` owns persistent Markov chains and exact finite-sector probabilities.
- `local_energy.py` evaluates generic exact operator connections and deduplicates within a
  batch.
- `geometry.py` builds a sampled QGT and solves either parameter-space SR or MinSR.
- `vmc.py` owns optimization and timing without embedding Hamiltonian or model details.

The allowed dependency graph is executable policy in `tests/test_architecture.py`. See
[ARCHITECTURE.md](ARCHITECTURE.md) for layer ownership, tensor contracts, performance
decisions, and extension recipes.

The geometry layer accepts a generic centered target. VMC contributes the energy-specific
target `2(E_local - <E>)`; a future real- or imaginary-time TDVP driver can reuse
`SampledQGT.solve` with its own target and integrator.

The current implementation is real-valued. Compact keys support at most 62 spin orbitals,
the dense quantum-chemistry spin-orbital two-electron tensor costs `O(M^4)` memory, and
exact enumeration is intended only for validation-sized sectors. These are explicit
replacement points, not hidden assumptions in the VMC driver.

See [benchmark/README.md](benchmark/README.md) for the matched protocol, configuration,
hardware, and measured results.
