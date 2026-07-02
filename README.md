# Model

Computational chemistry and physics code developed during PhD research. Each directory is an **independent sub-project** — browse the individual READMEs for build instructions and dependencies.

## Modules

### Electronic Structure

| Directory | Description | Language |
|-----------|-------------|----------|
| [teach_rhf/](teach_rhf/) | Educational RHF, UHF, GHF, and 4-component Dirac-HF from scratch | Python |
| [gto_integral/](gto_integral/) | Standalone GTO integral engine (overlap, kinetic, nuclear, ERI) | C++ |
| [slater_condon/](slater_condon/) | FCI solver using Slater-Condon rules with Davidson diagonalization | C++ (MKL) |

### DMRG & Strong Correlation

| Directory | Description | Language |
|-----------|-------------|----------|
| [idmrg/](idmrg/) | Toy DMRG for 1D Heisenberg model — superblock and MPS | Python / C++ |
| [Auger/](Auger/) | Auger electron spectroscopy via DMRG with RAS-constrained MPS | Python (pyblock2) |

### Cavity QED

| Directory | Description | Language |
|-----------|-------------|----------|
| [cqed/](cqed/) | CQED-RHF and polaritonic CQED for molecules in optical cavities | Python (pyblock2) |

### Nuclear-Electronic Orbital (NEO)

| Directory | Description | Language |
|-----------|-------------|----------|
| [neo/](neo/) | NEO-RHF for electron-proton correlation, NEO-DMRG with U(1)/SZ | Python (pyblock2) |

### Dynamics

| Directory | Description | Language |
|-----------|-------------|----------|
| [toy_fssh/](toy_fssh/) | Fewest-Switches Surface Hopping for Tully's model systems | Python / C++ |

### Vibrational Solvers

| Directory | Description | Language |
|-----------|-------------|----------|
| [dvr/](dvr/) | Discrete Variable Representation solver | Rust + PyO3 |
| [fgh/](fgh/) | Fourier Grid Hamiltonian solver | Rust + PyO3 |

### Machine Learning

| Directory | Description | Language |
|-----------|-------------|----------|
| [attention/](attention/) | Multi-head self-attention from scratch | Python (PyTorch) |
| [minigpt/](minigpt/) | Mini GPT-2 training on TinyStories | Python (JAX + Flax) |

### Mixed Basis

| Directory | Description | Language |
|-----------|-------------|----------|
| [Mixed-GTO-PW/](Mixed-GTO-PW/) | Mixed Gaussian / plane-wave integrals with pybind11 wrapper | C++ / Python |

## License

MIT — see [LICENSE](LICENSE).
