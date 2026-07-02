# DVR — Discrete Variable Representation

Colbert–Miller sinc-DVR solver for the vibrational Schrödinger equation, implemented in Rust with Python bindings via PyO3.

## Features

- Uniform sinc-DVR grid generation
- Analytic kinetic energy matrix (Colbert–Miller, JCP 1992)
- Hamiltonian construction + LAPACK diagonalization (nalgebra)
- Designed to interface with [PyPES](https://github.com/.../pypes) for potential energy surfaces

## Installation

```bash
cd dvr
pip install maturin
maturin develop --release
```

## Quick Start

```python
from dvr import solve_dvr_1d
import numpy as np

def v(x):
    return 0.5 * x**2  # harmonic oscillator

dvr, energies, wfns = solve_dvr_1d(
    v, x_min=-6, x_max=6, n_points=150, mass=1.0, n_states=5
)
print(energies)  # [0.5, 1.5, 2.5, 3.5, 4.5]
```

## Planned

- [ ] Multidimensional DVR (direct product / contracted basis)
- [ ] PyPES integration for PES evaluation
- [ ] Benchmark against PyVCI and Block2
- [ ] Non-uniform grids (Gauss-Hermite, Gauss-Legendre DVR)

## License

MIT
