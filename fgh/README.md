# FGH — Fourier Grid Hamiltonian

Fourier Grid Hamiltonian solver for the vibrational Schrödinger equation. Rust core (PyO3 bindings) with FFT-accelerated kinetic energy operator.

Reference: C. C. Marston & G. G. Balint-Kurti, J. Chem. Phys. 91, 3571 (1989).

## Features

- Uniform FGH grid generation
- Analytic kinetic energy matrix
- FFT-based kinetic operator application (avoids dense matrix construction)
- Dense Hamiltonian diagonalization via LAPACK (nalgebra)
- Designed to interface with [PySCF](https://pyscf.org/) for ab-initio single-point energies

## Installation

```bash
cd fgh
pip install maturin
maturin develop --release
```

## Quick Start

```python
from fgh import solve_fgh_1d
import numpy as np

def v(x):
    return 0.5 * x**2  # harmonic oscillator

fgh, energies, wfns = solve_fgh_1d(
    v, x_min=-10, x_max=10, n_points=256, mass=1.0, n_states=5
)
print(energies)  # [0.5, 1.5, 2.5, 3.5, 4.5]
```

## Planned

- [ ] PySCF integration for ab-initio potential curves
- [ ] Benchmark against NEO method
- [ ] Iterative (Lanczos) solver for large grids
- [ ] Multidimensional FGH (direct product)

## License

MIT
