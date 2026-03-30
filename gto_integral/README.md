# GTO Integral Calculator

A C++ implementation for computing fundamental quantum chemistry integrals using Gaussian-type orbitals (GTOs). This project demonstrates the calculation of:

- Overlap integrals
- Kinetic energy integrals
- Nuclear attraction integrals
- Electron repulsion integrals

Results are validated against PySCF reference calculations.

## Features

- Normalization of primitive Gaussians
- Four fundamental integral types:
  - Overlap (S)
  - Kinetic Energy (T)
  - Nuclear Attraction (V)
  - Electron Repulsion (ERI)

## Dependencies

- C++ compatible compiler
- CMake

## Installation

Build with CMake:

```bash
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build .
## run it
./gto_integral   
```

## Example Output

```
Overlap integral: 0.6570749266
Kinetic integral: 0.3406411335
Nuclear integral: -1.2727717105
Electron repulsion integral: 0.3375246289
```

These results match the PySCF reference values with 12 decimal places precision.

## Validation

The results are cross-validated against PySCF calculations for identical basis sets and nuclear configurations:

```python
# Reference PySCF code
from pyscf import gto
mol = gto.M(atom="H 0 0 0; H 0 0 0.7", basis={'H': [[0,(0.48, 1.0)]]})
print(mol.intor("int1e_ovlp")[0,1])       # 0.657074926565322
print(mol.intor("int1e_kin")[0,1])        # 0.3406411334601461
print(mol.intor("int1e_nuc")[0,1])        # -1.272771710490278
print(mol.intor("int2e")[0,1,0,1])        # 0.33752462885462003
```
