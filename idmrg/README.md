# DMRG toy code

Here is the documentation for the code:

This project provides two simple implementations of the Density Matrix Renormalization Group (DMRG) method, one based on the superblock approach and the other on Matrix Product States (MPS), for calculating the 1D Heisenberg model. The logic is straightforward, intended solely for learning purposes.

The `mps.py` code relies on NumPy for tensor calculations and requires SciPy for performing SVD (Singular Value Decomposition). The superblock implementation also depends on NumPy and uses `scipy.sparse` for storing sparse matrices and executing SVD and other related algorithms.

To run, use the following command:

```bash
python superblock.py
python mps.py 
```
