# Tully's FSSF Toy Code


This project implements the Fewest-Switches Surface Hopping (FSSH) method proposed by Tully (see Tully, J.C., J. Chem. Phys. (1990) 93, 1061). The method is simple yet practical. This project provides both Python and C++ versions, based on the first model in Tully's 1990 paper, with 20 initial momenta. By default, each momentum runs 1000 trajectories.

The Python version is based on `Numpy`, with `Matplotlib `for plotting and `mpi4py` for parallel processing. The C++ version depends on the `Eigen3 `library (included in the ThirdParty directory) and uses multithreading via the standard library's `Thread` class; direct plotting functionality is currently not provided.

The Python version produces correct results but is slow (over 3 hours). The C++ version is much faster (over 200 seconds) but produces incorrect results for high-momentum systems.

---
