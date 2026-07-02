"""FGH — Fourier Grid Hamiltonian vibrational solver (Rust core)."""

from fgh._fgh_core import hello, add
from fgh.fgh import FGH1D

__all__ = ["hello", "add", "FGH1D"]
