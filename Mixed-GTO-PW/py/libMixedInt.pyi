"""precision="low": double (default); precision="high": long double.

Use precision="high" for high precision. This selects internal arithmetic;
outputs are complex128 and the parameter is not an error guarantee.
_sph kernels include the raw-Cartesian angular normalization and support l=0..6.
Contractions are outermost, followed by x,y,z for p or m=-l..l for l>=2.
"""
import numpy as np
from typing import Literal

def int1e_ovlp_cart(
    buf: np.ndarray,
    shls: np.ndarray,
    atm: np.ndarray,
    natm: int,
    bas: np.ndarray,
    nbas: int,
    env: np.ndarray,
    k_vector: np.ndarray,
    precision: Literal["low", "high"] = "low",
) -> None: ...
def int1e_kin_cart(
    buf: np.ndarray,
    shls: np.ndarray,
    atm: np.ndarray,
    natm: int,
    bas: np.ndarray,
    nbas: int,
    env: np.ndarray,
    k_vector: np.ndarray,
    precision: Literal["low", "high"] = "low",
) -> None: ...
def int1e_nuc_cart(
    buf: np.ndarray,
    shls: np.ndarray,
    atm: np.ndarray,
    natm: int,
    bas: np.ndarray,
    nbas: int,
    env: np.ndarray,
    k_vector: np.ndarray,
    precision: Literal["low", "high"] = "low",
) -> None: ...
def int2e_cart(
    buf: np.ndarray,
    shls: np.ndarray,
    atm: np.ndarray,
    natm: int,
    bas: np.ndarray,
    nbas: int,
    env: np.ndarray,
    k_vector: np.ndarray,
    precision: Literal["low", "high"] = "low",
) -> None: ...
def int2e_gppg_cart(
    buf: np.ndarray,
    shls: np.ndarray,
    atm: np.ndarray,
    natm: int,
    bas: np.ndarray,
    nbas: int,
    env: np.ndarray,
    k_vector: np.ndarray,
    l_vector: np.ndarray,
    precision: Literal["low", "high"] = "low",
) -> None: ...


def cart2sph_coeff(angular: int) -> np.ndarray: ...

def int1e_ovlp_sph(
    buf: np.ndarray,
    shls: np.ndarray,
    atm: np.ndarray,
    natm: int,
    bas: np.ndarray,
    nbas: int,
    env: np.ndarray,
    k_vector: np.ndarray,
    precision: Literal["low", "high"] = "low",
) -> None: ...
def int1e_kin_sph(
    buf: np.ndarray,
    shls: np.ndarray,
    atm: np.ndarray,
    natm: int,
    bas: np.ndarray,
    nbas: int,
    env: np.ndarray,
    k_vector: np.ndarray,
    precision: Literal["low", "high"] = "low",
) -> None: ...
def int1e_nuc_sph(
    buf: np.ndarray,
    shls: np.ndarray,
    atm: np.ndarray,
    natm: int,
    bas: np.ndarray,
    nbas: int,
    env: np.ndarray,
    k_vector: np.ndarray,
    precision: Literal["low", "high"] = "low",
) -> None: ...
def int2e_sph(
    buf: np.ndarray,
    shls: np.ndarray,
    atm: np.ndarray,
    natm: int,
    bas: np.ndarray,
    nbas: int,
    env: np.ndarray,
    k_vector: np.ndarray,
    precision: Literal["low", "high"] = "low",
) -> None: ...
def int2e_gppg_sph(
    buf: np.ndarray,
    shls: np.ndarray,
    atm: np.ndarray,
    natm: int,
    bas: np.ndarray,
    nbas: int,
    env: np.ndarray,
    k_vector: np.ndarray,
    l_vector: np.ndarray,
    precision: Literal["low", "high"] = "low",
) -> None: ...
