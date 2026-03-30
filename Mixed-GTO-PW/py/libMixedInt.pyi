import numpy as np
from typing import Any

def int1e_ovlp_cart(
    buf: np.ndarray,
    shls: np.ndarray,
    atm: np.ndarray,
    natm: int,
    bas: np.ndarray,
    nbas: int,
    env: np.ndarray,
    k_vector: np.ndarray,
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
) -> None: ...
