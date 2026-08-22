"""Construct NetKet's spin-conserving operator from PySCF integrals."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import sparse
from netket.experimental.operator import (
    ParticleNumberAndSpinConservingFermioperator2nd,
)
from netket.hilbert import SpinOrbitalFermions
from pyscf import ao2mo


def from_pyscf_integrals(
    h1e: np.ndarray,
    eri: np.ndarray,
    core_energy: float,
    electron_counts: Sequence[int],
    *,
    cutoff: float = 1.0e-11,
) -> ParticleNumberAndSpinConservingFermioperator2nd:
    h1e = np.asarray(h1e)
    if h1e.ndim != 2 or h1e.shape[0] != h1e.shape[1]:
        raise ValueError("h1e must be square")
    n_orbitals = h1e.shape[0]
    n_alpha, n_beta = (int(value) for value in electron_counts)
    eri = np.asarray(eri)
    if eri.shape != (n_orbitals,) * 4:
        eri = ao2mo.restore(1, eri, n_orbitals)
    # NetKet stores a^dagger_p a^dagger_r a_s a_q, so (pq|rs) -> (p,r,s,q).
    eri = eri.transpose(0, 2, 3, 1)
    h1e = np.where(np.abs(h1e) > cutoff, h1e, 0.0)
    eri = np.where(np.abs(eri) > cutoff, eri, 0.0)
    hilbert = SpinOrbitalFermions(
        n_orbitals,
        s=0.5,
        n_fermions_per_spin=(n_alpha, n_beta),
    )
    arrays = [
        np.asarray(core_energy).item(),
        sparse.COO.from_numpy(h1e),
        0.5 * sparse.COO.from_numpy(eri),
    ]
    constructor = (
        ParticleNumberAndSpinConservingFermioperator2nd._from_sparse_arrays_normal_order_all_sectors
    )
    return constructor(
        hilbert,
        arrays,
        cutoff=cutoff,
    )
