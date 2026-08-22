"""Validated molecular integral data and PySCF/FCIDUMP adapters.

This module is an input boundary.  It normalizes external chemistry data to
CPU FP64 tensors in PySCF chemist notation ``(pq|rs)`` and records conserved
particle/symmetry metadata.  It does not construct a Hilbert space, operator,
sampler, or model; the composition layer performs those choices explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from pyscf import ao2mo, fci, symm
from pyscf.tools import fcidump

if TYPE_CHECKING:
    from pyscf.scf.hf import RHF


def _zero_based_irreps(values: list[int] | tuple[int, ...]) -> tuple[int, ...]:
    result = tuple(int(value) for value in values)
    if result and min(result) >= 1:
        result = tuple(value - 1 for value in result)
    return result


@dataclass(frozen=True)
class MolecularProblem:
    """Spatial-orbital integrals and the conserved electronic sector.

    ``eri[p, q, r, s]`` uses PySCF's chemist notation ``(pq|rs)``.
    Orbital and target irreps are zero-based abelian XOR labels, matching the
    convention used internally by the C++ reference implementation.
    """

    h1e: torch.Tensor
    eri: torch.Tensor
    core_energy: float
    n_alpha: int
    n_beta: int
    orbital_symmetries: tuple[int, ...]
    target_symmetry: int = 0
    name: str = "molecule"

    def __post_init__(self) -> None:
        h1e = torch.as_tensor(self.h1e, dtype=torch.float64, device="cpu").contiguous()
        eri = torch.as_tensor(self.eri, dtype=torch.float64, device="cpu").contiguous()
        if h1e.ndim != 2 or h1e.shape[0] != h1e.shape[1]:
            raise ValueError("h1e must be a square matrix")
        n_orbitals = h1e.shape[0]
        if eri.shape != (n_orbitals,) * 4:
            raise ValueError("eri must have shape [n_orbitals] * 4")
        if not 0 <= self.n_alpha <= n_orbitals:
            raise ValueError("n_alpha is outside the orbital space")
        if not 0 <= self.n_beta <= n_orbitals:
            raise ValueError("n_beta is outside the orbital space")
        if len(self.orbital_symmetries) != n_orbitals:
            raise ValueError("one orbital symmetry label is required per spatial orbital")
        if min(self.orbital_symmetries, default=0) < 0 or self.target_symmetry < 0:
            raise ValueError("symmetry labels must be non-negative")
        object.__setattr__(self, "h1e", h1e)
        object.__setattr__(self, "eri", eri)
        object.__setattr__(
            self, "orbital_symmetries", tuple(int(x) for x in self.orbital_symmetries)
        )

    @property
    def n_orbitals(self) -> int:
        return self.h1e.shape[0]

    @property
    def n_spin_orbitals(self) -> int:
        return 2 * self.n_orbitals

    @property
    def n_electrons(self) -> int:
        return self.n_alpha + self.n_beta

    @classmethod
    def from_fcidump(cls, path: str | Path, *, name: str | None = None) -> MolecularProblem:
        """Load a restricted-orbital FCIDUMP and normalize symmetry labels.

        FCIDUMP uses one-based Molpro-style abelian irrep labels, whereas the
        execution layer uses zero-based XOR labels.  ``MS2`` determines the
        separate alpha/beta counts without assuming a singlet.
        """

        path = Path(path)
        data = fcidump.read(str(path), verbose=False)
        n_orbitals = int(data["NORB"])
        n_electrons = int(data["NELEC"])
        ms2 = int(data.get("MS2", data.get("MS", 0)))
        if (n_electrons + ms2) % 2:
            raise ValueError("FCIDUMP NELEC and MS2 have incompatible parity")
        n_alpha = (n_electrons + ms2) // 2
        n_beta = n_electrons - n_alpha
        orbital_symmetries = _zero_based_irreps(data.get("ORBSYM", [1] * n_orbitals))
        target = int(data.get("ISYM", 1))
        if target >= 1:
            target -= 1
        return cls(
            h1e=torch.from_numpy(np.asarray(data["H1"], dtype=np.float64)),
            eri=torch.from_numpy(
                np.asarray(ao2mo.restore(1, data["H2"], n_orbitals), dtype=np.float64)
            ),
            core_energy=float(data.get("ECORE", 0.0)),
            n_alpha=n_alpha,
            n_beta=n_beta,
            orbital_symmetries=orbital_symmetries,
            target_symmetry=target,
            name=name or path.stem,
        )

    @classmethod
    def from_pyscf_rhf(
        cls,
        mean_field: RHF,
        *,
        target_irrep: str | int = "Ag",
        name: str = "molecule",
    ) -> MolecularProblem:
        """Transform AO integrals and symmetry labels from a converged RHF object."""

        molecule = mean_field.mol
        coefficients = np.asarray(mean_field.mo_coeff)
        h1e = coefficients.T @ mean_field.get_hcore() @ coefficients
        eri = ao2mo.restore(
            1,
            ao2mo.kernel(molecule, coefficients),
            coefficients.shape[1],
        )
        if molecule.symmetry:
            labels = symm.label_orb_symm(
                molecule,
                molecule.irrep_id,
                molecule.symm_orb,
                coefficients,
            )
            orbital_symmetries = tuple(int(label) for label in labels)
            target_symmetry = (
                int(target_irrep)
                if isinstance(target_irrep, int)
                else int(symm.irrep_name2id(molecule.groupname, target_irrep))
            )
        else:
            orbital_symmetries = (0,) * coefficients.shape[1]
            target_symmetry = 0
        n_electrons = int(molecule.nelectron)
        n_alpha = (n_electrons + int(molecule.spin)) // 2
        return cls(
            h1e=torch.from_numpy(np.asarray(h1e, dtype=np.float64)),
            eri=torch.from_numpy(np.asarray(eri, dtype=np.float64)),
            core_energy=float(molecule.energy_nuc()),
            n_alpha=n_alpha,
            n_beta=n_electrons - n_alpha,
            orbital_symmetries=orbital_symmetries,
            target_symmetry=target_symmetry,
            name=name,
        )


def fci_reference_energy(problem: MolecularProblem, **kwargs: Any) -> float:
    """Return an independent fixed-spin PySCF FCI energy for validation."""

    options: dict[str, Any] = {"tol": 1.0e-12, "max_cycle": 200}
    options.update(kwargs)
    energy, _ = fci.direct_spin1.kernel(
        problem.h1e.numpy(),
        problem.eri.numpy(),
        problem.n_orbitals,
        (problem.n_alpha, problem.n_beta),
        ecore=problem.core_energy,
        **options,
    )
    return float(energy)
