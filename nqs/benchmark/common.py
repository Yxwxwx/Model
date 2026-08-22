from __future__ import annotations

import json
from dataclasses import dataclass
from math import comb
from pathlib import Path

import numpy as np
from pyscf import ao2mo, fci
from pyscf.tools import fcidump


@dataclass(frozen=True)
class BenchmarkProblem:
    path: Path
    n_orbitals: int
    n_alpha: int
    n_beta: int
    h1e: np.ndarray
    eri: np.ndarray
    core_energy: float
    fci_energy: float

    @property
    def n_electrons(self) -> int:
        return self.n_alpha + self.n_beta

    @property
    def n_spin_orbitals(self) -> int:
        return 2 * self.n_orbitals

    @property
    def sector_size(self) -> int:
        return comb(self.n_orbitals, self.n_alpha) * comb(self.n_orbitals, self.n_beta)


def load_problem(
    path: str | Path,
    *,
    reference_energy: float | None = None,
) -> BenchmarkProblem:
    path = Path(path).resolve()
    data = fcidump.read(str(path), verbose=False)
    n_orbitals = int(data["NORB"])
    n_electrons = int(data["NELEC"])
    ms2 = int(data.get("MS2", data.get("MS", 0)))
    n_alpha = (n_electrons + ms2) // 2
    n_beta = n_electrons - n_alpha
    h1e = np.asarray(data["H1"], dtype=np.float64)
    eri = np.asarray(
        ao2mo.restore(1, data["H2"], n_orbitals),
        dtype=np.float64,
    )
    core_energy = float(data.get("ECORE", 0.0))
    if reference_energy is None:
        reference_energy, _ = fci.direct_spin1.kernel(
            h1e,
            eri,
            n_orbitals,
            (n_alpha, n_beta),
            ecore=core_energy,
            tol=1.0e-12,
            max_cycle=200,
        )
    return BenchmarkProblem(
        path=path,
        n_orbitals=n_orbitals,
        n_alpha=n_alpha,
        n_beta=n_beta,
        h1e=h1e,
        eri=eri,
        core_energy=core_energy,
        fci_energy=float(reference_energy),
    )


def emit_result(result: dict[str, object], output: str | Path | None) -> None:
    serialized = json.dumps(result, indent=2) + "\n"
    if output is None:
        print(serialized, end="")
        return
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(serialized)
