from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from time import perf_counter

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "qcnqs-matplotlib"))

import numpy as np
from common import load_problem
from netket_integrals import from_pyscf_integrals
from scipy.sparse.linalg import eigsh


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate the NetKet benchmark operator against PySCF FCI."
    )
    parser.add_argument("fcidump", type=Path)
    parser.add_argument("--energy-tolerance", type=float, default=1.0e-9)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.energy_tolerance <= 0:
        raise ValueError("energy tolerance must be positive")

    problem = load_problem(args.fcidump)
    operator = from_pyscf_integrals(
        problem.h1e,
        problem.eri,
        problem.core_energy,
        (problem.n_alpha, problem.n_beta),
    )
    start = perf_counter()
    sparse_matrix = operator.to_sparse()
    lowest = float(
        eigsh(
            sparse_matrix,
            k=1,
            which="SA",
            return_eigenvectors=False,
            tol=1.0e-11,
            v0=np.ones(sparse_matrix.shape[0], dtype=np.float64),
        )[0].real
    )
    elapsed = perf_counter() - start
    error = abs(lowest - problem.fci_energy)
    passed = error <= args.energy_tolerance
    result: dict[str, object] = {
        "problem": problem.path.name,
        "matrix_shape": list(sparse_matrix.shape),
        "nonzero_matrix_elements": int(sparse_matrix.nnz),
        "lowest_eigenvalue_ha": lowest,
        "pyscf_fci_energy_ha": problem.fci_energy,
        "absolute_energy_error_ha": error,
        "elapsed_seconds": elapsed,
        "passed": passed,
    }
    serialized = json.dumps(result, indent=2) + "\n"
    print(serialized, end="")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized)
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
