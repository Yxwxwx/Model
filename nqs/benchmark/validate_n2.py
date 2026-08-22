from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import torch

from qcnqs.assembly import hilbert_from_problem
from qcnqs.chemistry import MolecularProblem, fci_reference_energy
from qcnqs.hamiltonian import QCHamiltonian


def _device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Independently validate the N2/STO-3G point-group Hamiltonian."
    )
    parser.add_argument("fcidump", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--energy-tolerance", type=float, default=1.0e-10)
    parser.add_argument("--hermiticity-tolerance", type=float, default=1.0e-12)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.energy_tolerance <= 0 or args.hermiticity_tolerance <= 0:
        raise ValueError("validation tolerances must be positive")
    device = _device(args.device)
    problem = MolecularProblem.from_fcidump(args.fcidump, name="N2_STO3G_FCIDUMP")
    full_hilbert = hilbert_from_problem(problem, point_group=False)
    point_group_hilbert = hilbert_from_problem(problem, point_group=True)
    point_group_basis = point_group_hilbert.enumerate(device)
    hamiltonian = QCHamiltonian(problem, device=device)

    start = perf_counter()
    matrix = hamiltonian.dense_matrix(point_group_basis)
    hermiticity_error = float((matrix - matrix.T).abs().max().item())
    lowest_eigenvalue = float(torch.linalg.eigvalsh(matrix)[0].item())
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = perf_counter() - start

    pyscf_fci = fci_reference_energy(problem)
    energy_error = abs(lowest_eigenvalue - pyscf_fci)
    passed = (
        energy_error <= args.energy_tolerance and hermiticity_error <= args.hermiticity_tolerance
    )
    result: dict[str, object] = {
        "problem": problem.name,
        "device": str(device),
        "n_spin_orbitals": problem.n_spin_orbitals,
        "n_alpha": problem.n_alpha,
        "n_beta": problem.n_beta,
        "connections_per_determinant": hamiltonian.n_connections,
        "full_sector_size": full_hilbert.sector_size(),
        "point_group_sector_size": point_group_hilbert.sector_size(),
        "dense_matrix_shape": list(matrix.shape),
        "hermiticity_error": hermiticity_error,
        "lowest_eigenvalue_ha": lowest_eigenvalue,
        "pyscf_fci_energy_ha": pyscf_fci,
        "absolute_energy_error_ha": energy_error,
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
