"""Editable qcnqs calculation script.

qcnqs is a library, so this file is intentionally ordinary Python rather than
a command-line interface.  Select a system and change numerical settings in
the single ``PARAMETERS`` dictionary, then run ``uv run python main.py``.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from qcnqs import (
    DeterminantBackflow,
    FermionSampler,
    Hubbard1D,
    Hubbard2D,
    MolecularProblem,
    QCHamiltonian,
    VMCDriver,
    exact_model_expectation,
    fci_reference_energy,
    hilbert_from_problem,
    make_n2_sto3g,
)

# This is the only place a normal calculation needs to be configured.  Change
# ``system.kind`` to ``hubbard_1d`` or ``hubbard_2d`` to use the corresponding
# subsection; inactive system subsections are ignored.
PARAMETERS: dict[str, Any] = {
    "system": {
        "kind": "n2",
        "n2": {
            "fcidump": "benchmark/data/N2.STO3G.RHF.FCIDUMP",
            "sector": "point-group",  # "point-group" or "full"
        },
        "hubbard_1d": {
            "sites": 8,
            "n_alpha": 4,
            "n_beta": 4,
            "hopping": 1.0,
            "interaction": 4.0,
            "periodic": True,
        },
        "hubbard_2d": {
            "rows": 2,
            "columns": 4,
            "n_alpha": 4,
            "n_beta": 4,
            "hopping": 1.0,
            "interaction": 4.0,
            "periodic": True,
        },
    },
    "model": {
        "hidden_units": 20,
        "activation": "tanh",
        "input_encoding": "occupation01",
        "initialization": "orthogonal",
    },
    "sampling": {
        "mode": "metropolis",  # use "exact" only for a small enumerated sector
        "samples": 4096,
        "chains": 1024,
        "burn_in_sweeps": 16,
        "sweep_size": None,
        "sample_interval": 1,
        "single_probability": 0.05,
        "compile_transitions": False,
    },
    "optimization": {
        "iterations": 300,
        "learning_rate": 0.05,
        "diagonal_shift": 0.05,
        "qgt": "auto",  # "auto", "dense", or "minsr"
    },
    "diagnostics": {
        "reference_energy": True,
        "exact_model_expectation": True,
        "max_dense_states": 4096,
        "max_exact_model_states": 100_000,
        "exact_batch_size": 2048,
    },
    "runtime": {
        "device": "auto",
        "seed": 3,
        "log_every": 10,
        "output": None,  # for example "benchmark/results/my_run.json"
    },
}


def _execution_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def _build_system(
    system: Mapping[str, Any],
    device: torch.device,
) -> tuple[str, Any, Any, MolecularProblem | None]:
    kind = str(system["kind"])
    if kind == "n2":
        settings = system["n2"]
        fcidump = settings.get("fcidump")
        problem = (
            MolecularProblem.from_fcidump(fcidump, name="N2_STO3G_FCIDUMP")
            if fcidump is not None
            else make_n2_sto3g()
        )
        sector = str(settings["sector"])
        if sector not in {"full", "point-group"}:
            raise ValueError("N2 sector must be 'full' or 'point-group'")
        space = hilbert_from_problem(problem, point_group=sector == "point-group")
        return f"{problem.name} ({sector})", space, QCHamiltonian(problem, device=device), problem

    if kind == "hubbard_1d":
        settings = system["hubbard_1d"]
        operator = Hubbard1D(
            settings["sites"],
            settings["n_alpha"],
            settings["n_beta"],
            hopping=settings["hopping"],
            interaction=settings["interaction"],
            periodic=settings["periodic"],
            device=device,
        )
        name = f"Hubbard1D(L={operator.length}, periodic={operator.periodic})"
        return name, operator.space, operator, None

    if kind == "hubbard_2d":
        settings = system["hubbard_2d"]
        operator = Hubbard2D(
            settings["rows"],
            settings["columns"],
            settings["n_alpha"],
            settings["n_beta"],
            hopping=settings["hopping"],
            interaction=settings["interaction"],
            periodic=settings["periodic"],
            device=device,
        )
        name = f"Hubbard2D({operator.rows}x{operator.columns}, periodic={operator.periodic})"
        return name, operator.space, operator, None

    raise ValueError("system.kind must be 'n2', 'hubbard_1d', or 'hubbard_2d'")


def _reference_energy(
    operator: Any,
    space: Any,
    problem: MolecularProblem | None,
    diagnostics: Mapping[str, Any],
) -> float | None:
    if not diagnostics["reference_energy"]:
        return None
    if problem is not None:
        return fci_reference_energy(problem)
    if space.sector_size() > diagnostics["max_dense_states"]:
        print(
            "Skipping exact ground-state reference: "
            f"{space.sector_size():,} states exceeds max_dense_states.",
            flush=True,
        )
        return None
    basis = space.enumerate(operator.device)
    matrix = operator.dense_matrix(basis, max_states=diagnostics["max_dense_states"])
    return float(torch.linalg.eigvalsh(matrix)[0].item())


def run(parameters: Mapping[str, Any] = PARAMETERS) -> dict[str, Any]:
    """Compose and run one library calculation from ``parameters``."""

    runtime = parameters["runtime"]
    sampling = parameters["sampling"]
    optimization = parameters["optimization"]
    diagnostics = parameters["diagnostics"]
    device = _execution_device(runtime["device"])
    seed = int(runtime["seed"])
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    name, space, operator, problem = _build_system(parameters["system"], device)
    model_settings = parameters["model"]
    model = DeterminantBackflow(
        space.n_spin_orbitals,
        space.n_electrons,
        model_settings["hidden_units"],
        activation=model_settings["activation"],
        input_encoding=model_settings["input_encoding"],
        initialization=model_settings["initialization"],
        device=device,
        dtype=torch.float64,
    )
    sampler = FermionSampler(
        space,
        model,
        n_samples=sampling["samples"],
        n_chains=sampling["chains"],
        seed=seed,
        device=device,
        mode=sampling["mode"],
        burn_in_sweeps=sampling["burn_in_sweeps"],
        sweep_size=sampling["sweep_size"],
        sample_interval=sampling["sample_interval"],
        single_probability=sampling["single_probability"],
        compile_transitions=sampling["compile_transitions"],
    )
    driver = VMCDriver(
        operator,
        space,
        model,
        sampler,
        learning_rate=optimization["learning_rate"],
        diagonal_shift=optimization["diagonal_shift"],
        qgt_method=optimization["qgt"],
    )
    reference = _reference_energy(operator, space, problem, diagnostics)
    print(
        f"{name}: {space.n_spin_orbitals} spin orbitals, "
        f"({space.n_alpha} alpha, {space.n_beta} beta), "
        f"{space.sector_size():,} states on {device}",
        flush=True,
    )
    print(
        f"Backflow H={model_settings['hidden_units']}, parameters={model.parameter_count:,}"
        + (f", exact ground energy={reference:.12f}" if reference is not None else ""),
        flush=True,
    )

    log_every = int(runtime["log_every"])

    def log_step(step: Any) -> None:
        if log_every > 0 and step.iteration % log_every == 0:
            print(
                f"step {step.iteration:4d} | E={step.energy: .12f} "
                f"+/- {step.standard_error:.3e} | Var={step.variance:.3e} | "
                f"accept={step.acceptance_rate:.3f} | "
                f"{step.timings.total_seconds * 1e3:.2f} ms",
                flush=True,
            )

    result = driver.run(
        int(optimization["iterations"]),
        final_estimate=True,
        callback=log_step,
    )
    terminal = result.final_estimate
    assert terminal is not None
    exact_energy = None
    if diagnostics["exact_model_expectation"]:
        if space.sector_size() <= diagnostics["max_exact_model_states"]:
            exact_energy = exact_model_expectation(
                operator,
                model,
                space.enumerate(device),
                batch_size=diagnostics["exact_batch_size"],
            )
        else:
            print(
                "Skipping exact model expectation: "
                f"{space.sector_size():,} states exceeds max_exact_model_states.",
                flush=True,
            )

    record: dict[str, Any] = {
        "implementation": "qcnqs",
        "problem": name,
        "system": name,
        "sector": parameters["system"].get("n2", {}).get("sector", "full")
        if parameters["system"]["kind"] == "n2"
        else "full",
        "device": str(device),
        "dtype": "float64",
        "seed": seed,
        "sector_size": space.sector_size(),
        "spin_orbitals": space.n_spin_orbitals,
        "electrons": {"alpha": space.n_alpha, "beta": space.n_beta},
        "proposal": "singles"
        if sampling["single_probability"] == 1.0
        else "doubles"
        if sampling["single_probability"] == 0.0
        else "singles-doubles",
        "single_probability": float(sampling["single_probability"]),
        "hidden_units": int(model_settings["hidden_units"]),
        "parameter_count": model.parameter_count,
        "initialization": model_settings["initialization"],
        "iterations": int(optimization["iterations"]),
        "samples": int(sampling["samples"]),
        "chains": int(sampling["chains"]),
        "discard_sweeps": int(sampling["burn_in_sweeps"]),
        "sweep_size": sampler.sweep_size,
        "learning_rate": float(optimization["learning_rate"]),
        "diagonal_shift": float(optimization["diagonal_shift"]),
        "training_seconds": result.training_seconds,
        "reference_energy": reference,
        "fci_energy_ha": reference,
        "sampled_energy": terminal.mean,
        "sampled_energy_ha": terminal.mean,
        "sampled_standard_error": terminal.standard_error,
        "sampled_standard_error_ha": terminal.standard_error,
        "sampled_variance": terminal.variance,
        "sampled_variance_ha2": terminal.variance,
        "sampled_absolute_error_mha": None
        if reference is None
        else 1_000.0 * abs(terminal.mean - reference),
        "exact_model_energy": exact_energy,
        "exact_model_energy_ha": exact_energy,
        "exact_model_absolute_error_mha": None
        if reference is None or exact_energy is None
        else 1_000.0 * abs(exact_energy - reference),
        "trajectory": [step.energy for step in result.steps],
        "trajectory_ha": [step.energy for step in result.steps],
        "steps": [step.as_dict() for step in result.steps],
    }
    error = None if reference is None else 1_000.0 * abs(terminal.mean - reference)
    print(
        f"terminal E={terminal.mean:.12f} +/- {terminal.standard_error:.3e}; "
        f"runtime={result.training_seconds:.3f} s"
        + (f"; |E-E0|={error:.3f} mHa" if error is not None else ""),
        flush=True,
    )
    if exact_energy is not None:
        print(f"exact model expectation={exact_energy:.12f}", flush=True)

    output = runtime["output"]
    if output is not None:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record, indent=2) + "\n")
        print(f"Wrote {path}", flush=True)
    return record


if __name__ == "__main__":
    run()
