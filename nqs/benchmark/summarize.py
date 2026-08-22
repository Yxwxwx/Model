from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

METRICS = (
    "training_seconds",
    "sampled_absolute_error_mha",
    "exact_model_absolute_error_mha",
)


def _label(record: dict[str, Any]) -> str:
    implementation = str(record["implementation"])
    sector = str(record.get("sector", "full"))
    proposal = str(record.get("proposal", "unknown"))
    if implementation == "qcnqs" and sector == "full" and proposal == "singles":
        return "qcnqs full (matched)"
    if implementation == "netket" and sector == "full" and proposal == "singles":
        return "NetKet full (matched)"
    if implementation == "qcnqs" and sector == "point-group":
        return "qcnqs point-group"
    return f"{implementation} {sector} ({proposal})"


def _mean_sd(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"mean": None, "sample_standard_deviation": None, "count": 0}
    return {
        "mean": statistics.fmean(values),
        "sample_standard_deviation": statistics.stdev(values) if len(values) > 1 else 0.0,
        "count": len(values),
    }


def _matched_protocol(records: list[dict[str, Any]]) -> dict[str, Any]:
    fields = (
        "sector",
        "sector_size",
        "iterations",
        "samples",
        "chains",
        "discard_sweeps",
        "sweep_size",
        "dtype",
        "proposal",
        "hidden_units",
        "parameter_count",
        "initialization",
        "learning_rate",
        "diagonal_shift",
        "fci_energy_ha",
    )
    protocol = {field: records[0].get(field) for field in fields}
    mismatches = {
        field: sorted({str(record.get(field)) for record in records})
        for field in fields
        if any(record.get(field) != records[0].get(field) for record in records[1:])
    }
    if mismatches:
        details = ", ".join(f"{key}={values}" for key, values in mismatches.items())
        raise ValueError(f"matched benchmark records use different protocols: {details}")
    return protocol


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize qcnqs/NetKet JSON benchmark records.")
    parser.add_argument("records", type=Path, nargs="+")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for path in args.records:
        record = json.loads(path.read_text())
        grouped[_label(record)].append(record)
    for records in grouped.values():
        records.sort(key=lambda record: int(record["seed"]))

    matched = [
        record
        for label, records in grouped.items()
        if label in {"qcnqs full (matched)", "NetKet full (matched)"}
        for record in records
    ]
    protocol = _matched_protocol(matched) if matched else None
    summary: dict[str, Any] = {"matched_protocol": protocol, "groups": {}}
    for label, records in grouped.items():
        summary["groups"][label] = {
            "seeds": [int(record["seed"]) for record in records],
            **{
                metric: _mean_sd(
                    [float(record[metric]) for record in records if record.get(metric) is not None]
                )
                for metric in METRICS
            },
        }

    qcnqs = summary["groups"].get("qcnqs full (matched)")
    netket = summary["groups"].get("NetKet full (matched)")
    if qcnqs is not None and netket is not None:
        if qcnqs["seeds"] != netket["seeds"]:
            raise ValueError(
                "matched qcnqs and NetKet groups must contain the same ordered seeds; "
                f"got {qcnqs['seeds']} and {netket['seeds']}"
            )
        summary["matched_end_to_end_speedup"] = (
            netket["training_seconds"]["mean"] / qcnqs["training_seconds"]["mean"]
        )

    serialized = json.dumps(summary, indent=2) + "\n"
    print(serialized, end="")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized)


if __name__ == "__main__":
    main()
