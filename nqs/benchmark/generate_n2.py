from __future__ import annotations

import argparse
from pathlib import Path

from pyscf import gto, scf
from pyscf.tools import fcidump


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the matched N2/STO-3G FCIDUMP.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "data" / "N2.STO3G.RHF.FCIDUMP",
    )
    args = parser.parse_args()
    molecule = gto.M(
        atom="N 0 0 0; N 0 0 1.1",
        basis="sto-3g",
        unit="Angstrom",
        symmetry="d2h",
        spin=0,
        charge=0,
        verbose=0,
    )
    mean_field = scf.RHF(molecule).run(conv_tol=1.0e-14)
    if not mean_field.converged:
        raise RuntimeError("N2 RHF calculation did not converge")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fcidump.from_scf(mean_field, str(args.output), tol=1.0e-15)
    print(args.output)


if __name__ == "__main__":
    main()
