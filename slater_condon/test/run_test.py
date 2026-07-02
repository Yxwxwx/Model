#!/usr/bin/env python3
"""N2 STO-3G FCI test for slater_condon / Edgerunner.

Reference energy from PySCF FCI: -107.654122447523 Ha
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BINARY = ROOT / "build" / "Edgerunner"
FCIDUMP = Path(__file__).resolve().parent / "FCIDUMP"

# PySCF reference
E_FCI_REF = -107.654122447523
TOLERANCE = 1e-6

def main():
    if not BINARY.exists():
        print(f"Binary not found: {BINARY}")
        print("Build first: cd .. && mkdir -p build && cd build && cmake .. && cmake --build .")
        sys.exit(1)

    cmd = [str(BINARY), str(FCIDUMP), "10", "14", "0", "1"]
    print(f"Running: {' '.join(cmd)}")

    result = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    if result.returncode != 0:
        print("Binary crashed:")
        print(result.stderr)
        sys.exit(1)

    # Parse output for eigenvalue
    for line in result.stdout.splitlines():
        if "Eigenvalue  1:" in line:
            energy = float(line.split()[-1])
            error = abs(energy - E_FCI_REF)
            status = "PASS" if error < TOLERANCE else "FAIL"
            print(f"\n  Reference E(FCI) = {E_FCI_REF:.12f}")
            print(f"  Edgerunner E(FCI) = {energy:.12f}")
            print(f"  |ΔE| = {error:.2e}")
            print(f"\n  {status}")
            if status == "FAIL":
                sys.exit(1)
            return

    print("Could not find eigenvalue in output")
    print(result.stdout)
    sys.exit(1)

if __name__ == "__main__":
    main()
