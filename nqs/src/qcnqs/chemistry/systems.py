"""Small reproducible molecular systems used by examples and validation."""

from __future__ import annotations

from pyscf import gto, scf

from qcnqs.chemistry.problem import MolecularProblem


def make_n2_sto3g(
    *,
    bond_length: float = 1.1,
    unit: str = "Angstrom",
    symmetry: bool = True,
) -> MolecularProblem:
    """Build the N2/STO-3G RHF benchmark used by both reference implementations.

    The default 1.1 Angstrom D2h geometry yields 20 spin orbitals, 14 electrons,
    and an Ag target sector.  Production workflows can bypass this convenience
    function and construct :class:`MolecularProblem` from any supported input.
    """

    molecule = gto.M(
        atom=f"N 0 0 0; N 0 0 {bond_length}",
        basis="sto-3g",
        unit=unit,
        symmetry="d2h" if symmetry else False,
        spin=0,
        charge=0,
        verbose=0,
    )
    mean_field = scf.RHF(molecule).run(conv_tol=1.0e-14)
    if not mean_field.converged:
        raise RuntimeError("N2 RHF calculation did not converge")
    return MolecularProblem.from_pyscf_rhf(
        mean_field,
        target_irrep="Ag",
        name="N2_STO3G_RHF_1.1A",
    )
