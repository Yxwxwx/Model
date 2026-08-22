"""Public qcnqs API with lazy imports across independent implementation layers.

Importing ``qcnqs`` itself loads only the version and this small dispatch table.
Concrete chemistry, operator, model, sampler, and driver modules are imported
when their public symbol is first accessed.  Besides reducing startup cost,
this preserves runtime decoupling: importing ``qcnqs.geometry`` does not
implicitly import PySCF or the QC Hamiltonian through package initialization.
"""

from __future__ import annotations

from importlib import import_module

from qcnqs._version import __version__

_LAZY_EXPORTS = {
    "ConnectedOperator": ("qcnqs.interfaces", "ConnectedOperator"),
    "Connections": ("qcnqs.interfaces", "Connections"),
    "DeterminantBackflow": ("qcnqs.models", "DeterminantBackflow"),
    "FermionHilbert": ("qcnqs.hilbert", "FermionHilbert"),
    "FermionSampler": ("qcnqs.sampler", "FermionSampler"),
    "Hubbard1D": ("qcnqs.hubbard", "Hubbard1D"),
    "Hubbard2D": ("qcnqs.hubbard", "Hubbard2D"),
    "HubbardHamiltonian": ("qcnqs.hubbard", "HubbardHamiltonian"),
    "LogAmplitude": ("qcnqs.wavefunction", "LogAmplitude"),
    "LogWaveFunction": ("qcnqs.wavefunction", "LogWaveFunction"),
    "MolecularProblem": ("qcnqs.chemistry", "MolecularProblem"),
    "QCHamiltonian": ("qcnqs.hamiltonian", "QCHamiltonian"),
    "QGTSolveResult": ("qcnqs.geometry", "QGTSolveResult"),
    "SampleBatch": ("qcnqs.interfaces", "SampleBatch"),
    "SampledQGT": ("qcnqs.geometry", "SampledQGT"),
    "SamplerStats": ("qcnqs.interfaces", "SamplerStats"),
    "SectorBasis": ("qcnqs.hilbert", "SectorBasis"),
    "StateSpace": ("qcnqs.interfaces", "StateSpace"),
    "VMCDriver": ("qcnqs.vmc", "VMCDriver"),
    "VariationalSampler": ("qcnqs.interfaces", "VariationalSampler"),
    "energy_sr_direction": ("qcnqs.geometry", "energy_sr_direction"),
    "dense_operator_matrix": ("qcnqs.operator_tools", "dense_operator_matrix"),
    "exact_model_expectation": ("qcnqs.vmc", "exact_model_expectation"),
    "fci_reference_energy": ("qcnqs.chemistry", "fci_reference_energy"),
    "hilbert_from_problem": ("qcnqs.assembly", "hilbert_from_problem"),
    "make_n2_sto3g": ("qcnqs.chemistry", "make_n2_sto3g"),
    "nearest_neighbor_bonds_1d": ("qcnqs.hubbard", "nearest_neighbor_bonds_1d"),
    "nearest_neighbor_bonds_2d": ("qcnqs.hubbard", "nearest_neighbor_bonds_2d"),
}

__all__ = [*_LAZY_EXPORTS, "__version__"]


def __getattr__(name: str) -> object:
    """Resolve and cache one public symbol on first access."""

    try:
        module_name, attribute_name = _LAZY_EXPORTS[name]
    except KeyError as error:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from error
    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Expose lazy public symbols to interactive discovery and IDE tooling."""

    return sorted({*globals(), *__all__})
