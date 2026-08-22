# ruff: noqa: E402

from __future__ import annotations

import argparse
import os
import platform
import sys
import tempfile
from pathlib import Path
from time import perf_counter

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "qcnqs-matplotlib"))
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import flax
import jax
from flax import nnx

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import netket as nk
import numpy as np
from common import emit_result, load_problem
from netket_integrals import from_pyscf_integrals


class MatchedDeterminantBackflow(nnx.Module):
    """NetKet/JAX form of qcnqs's 6,300-parameter H=20 Backflow."""

    def __init__(
        self,
        n_spin_orbitals: int,
        n_electrons: int,
        hidden_units: int,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        orthogonal = flax.linen.initializers.orthogonal()
        self.n_spin_orbitals = n_spin_orbitals
        self.n_electrons = n_electrons
        self.hidden_units = hidden_units
        self.base_orbitals = nnx.Param(
            orthogonal(
                rngs.params(),
                (n_spin_orbitals, n_electrons),
                np.float64,
            )
        )
        self.input_weight = nnx.Param(
            orthogonal(
                rngs.params(),
                (n_spin_orbitals, hidden_units),
                np.float64,
            )
        )
        self.input_bias = nnx.Param(jnp.zeros(hidden_units, dtype=jnp.float64))
        self.output_weight = nnx.Param(
            orthogonal(
                rngs.params(),
                (hidden_units, n_spin_orbitals * n_electrons),
                np.float64,
            )
        )

    def __call__(self, occupation: jax.Array) -> jax.Array:
        hidden = jnp.tanh(occupation @ self.input_weight[...] + self.input_bias[...])
        correction = hidden @ self.output_weight[...]
        matrices = self.base_orbitals[...] + correction.reshape(
            *occupation.shape[:-1], self.n_spin_orbitals, self.n_electrons
        )
        flat_occupation = occupation.reshape(-1, self.n_spin_orbitals)
        flat_matrices = matrices.reshape(
            -1,
            self.n_spin_orbitals,
            self.n_electrons,
        )

        def single_log_amplitude(state: jax.Array, matrix: jax.Array) -> jax.Array:
            occupied = jnp.nonzero(state, size=self.n_electrons)[0]
            sign, logabs = jnp.linalg.slogdet(matrix[occupied, :])
            phase = jnp.where(sign < 0, 1j * jnp.pi, 0j)
            return logabs.astype(jnp.complex128) + phase

        values = jax.vmap(single_log_amplitude)(flat_occupation, flat_matrices)
        return values.reshape(occupation.shape[:-1])


def spin_conserving_graph(hilbert: nk.hilbert.SpinOrbitalFermions) -> nk.graph.Graph:
    edges = []
    for spin in range(hilbert.n_spin_subsectors):
        offset = spin * hilbert.n_orbitals
        edges.extend(
            (offset + left, offset + right)
            for left in range(hilbert.n_orbitals)
            for right in range(left + 1, hilbert.n_orbitals)
        )
    return nk.graph.Graph(n_nodes=hilbert.size, edges=edges)


def main() -> None:
    parser = argparse.ArgumentParser(description="Matched NetKet N2/STO-3G benchmark.")
    parser.add_argument("fcidump", type=Path)
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--samples", type=int, default=4096)
    parser.add_argument("--chains", type=int, default=1024)
    parser.add_argument("--discard", type=int, default=16)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--hidden-units", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--diag-shift", type=float, default=0.05)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--reference-energy", type=float)
    parser.add_argument("--exact-model", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    problem = load_problem(args.fcidump, reference_energy=args.reference_energy)
    hamiltonian = from_pyscf_integrals(
        problem.h1e,
        problem.eri,
        problem.core_energy,
        (problem.n_alpha, problem.n_beta),
    )
    hilbert = hamiltonian.hilbert
    model = MatchedDeterminantBackflow(
        problem.n_spin_orbitals,
        problem.n_electrons,
        args.hidden_units,
        rngs=nnx.Rngs(args.seed),
    )
    sampler = nk.sampler.MetropolisFermionHop(
        hilbert,
        graph=spin_conserving_graph(hilbert),
        n_chains=args.chains,
        sweep_size=problem.n_spin_orbitals,
        reset_chains=False,
    )
    state = nk.vqs.MCState(
        sampler,
        model,
        n_samples=args.samples,
        n_discard_per_chain=args.discard,
        chunk_size=args.chunk_size,
        seed=args.seed,
    )
    optimizer = nk.optimizer.Sgd(learning_rate=args.learning_rate)
    driver = nk.driver.VMC_SR(
        hamiltonian,
        optimizer,
        variational_state=state,
        diag_shift=args.diag_shift,
        mode="real",
        on_the_fly=False,
    )
    runtime_log = nk.logging.RuntimeLog()
    start = perf_counter()
    driver.run(n_iter=args.iterations, out=runtime_log, show_progress=False)
    jax.block_until_ready(state.parameters)
    sampled = state.expect(hamiltonian)
    jax.block_until_ready(sampled.mean)
    training_seconds = perf_counter() - start
    exact_model_energy = None
    if args.exact_model:
        vector = np.asarray(state.to_array(normalize=True))
        sparse_hamiltonian = hamiltonian.to_sparse()
        exact_model_energy = float(np.vdot(vector, sparse_hamiltonian @ vector).real)
    energy_data = runtime_log.data["Energy"]
    trajectory = [float(value.real) for value in np.asarray(energy_data["Mean"])]
    parameter_count = sum(
        int(np.prod(leaf.shape)) for leaf in jax.tree_util.tree_leaves(nnx.state(model, nnx.Param))
    )
    result: dict[str, object] = {
        "implementation": "netket",
        "problem": problem.path.name,
        "sector": "full",
        "sector_size": problem.sector_size,
        "seed": args.seed,
        "iterations": args.iterations,
        "samples": args.samples,
        "chains": args.chains,
        "discard_sweeps": args.discard,
        "sweep_size": problem.n_spin_orbitals,
        "dtype": "float64",
        "proposal": "singles",
        "hidden_units": args.hidden_units,
        "parameter_count": parameter_count,
        "initialization": "orthogonal",
        "learning_rate": args.learning_rate,
        "diagonal_shift": args.diag_shift,
        "chunk_size": args.chunk_size,
        "training_seconds": training_seconds,
        "seconds_per_iteration": training_seconds / max(args.iterations, 1),
        "fci_energy_ha": problem.fci_energy,
        "sampled_energy_ha": float(sampled.mean.real),
        "sampled_standard_error_ha": float(sampled.error_of_mean),
        "sampled_variance_ha2": float(sampled.variance.real),
        "sampled_absolute_error_mha": 1_000.0 * abs(float(sampled.mean.real) - problem.fci_energy),
        "exact_model_energy_ha": exact_model_energy,
        "exact_model_absolute_error_mha": None
        if exact_model_energy is None
        else 1_000.0 * abs(exact_model_energy - problem.fci_energy),
        "trajectory_ha": trajectory,
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "jax": jax.__version__,
            "netket": nk.__version__,
            "backend": jax.default_backend(),
            "device": str(jax.devices()[0]),
        },
    }
    print(
        f"NetKet terminal E={result['sampled_energy_ha']:.12f} Ha; "
        f"|E-FCI|={result['sampled_absolute_error_mha']:.3f} mHa; "
        f"runtime={training_seconds:.3f} s",
        flush=True,
    )
    emit_result(result, args.output)


if __name__ == "__main__":
    main()
