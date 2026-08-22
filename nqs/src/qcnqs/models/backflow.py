"""Single-determinant neural Backflow implemented with dense PyTorch kernels.

For occupation vector ``n`` the network constructs a state-dependent orbital
matrix

``Phi(n) = Phi_0 + reshape(W_out * activation(W_in n + b))``

with shape ``[n_spin_orbitals, n_electrons]``.  Selecting occupied rows gives a
square matrix whose determinant is the real wave-function amplitude.  Batched
``torch.linalg.slogdet`` provides a stable sign/log-magnitude representation.

The forward path contains no sample-wise Python loop.  Static orbital indices
are registered as a non-persistent buffer so repeated MCMC evaluations do not
allocate ``arange`` tensors and model/device transfers remain automatic.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn
from torch.nn import functional as functional

from qcnqs.wavefunction import LogAmplitude

_ACTIVATIONS: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    "tanh": torch.tanh,
    "gelu": functional.gelu,
    "relu": functional.relu,
    "silu": functional.silu,
}


class DeterminantBackflow(nn.Module):
    """One-hidden-layer Backflow wave function with one Slater determinant.

    Parameter tensors use the orientation expected by ``torch.nn.functional``:
    input weights are ``[hidden, spin_orbitals]`` and output weights are
    ``[spin_orbitals * electrons, hidden]``.  This keeps the two network stages
    on optimized GEMM paths for both CPU and CUDA.
    """

    def __init__(
        self,
        n_spin_orbitals: int,
        n_electrons: int,
        hidden_units: int,
        *,
        activation: str = "tanh",
        input_encoding: str = "occupation01",
        initialization: str = "orthogonal",
        neural_scale: float = 1.0,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__()
        if not 0 < n_electrons <= n_spin_orbitals:
            raise ValueError("n_electrons must lie inside the spin-orbital space")
        if hidden_units <= 0:
            raise ValueError("hidden_units must be positive")
        if neural_scale < 0:
            raise ValueError("neural_scale must be non-negative")
        if activation not in {"tanh", "gelu", "relu", "silu"}:
            raise ValueError(f"unsupported activation: {activation}")
        if input_encoding not in {"occupation01", "spin-pm1"}:
            raise ValueError(f"unsupported input encoding: {input_encoding}")
        if initialization not in {"orthogonal", "normal", "pytorch"}:
            raise ValueError(f"unsupported initialization: {initialization}")
        factory = {"device": device, "dtype": dtype}
        self.n_spin_orbitals = int(n_spin_orbitals)
        self.n_electrons = int(n_electrons)
        self.hidden_units = int(hidden_units)
        self.activation_name = activation
        self.input_encoding = input_encoding
        self.initialization = initialization
        self.neural_scale = float(neural_scale)
        self.base_orbitals = nn.Parameter(torch.empty((n_spin_orbitals, n_electrons), **factory))
        self.input_weight = nn.Parameter(torch.empty((hidden_units, n_spin_orbitals), **factory))
        self.input_bias = nn.Parameter(torch.empty(hidden_units, **factory))
        self.output_weight = nn.Parameter(
            torch.empty((n_spin_orbitals * n_electrons, hidden_units), **factory)
        )
        self.register_buffer(
            "_orbital_indices",
            torch.arange(n_spin_orbitals, device=device),
            persistent=False,
        )
        self.reset_parameters()

    @property
    def parameter_count(self) -> int:
        """Return the exact number of trainable scalar parameters."""

        return sum(parameter.numel() for parameter in self.parameters())

    def reset_parameters(self) -> None:
        """Initialize every matrix according to the selected reproducible policy."""

        with torch.no_grad():
            if self.initialization == "pytorch":
                input_bound = 0.1 / self.n_spin_orbitals**0.5
                output_bound = 0.1 / self.hidden_units**0.5
                self.input_weight.uniform_(-input_bound, input_bound)
                self.input_bias.uniform_(-input_bound, input_bound)
                self.output_weight.uniform_(
                    -output_bound * self.neural_scale,
                    output_bound * self.neural_scale,
                )
                self.base_orbitals.uniform_(-output_bound, output_bound)
                return
            if self.initialization == "orthogonal":
                nn.init.orthogonal_(self.base_orbitals)
                nn.init.orthogonal_(self.input_weight)
                nn.init.orthogonal_(self.output_weight)
            else:
                nn.init.normal_(self.base_orbitals, std=self.n_spin_orbitals**-0.5)
                nn.init.normal_(self.input_weight, std=self.n_spin_orbitals**-0.5)
                nn.init.normal_(self.output_weight, std=self.hidden_units**-0.5)
            self.input_bias.zero_()
            self.output_weight.mul_(self.neural_scale)

    def _activation(self) -> Callable[[torch.Tensor], torch.Tensor]:
        return _ACTIVATIONS[self.activation_name]

    def _encode(self, states: torch.Tensor) -> torch.Tensor:
        values = states.to(device=self.base_orbitals.device, dtype=self.base_orbitals.dtype)
        if self.input_encoding == "spin-pm1":
            values = 2.0 * values - 1.0
        return values

    def occupied_indices(self, states: torch.Tensor) -> torch.Tensor:
        """Return sorted occupied rows with a fixed, compile-friendly shape.

        ``torch.nonzero`` determines its output size dynamically and can force a
        CUDA synchronization.  Sorting sentinel-filled fixed-width rows avoids
        that synchronization and is faster for the small orbital dimensions of
        determinant Backflow.
        """

        occupied = torch.where(states, self._orbital_indices, self.n_spin_orbitals)
        return occupied.sort(dim=-1).values[..., : self.n_electrons]

    def orbital_matrices(self, states: torch.Tensor) -> torch.Tensor:
        """Build one full state-dependent orbital matrix per input state."""

        encoded = self._encode(states)
        hidden = self._activation()(functional.linear(encoded, self.input_weight, self.input_bias))
        correction = functional.linear(hidden, self.output_weight)
        correction = correction.reshape(
            *states.shape[:-1],
            self.n_spin_orbitals,
            self.n_electrons,
        )
        return self.base_orbitals + correction

    def selected_matrices(self, states: torch.Tensor) -> torch.Tensor:
        """Gather occupied rows into square determinant matrices."""

        if states.shape[-1] != self.n_spin_orbitals:
            raise ValueError("state has the wrong number of spin orbitals")
        if states.ndim == 1:
            states = states.unsqueeze(0)
        states = states.to(self.base_orbitals.device).bool()
        matrices = self.orbital_matrices(states)
        occupied = self.occupied_indices(states)
        row_index = occupied.unsqueeze(-1).expand(-1, -1, self.n_electrons)
        return matrices.gather(dim=-2, index=row_index)

    def log_amplitude(self, states: torch.Tensor) -> LogAmplitude:
        """Return stable real determinant signs and logarithmic magnitudes."""

        single = states.ndim == 1
        selected = self.selected_matrices(states)
        sign, logabs = torch.linalg.slogdet(selected)
        if single:
            sign = sign.squeeze(0)
            logabs = logabs.squeeze(0)
        # The sign is piecewise constant away from determinant nodes and should
        # not participate in automatic differentiation of log|psi|.
        return LogAmplitude(logabs=logabs, sign=sign.detach())

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """Return log|psi| so torch.func can differentiate the module."""

        return self.log_amplitude(states).logabs
