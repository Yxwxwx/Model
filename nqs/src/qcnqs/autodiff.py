"""Per-sample logarithmic derivatives built with PyTorch function transforms.

Stochastic reconfiguration needs ``O[x, k] = d log|psi(x)| / d theta_k`` for
every unique sample ``x``.  ``torch.func.grad`` differentiates one pure
functional model call and ``vmap`` applies that gradient over the sample axis.
This avoids a Python autograd loop and exposes the full Jacobian workload to
PyTorch's vectorized CPU/CUDA execution.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.func import functional_call, grad, vmap


def log_derivative_matrix(model: nn.Module, states: torch.Tensor) -> torch.Tensor:
    """Return ``d log|psi(x)| / d theta`` as ``[samples, parameters]``.

    Parameter blocks are flattened in ``named_parameters`` order.  The update
    routine uses the same order, making the dense matrix a stable interface
    between automatic differentiation and QGT solvers.  Buffers are passed
    explicitly so registered device metadata remains compatible with
    ``functional_call`` and ``torch.compile``.
    """

    parameters = dict(model.named_parameters())
    buffers = dict(model.named_buffers())
    names = tuple(parameters)

    def single_logabs(
        current_parameters: dict[str, torch.Tensor],
        current_buffers: dict[str, torch.Tensor],
        state: torch.Tensor,
    ) -> torch.Tensor:
        value = functional_call(
            model,
            (current_parameters, current_buffers),
            (state.unsqueeze(0),),
        )
        return value.squeeze(0)

    # ``randomness='error'`` makes accidental stochastic layers fail loudly;
    # a deterministic log wave function is required for a well-defined QGT.
    per_sample = vmap(
        grad(single_logabs),
        in_dims=(None, None, 0),
        randomness="error",
    )(parameters, buffers, states)
    return torch.cat(
        [per_sample[name].reshape(states.shape[0], -1) for name in names],
        dim=1,
    )
