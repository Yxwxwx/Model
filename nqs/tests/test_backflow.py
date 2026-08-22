from __future__ import annotations

import torch

from qcnqs.autodiff import log_derivative_matrix
from qcnqs.models import DeterminantBackflow


def _states() -> torch.Tensor:
    return torch.tensor(
        [
            [1, 1, 0, 0, 0, 0],
            [1, 0, 0, 1, 0, 0],
            [0, 0, 1, 1, 0, 0],
        ],
        dtype=torch.bool,
    )


def test_backflow_matches_manual_determinant() -> None:
    torch.manual_seed(11)
    model = DeterminantBackflow(6, 2, 4)
    states = _states()
    encoded = states.to(torch.float64)
    hidden = torch.tanh(encoded @ model.input_weight.T + model.input_bias)
    matrices = model.base_orbitals + (hidden @ model.output_weight.T).reshape(3, 6, 2)
    occupied = model.occupied_indices(states)
    selected = matrices.gather(1, occupied[..., None].expand(-1, -1, 2))
    expected_sign, expected_logabs = torch.linalg.slogdet(selected)
    actual = model.log_amplitude(states)
    torch.testing.assert_close(actual.logabs, expected_logabs)
    torch.testing.assert_close(actual.sign, expected_sign)


def test_n2_parameter_count() -> None:
    model = DeterminantBackflow(20, 14, 20)
    assert model.parameter_count == 6_300


def test_vectorized_log_derivatives_match_autograd_loop() -> None:
    torch.manual_seed(13)
    model = DeterminantBackflow(6, 2, 4, initialization="normal")
    states = _states()
    actual = log_derivative_matrix(model, states)
    expected_rows = []
    parameters = tuple(model.parameters())
    for state in states:
        value = model(state)
        gradients = torch.autograd.grad(value, parameters)
        expected_rows.append(torch.cat([gradient.reshape(-1) for gradient in gradients]))
    expected = torch.stack(expected_rows)
    torch.testing.assert_close(actual, expected, atol=1.0e-10, rtol=1.0e-10)
