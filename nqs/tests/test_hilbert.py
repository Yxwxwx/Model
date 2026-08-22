from __future__ import annotations

import torch

from qcnqs.hilbert import FermionHilbert, decode_states, encode_states

N2_ORBITAL_SYMMETRIES = (0, 4, 0, 4, 2, 1, 0, 5, 6, 4)


def test_state_key_round_trip() -> None:
    states = torch.tensor(
        [[1, 0, 0, 1, 1, 0], [0, 1, 1, 0, 0, 1]],
        dtype=torch.bool,
    )
    assert torch.equal(decode_states(encode_states(states), 6), states)


def test_n2_full_and_ag_sector_sizes() -> None:
    full = FermionHilbert(10, 7, 7)
    ag = FermionHilbert(
        10,
        7,
        7,
        orbital_symmetries=N2_ORBITAL_SYMMETRIES,
        target_symmetry=0,
    )
    assert full.sector_size() == 14_400
    assert ag.sector_size() == 1_824
    basis = ag.enumerate()
    assert basis.states.shape == (1_824, 20)
    assert bool(torch.all(ag.contains(basis.states)))


def test_basis_lookup_marks_states_outside_sector() -> None:
    hilbert = FermionHilbert(4, 2, 1)
    basis = hilbert.enumerate()
    indices, valid = basis.index(basis.states.flip(0))
    assert bool(torch.all(valid))
    assert torch.equal(basis.states[indices], basis.states.flip(0))
    invalid = basis.states[0].clone()
    invalid[0] = ~invalid[0]
    _, valid = basis.index(invalid)
    assert not bool(valid)


def test_vectorized_xor_symmetry_matches_scalar_reference() -> None:
    hilbert = FermionHilbert(
        5,
        2,
        2,
        orbital_symmetries=(0, 1, 2, 3, 5),
        target_symmetry=3,
    )
    generator = torch.Generator().manual_seed(47)
    states = torch.rand((4, 7, 10), generator=generator) > 0.5
    actual = hilbert.device_filter("cpu").state_symmetry(states)
    expected = torch.zeros((4, 7), dtype=torch.int64)
    for orbital, label in enumerate(hilbert.orbital_symmetries or ()):
        parity = torch.logical_xor(states[..., 2 * orbital], states[..., 2 * orbital + 1])
        expected = torch.bitwise_xor(expected, parity.to(torch.int64) * label)
    torch.testing.assert_close(actual, expected)
