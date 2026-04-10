r"""Tests for linear innovation state updaters."""

import pytest
import torch
from torch import nn

from linodenet.nn.rezero import ReZero
from linodenet.state_update import LinearInnovationCell


def test_linear_innovation_cell_identity_gate_matches_plain_update() -> None:
    r"""The identity gate should preserve the plain innovation update."""
    cell = LinearInnovationCell(3, 5, gate="identity")
    y = torch.randn(7, 3)
    x = torch.randn(7, 5)

    assert isinstance(cell.gain, nn.Linear)
    assert isinstance(cell.observation_map, nn.Linear)
    innovation = y - cell.observation_map(x)
    expected = x - cell.gain(innovation)

    torch.testing.assert_close(cell(y, x), expected)


def test_linear_innovation_cell_rezero_starts_as_identity() -> None:
    r"""ReZero mode should initialize the innovation path at zero."""
    cell = LinearInnovationCell(3, 5)
    y = torch.randn(7, 3)
    x = torch.randn(7, 5)

    assert isinstance(cell.gate, ReZero)
    assert cell.gate.scalar.requires_grad is True
    torch.testing.assert_close(cell(y, x), x)


def test_linear_innovation_cell_rezero_scalar_controls_correction() -> None:
    r"""Setting the ReZero scalar to one should recover the plain innovation update."""
    plain = LinearInnovationCell(3, 5, gate="identity")
    rezero = LinearInnovationCell(3, 5)
    y = torch.randn(7, 3)
    x = torch.randn(7, 5)

    with torch.no_grad():
        rezero.gain.weight.copy_(plain.gain.weight)
        assert isinstance(plain.observation_map, nn.Linear)
        assert isinstance(rezero.observation_map, nn.Linear)
        rezero.observation_map.weight.copy_(plain.observation_map.weight)
        assert isinstance(rezero.gate, ReZero)
        rezero.gate.scalar.copy_(torch.tensor(1.0))

    torch.testing.assert_close(rezero(y, x), plain(y, x))


def test_linear_innovation_cell_identity_observation_map_uses_x_directly() -> None:
    r"""Identity observation maps should use the hidden state directly."""
    cell = LinearInnovationCell(4, 4, observation_map="identity", gate="identity")
    y = torch.randn(7, 4)
    x = torch.randn(7, 4)

    assert isinstance(cell.observation_map, nn.Identity)
    innovation = y - x

    torch.testing.assert_close(cell(y, x), x - cell.gain(innovation))


def test_linear_innovation_cell_accepts_custom_observation_map() -> None:
    r"""Custom observation maps should be used verbatim."""
    observation_map = nn.Linear(5, 3, bias=False)
    cell = LinearInnovationCell(3, 5, observation_map=observation_map)

    assert cell.observation_map is observation_map


def test_linear_innovation_cell_accepts_custom_gate() -> None:
    r"""Custom gates should be used verbatim."""
    gate = nn.Tanh()
    cell = LinearInnovationCell(3, 5, gate=gate)

    assert cell.gate is gate


def test_linear_innovation_cell_none_gate_maps_to_identity() -> None:
    r"""A None gate should behave like the identity gate."""
    none_gate = LinearInnovationCell(3, 5, gate=None)
    identity_gate = LinearInnovationCell(3, 5, gate="identity")
    y = torch.randn(7, 3)
    x = torch.randn(7, 5)

    with torch.no_grad():
        none_gate.gain.weight.copy_(identity_gate.gain.weight)
        assert isinstance(none_gate.observation_map, nn.Linear)
        assert isinstance(identity_gate.observation_map, nn.Linear)
        none_gate.observation_map.weight.copy_(identity_gate.observation_map.weight)

    assert isinstance(none_gate.gate, nn.Identity)
    torch.testing.assert_close(none_gate(y, x), identity_gate(y, x))


def test_linear_innovation_cell_rejects_identity_for_nonsquare_shapes() -> None:
    r"""Identity observation maps require matching input and hidden sizes."""
    with pytest.raises(
        ValueError,
        match=r"observation_map='identity' requires input_size == hidden_size!",
    ):
        LinearInnovationCell(3, 5, observation_map="identity")


def test_linear_innovation_cell_rejects_unknown_gate() -> None:
    r"""Unknown gate strings should fail explicitly."""
    with pytest.raises(
        ValueError,
        match=r"Unknown gate: 'other'. Expected 'rezero', 'identity', None, or an nn.Module.",
    ):
        LinearInnovationCell(3, 5, gate="other")
