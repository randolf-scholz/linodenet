r"""Tests for linear innovation state updaters."""

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from linodenet.nn.containers import Constant
from linodenet.nn.rezero import ReZero
from linodenet.state_update import AttentionGain, LinearCell


def compute_correction(
    r: torch.Tensor, gain: nn.Module, x: torch.Tensor
) -> torch.Tensor:
    return F.linear(r, gain(x))


class TestLinearCell:
    def test_identity_gate_matches_plain_update(self) -> None:
        r"""The identity gate should preserve the plain innovation update."""
        cell = LinearCell(3, 5, gate="identity")
        y = torch.randn(7, 3)
        x = torch.randn(7, 5)

        assert isinstance(cell.gain, Constant)
        assert isinstance(cell.observation_map, nn.Linear)
        innovation = cell.observation_map(x) - y
        expected = x - compute_correction(innovation, cell.gain, x)

        torch.testing.assert_close(cell(y, x), expected)

    def test_rezero_starts_as_identity(self) -> None:
        r"""ReZero mode should initialize the innovation path at zero."""
        cell = LinearCell(3, 5)
        y = torch.randn(7, 3)
        x = torch.randn(7, 5)

        assert isinstance(cell.gate, ReZero)
        assert cell.gate.scalar.requires_grad is True
        torch.testing.assert_close(cell(y, x), x)

    def test_rezero_scalar_controls_correction(self) -> None:
        r"""Setting the ReZero scalar to one should recover the plain innovation update."""
        plain = LinearCell(3, 5, gate="identity")
        rezero = LinearCell(3, 5)
        y = torch.randn(7, 3)
        x = torch.randn(7, 5)

        with torch.no_grad():
            assert isinstance(plain.gain, Constant)
            assert isinstance(rezero.gain, Constant)
            rezero.gain.value.copy_(plain.gain.value)
            assert isinstance(plain.observation_map, nn.Linear)
            assert isinstance(rezero.observation_map, nn.Linear)
            rezero.observation_map.weight.copy_(plain.observation_map.weight)
            assert isinstance(rezero.gate, ReZero)
            rezero.gate.scalar.copy_(torch.tensor(1.0))

        torch.testing.assert_close(rezero(y, x), plain(y, x))

    def test_identity_observation_map_uses_x_directly(self) -> None:
        r"""Identity observation maps should use the hidden state directly."""
        cell = LinearCell(4, 4, observation_map="identity", gate="identity")
        y = torch.randn(7, 4)
        x = torch.randn(7, 4)

        assert isinstance(cell.observation_map, nn.Identity)
        innovation = x - y

        torch.testing.assert_close(
            cell(y, x), x - compute_correction(innovation, cell.gain, x)
        )

    def test_from_direct_observation_model_recovers_observation(self) -> None:
        r"""The direct-observation constructor should recover the observation exactly."""
        cell = LinearCell.from_direct_observation_model(4, gate="identity")
        x = torch.randn(7, 4)
        y = torch.randn(7, 4)

        assert isinstance(cell.gain, Constant)
        assert isinstance(cell.observation_map, nn.Identity)
        torch.testing.assert_close(cell.gain.value, torch.eye(4))
        torch.testing.assert_close(cell(y, x), y)

    def test_accepts_custom_gain(self) -> None:
        r"""Custom gains should be used verbatim."""

        class DiagonalGain(nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return torch.diag(x.mean(dim=0))

        gain = DiagonalGain()
        cell = LinearCell(3, 3, gain=gain, gate="identity")
        x = torch.randn(7, 3)
        y = torch.randn(7, 3)

        assert cell.gain is gain
        innovation = cell.observation_map(x) - y

        torch.testing.assert_close(
            cell(y, x), x - compute_correction(innovation, gain, x)
        )

    def test_from_direct_observation_model_last_value_gate(self) -> None:
        r"""The last-value preset should initialize ReZero at one."""
        cell = LinearCell.from_direct_observation_model(4)
        x = torch.randn(7, 4)
        y = torch.randn(7, 4)

        assert isinstance(cell.gate, ReZero)
        torch.testing.assert_close(cell.gate.scalar, torch.tensor(1.0))
        torch.testing.assert_close(cell(y, x), y)

    def test_from_direct_observation_model_average_value_gate(self) -> None:
        r"""The average-value preset should initialize ReZero at one half."""
        cell = LinearCell.from_direct_observation_model(4, gate="average-value")
        x = torch.randn(7, 4)
        y = torch.randn(7, 4)

        assert isinstance(cell.gate, ReZero)
        torch.testing.assert_close(cell.gate.scalar, torch.tensor(0.5))
        torch.testing.assert_close(cell(y, x), 0.5 * (x + y))

    def test_from_direct_observation_model_keep_state_gate(self) -> None:
        r"""The keep-state preset should initialize ReZero at zero."""
        cell = LinearCell.from_direct_observation_model(4, gate="keep-state")
        x = torch.randn(7, 4)
        y = torch.randn(7, 4)

        assert isinstance(cell.gate, ReZero)
        torch.testing.assert_close(cell.gate.scalar, torch.tensor(0.0))
        torch.testing.assert_close(cell(y, x), x)

    def test_from_direct_observation_model_copies_only_observed_values(self) -> None:
        r"""The direct-observation constructor should overwrite only observed coordinates."""
        cell = LinearCell.from_direct_observation_model(4, gate="identity")
        x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        y = torch.tensor([[10.0, float("nan"), -5.0, float("nan")]])

        expected = torch.tensor([[10.0, 2.0, -5.0, 4.0]])

        torch.testing.assert_close(cell(y, x), expected)

    def test_from_direct_observation_model_rejects_unknown_gate(self) -> None:
        r"""Unknown direct-observation gate presets should fail explicitly."""
        with pytest.raises(
            ValueError,
            match=(
                r"Unknown direct-observation gate: 'other'. Expected "
                r"'last-value', 'average-value', 'first-value', 'keep-state', "
                r"'rezero', 'identity', None, or an nn.Module."
            ),
        ):
            LinearCell.from_direct_observation_model(4, gate="other")

    def test_attention_gain_uses_attention_module(self) -> None:
        r"""The attention gain option should instantiate `AttentionGain`."""
        cell = LinearCell(3, 5, gain="attention", gate="identity")
        x = torch.randn(7, 5)

        assert isinstance(cell.gain, AttentionGain)
        gain = cell.gain(x)

        assert gain.shape == (7, 5, 3)
        assert torch.isfinite(gain).all()
        torch.testing.assert_close(
            gain.sum(dim=-1),
            torch.ones(7, 5),
        )

    def test_accepts_custom_observation_map(self) -> None:
        r"""Custom observation maps should be used verbatim."""
        observation_map = nn.Linear(5, 3, bias=False)
        cell = LinearCell(3, 5, observation_map=observation_map)

        assert cell.observation_map is observation_map

    def test_accepts_custom_gate(self) -> None:
        r"""Custom gates should be used verbatim."""
        gate = nn.Tanh()
        cell = LinearCell(3, 5, gate=gate)

        assert cell.gate is gate

    def test_none_gate_maps_to_identity(self) -> None:
        r"""A None gate should behave like the identity gate."""
        none_gate = LinearCell(3, 5, gate=None)
        identity_gate = LinearCell(3, 5, gate="identity")
        y = torch.randn(7, 3)
        x = torch.randn(7, 5)

        with torch.no_grad():
            assert isinstance(none_gate.gain, Constant)
            assert isinstance(identity_gate.gain, Constant)
            none_gate.gain.value.copy_(identity_gate.gain.value)
            assert isinstance(none_gate.observation_map, nn.Linear)
            assert isinstance(identity_gate.observation_map, nn.Linear)
            none_gate.observation_map.weight.copy_(identity_gate.observation_map.weight)

        assert isinstance(none_gate.gate, nn.Identity)
        torch.testing.assert_close(none_gate(y, x), identity_gate(y, x))

    def test_masked_backward_has_finite_gradients(self) -> None:
        r"""Masked observations should not introduce NaNs into outputs or gradients."""
        torch.manual_seed(0)

        cell = LinearCell(5, 7, gate="identity")
        x = torch.randn(8, 7, requires_grad=True)
        y = torch.randn(8, 5)
        mask = torch.rand(8, 5) < 0.5
        y = y.masked_fill(mask, float("nan"))

        output = cell(y, x)
        loss = output.square().mean()
        loss.backward()

        assert torch.isfinite(output).all()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()

        for parameter in cell.parameters():
            assert parameter.grad is not None
            assert torch.isfinite(parameter.grad).all()

    def test_rejects_identity_for_nonsquare_shapes(self) -> None:
        r"""Identity observation maps require matching input and hidden sizes."""
        with pytest.raises(
            ValueError,
            match=r"observation_map='identity' requires input_size == hidden_size!",
        ):
            LinearCell(3, 5, observation_map="identity")

    def test_rejects_unknown_gain(self) -> None:
        r"""Unknown gain strings should fail explicitly."""
        with pytest.raises(
            ValueError,
            match=r"Unknown gain: 'other'. Expected 'constant', 'attention', or an nn.Module.",
        ):
            LinearCell(3, 5, gain="other")

    def test_rejects_unknown_gate(self) -> None:
        r"""Unknown gate strings should fail explicitly."""
        with pytest.raises(
            ValueError,
            match=r"Unknown gate: 'other'. Expected 'rezero', 'identity', None, or an nn.Module.",
        ):
            LinearCell(3, 5, gate="other")


def test_attention_gain_backward_has_finite_gradients() -> None:
    r"""Attention-based gains should support stable forward and backward passes."""
    torch.manual_seed(0)

    cell = LinearCell(4, 6, gain="attention", gate="identity")
    x = torch.randn(8, 6, requires_grad=True)
    y = torch.randn(8, 4)
    y[torch.rand(8, 4) < 0.4] = float("nan")

    output = cell(y, x)
    loss = output.square().mean()
    loss.backward()

    assert torch.isfinite(output).all()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()

    for parameter in cell.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
