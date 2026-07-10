r"""Tests for linear innovation state updaters."""

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from linodenet.nn.rezero import ReZero
from linodenet.state_update import AttentionGain, InnovationCell
from linodenet.state_update.linear import ConstantGain


def compute_correction(
    r: torch.Tensor, gain: nn.Module, x: torch.Tensor
) -> torch.Tensor:
    return gain(r, x)


class TestInnovationCell:
    @pytest.mark.parametrize("gain_name", ["constant", "attention"])
    def test_identity_gate_matches_plain_update(self, gain_name: str) -> None:
        r"""The identity gate should preserve the plain innovation update."""
        cell = InnovationCell(3, 5, gain=gain_name, gate="identity")
        y = torch.randn(7, 3)
        x = torch.randn(7, 5)

        assert isinstance(cell.gain, (ConstantGain, AttentionGain))
        assert isinstance(cell.observation_map, nn.Linear)
        innovation = cell.observation_map(x) - y
        expected = x - compute_correction(innovation, cell.gain, x)

        torch.testing.assert_close(cell(y, x), expected)

    def test_rezero_starts_as_identity(self) -> None:
        r"""ReZero mode should initialize the innovation path at zero."""
        cell = InnovationCell(3, 5)
        y = torch.randn(7, 3)
        x = torch.randn(7, 5)

        assert isinstance(cell.gate, ReZero)
        assert cell.gate.scalar.requires_grad is True
        torch.testing.assert_close(cell(y, x), x)

    def test_rezero_scalar_controls_correction(self) -> None:
        r"""Setting the ReZero scalar to one should recover the plain innovation update."""
        plain = InnovationCell(3, 5, gate="identity")
        rezero = InnovationCell(3, 5)
        y = torch.randn(7, 3)
        x = torch.randn(7, 5)

        with torch.no_grad():
            assert isinstance(plain.gain, ConstantGain)
            assert isinstance(rezero.gain, ConstantGain)
            rezero.gain.weight.copy_(plain.gain.weight)
            assert isinstance(plain.observation_map, nn.Linear)
            assert isinstance(rezero.observation_map, nn.Linear)
            rezero.observation_map.weight.copy_(plain.observation_map.weight)
            assert isinstance(rezero.gate, ReZero)
            rezero.gate.scalar.copy_(torch.tensor(1.0))

        torch.testing.assert_close(rezero(y, x), plain(y, x))

    def test_identity_observation_map_uses_x_directly(self) -> None:
        r"""Identity observation maps should use the hidden state directly."""
        cell = InnovationCell(4, 4, observation_map="identity", gate="identity")
        y = torch.randn(7, 4)
        x = torch.randn(7, 4)

        assert isinstance(cell.observation_map, nn.Identity)
        innovation = x - y

        torch.testing.assert_close(
            cell(y, x), x - compute_correction(innovation, cell.gain, x)
        )

    def test_from_direct_observation_model_recovers_observation(self) -> None:
        r"""The direct-observation constructor should recover the observation exactly."""
        cell = InnovationCell.from_direct_observation_model(4, gate="identity")
        x = torch.randn(7, 4)
        y = torch.randn(7, 4)

        assert isinstance(cell.gain, ConstantGain)
        assert isinstance(cell.observation_map, nn.Identity)
        torch.testing.assert_close(cell.gain.weight, torch.eye(4))
        torch.testing.assert_close(cell(y, x), y)

    def test_accepts_custom_gain(self) -> None:
        r"""Custom gains should be used verbatim."""

        class DiagonalGain(nn.Module):
            def forward(self, r: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
                return r @ torch.diag(x.mean(dim=0)).mT

        gain = DiagonalGain()
        cell = InnovationCell(3, 3, gain=gain, gate="identity")
        x = torch.randn(7, 3)
        y = torch.randn(7, 3)

        assert cell.gain is gain
        innovation = cell.observation_map(x) - y

        torch.testing.assert_close(
            cell(y, x), x - compute_correction(innovation, gain, x)
        )

    def test_from_direct_observation_model_last_value_gate(self) -> None:
        r"""The last-value preset should initialize ReZero at one."""
        cell = InnovationCell.from_direct_observation_model(4)
        x = torch.randn(7, 4)
        y = torch.randn(7, 4)

        assert isinstance(cell.gate, ReZero)
        torch.testing.assert_close(cell.gate.scalar, torch.tensor(1.0))
        torch.testing.assert_close(cell(y, x), y)

    def test_from_direct_observation_model_average_value_gate(self) -> None:
        r"""The average-value preset should initialize ReZero at one half."""
        cell = InnovationCell.from_direct_observation_model(4, gate="average-value")
        x = torch.randn(7, 4)
        y = torch.randn(7, 4)

        assert isinstance(cell.gate, ReZero)
        torch.testing.assert_close(cell.gate.scalar, torch.tensor(0.5))
        torch.testing.assert_close(cell(y, x), 0.5 * (x + y))

    def test_from_direct_observation_model_keep_state_gate(self) -> None:
        r"""The keep-state preset should initialize ReZero at zero."""
        cell = InnovationCell.from_direct_observation_model(4, gate="keep-state")
        x = torch.randn(7, 4)
        y = torch.randn(7, 4)

        assert isinstance(cell.gate, ReZero)
        torch.testing.assert_close(cell.gate.scalar, torch.tensor(0.0))
        torch.testing.assert_close(cell(y, x), x)

    def test_from_direct_observation_model_copies_only_observed_values(self) -> None:
        r"""The direct-observation constructor should overwrite only observed coordinates."""
        cell = InnovationCell.from_direct_observation_model(4, gate="identity")
        x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        y = torch.tensor([[10.0, 0.0, -5.0, 0.0]])
        mask = torch.tensor([[True, False, True, False]])

        expected = torch.tensor([[10.0, 2.0, -5.0, 4.0]])

        torch.testing.assert_close(cell(y, x, mask=mask), expected)

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
            InnovationCell.from_direct_observation_model(4, gate="other")

    @pytest.mark.parametrize(
        ("gain_name", "gain_cls"),
        [("constant", ConstantGain), ("attention", AttentionGain)],
    )
    def test_builtin_gain_option_instantiates_expected_module(
        self, gain_name: str, gain_cls: type[nn.Module]
    ) -> None:
        r"""Built-in gain strings should instantiate the expected module."""
        cell = InnovationCell(3, 5, gain=gain_name, gate="identity")

        assert isinstance(cell.gain, gain_cls)

    def test_accepts_custom_observation_map(self) -> None:
        r"""Custom observation maps should be used verbatim."""
        observation_map = nn.Linear(5, 3, bias=False)
        cell = InnovationCell(3, 5, observation_map=observation_map)

        assert cell.observation_map is observation_map

    def test_accepts_custom_gate(self) -> None:
        r"""Custom gates should be used verbatim."""
        gate = nn.Tanh()
        cell = InnovationCell(3, 5, gate=gate)

        assert cell.gate is gate

    @pytest.mark.parametrize("gain_name", ["constant", "attention"])
    def test_none_gate_maps_to_identity(self, gain_name: str) -> None:
        r"""A None gate should behave like the identity gate."""
        none_gate = InnovationCell(3, 5, gain=gain_name, gate=None)
        identity_gate = InnovationCell(3, 5, gain=gain_name, gate="identity")
        y = torch.randn(7, 3)
        x = torch.randn(7, 5)

        with torch.no_grad():
            match none_gate.gain, identity_gate.gain:
                case ConstantGain(), ConstantGain():
                    none_gate.gain.weight.copy_(identity_gate.gain.weight)
                case AttentionGain(), AttentionGain():
                    none_gate.gain.query.weight.copy_(identity_gate.gain.query.weight)
                    none_gate.gain.key.weight.copy_(identity_gate.gain.key.weight)
                case _:
                    raise AssertionError("Unexpected gain types.")
            assert isinstance(none_gate.observation_map, nn.Linear)
            assert isinstance(identity_gate.observation_map, nn.Linear)
            none_gate.observation_map.weight.copy_(identity_gate.observation_map.weight)

        assert isinstance(none_gate.gate, nn.Identity)
        torch.testing.assert_close(none_gate(y, x), identity_gate(y, x))

    @pytest.mark.parametrize("gain_name", ["constant", "attention"])
    def test_masked_backward_has_finite_gradients(self, gain_name: str) -> None:
        r"""Masked observations should not destabilize outputs or gradients."""
        torch.manual_seed(0)

        cell = InnovationCell(5, 7, gain=gain_name, gate="identity")
        x = torch.randn(8, 7, requires_grad=True)
        y = torch.randn(8, 5)
        mask = torch.rand(8, 5) < 0.5

        output = cell(y, x, mask=mask)
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
            InnovationCell(3, 5, observation_map="identity")

    def test_rejects_unknown_gain(self) -> None:
        r"""Unknown gain strings should fail explicitly."""
        with pytest.raises(
            ValueError,
            match=r"Unknown gain: 'other'. Expected 'constant', 'attention', or an nn.Module.",
        ):
            InnovationCell(3, 5, gain="other")

    def test_rejects_unknown_gate(self) -> None:
        r"""Unknown gate strings should fail explicitly."""
        with pytest.raises(
            ValueError,
            match=r"Unknown gate: 'other'. Expected 'rezero', 'identity', None, or an nn.Module.",
        ):
            InnovationCell(3, 5, gate="other")


def test_attention_gain_matches_manual_attention_formula() -> None:
    r"""AttentionGain should match the manual softmax attention formula."""
    gain = AttentionGain(3, 5, hidden_size=2)
    r = torch.randn(2, 7, 3)
    x = torch.randn(2, 7, 5)

    correction = gain(r, x)
    query = gain.query_proj(x).unflatten(
        -1, (gain.output_size, gain.num_heads, gain.head_dim)
    )
    key = gain.key_proj(x).unflatten(
        -1, (gain.input_size, gain.num_heads, gain.head_dim)
    )
    scores = gain.head_dim**-0.5 * (query.swapaxes(-2, -3) @ key.swapaxes(-2, -3).mT)
    expected = (
        (scores.softmax(dim=-1) @ r.unsqueeze(-1).unsqueeze(-3)).squeeze(-3).squeeze(-1)
    )

    assert correction.shape == (2, 7, 5)
    torch.testing.assert_close(correction, expected)


def test_attention_gain_matches_scaled_dot_product_attention() -> None:
    r"""AttentionGain should agree with scaled_dot_product_attention directly."""
    gain = AttentionGain(6, 4, hidden_size=3)
    r = torch.randn(8, 6)
    x = torch.randn(8, 4)

    query = gain.query_proj(x).unflatten(
        -1, (gain.output_size, gain.num_heads, gain.head_dim)
    )
    key = gain.key_proj(x).unflatten(
        -1, (gain.input_size, gain.num_heads, gain.head_dim)
    )
    expected = (
        F.scaled_dot_product_attention(
            query.swapaxes(-2, -3),
            key.swapaxes(-2, -3),
            r.unsqueeze(-1).unsqueeze(-1).swapaxes(-2, -3),
            dropout_p=0.0,
        )
        .squeeze(-3)
        .squeeze(-1)
    )

    torch.testing.assert_close(gain(r, x), expected)


def test_attention_gain_defaults_context_size_to_output_size() -> None:
    r"""AttentionGain should default the context size to the output size."""
    gain = AttentionGain(3, 5, hidden_size=2)

    assert gain.context_size == gain.output_size == 5


def test_attention_gain_rejects_non_positive_hidden_size() -> None:
    r"""AttentionGain should reject non-positive hidden sizes."""
    with pytest.raises(ValueError, match=r"hidden_size must be a positive integer."):
        AttentionGain(3, 5, hidden_size=0)


def test_attention_gain_rejects_non_positive_context_size() -> None:
    r"""AttentionGain should reject non-positive context sizes."""
    with pytest.raises(ValueError, match=r"context_size must be a positive integer."):
        AttentionGain(3, 5, context_size=0)


@pytest.mark.parametrize("gain_name", ["constant", "attention"])
def test_builtin_gain_backward_has_finite_gradients(gain_name: str) -> None:
    r"""Built-in gains should support stable forward and backward passes."""
    torch.manual_seed(0)

    cell = InnovationCell(4, 6, gain=gain_name, gate="identity")
    x = torch.randn(8, 6, requires_grad=True)
    y = torch.randn(8, 4)
    mask = torch.rand(8, 4) < 0.6

    output = cell(y, x, mask=mask)
    loss = output.square().mean()
    loss.backward()

    assert torch.isfinite(output).all()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()

    for parameter in cell.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
