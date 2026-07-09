r"""Tests for residual state-updater wrappers."""

import torch
from torch import nn
from torch.nn import GRUCell

from linodenet.nn.rezero import ReZero
from linodenet.state_update import LinearRNNCell, ResidualCell


class TestResidualCell:
    def test_derives_sizes_from_wrapped_cell(self) -> None:
        r"""ResidualCell should reuse the wrapped cell dimensions."""
        cell = GRUCell(3, 5)
        residual = ResidualCell(cell)

        y = torch.randn(7, 3)
        x = torch.randn(7, 5)

        assert residual.input_size == 3
        assert residual.hidden_size == 5
        assert isinstance(residual.gate, nn.Identity)
        torch.testing.assert_close(residual(y, x), x - cell(y, x))

    def test_applies_gate_to_cell_output(self) -> None:
        r"""ResidualCell should apply the gate before adding the residual update."""
        cell = LinearRNNCell(3, 5)
        gate = nn.Tanh()
        residual = ResidualCell(cell, gate)

        y = torch.randn(7, 3)
        x = torch.randn(7, 5)

        torch.testing.assert_close(residual(y, x), x - gate(cell(y, x)))

    def test_rezero_gate_starts_as_identity(self) -> None:
        r"""The ReZero gate should initialize the residual correction at zero."""
        cell = LinearRNNCell(3, 5)
        residual = ResidualCell(cell, gate="rezero")
        y = torch.randn(7, 3)
        x = torch.randn(7, 5)

        assert isinstance(residual.gate, ReZero)
        torch.testing.assert_close(residual(y, x), x)

    def test_none_gate_maps_to_identity(self) -> None:
        r"""A None gate should resolve to the identity gate."""
        cell = GRUCell(3, 5)
        residual = ResidualCell(cell, gate=None)

        assert isinstance(residual.gate, nn.Identity)
