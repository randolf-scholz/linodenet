r"""Tests for residual state-updater wrappers."""

import torch
from torch import nn
from torch.nn import GRUCell

from linodenet.state_update import LinearCell, ResidualCell


def test_residual_cell_derives_sizes_from_wrapped_cell() -> None:
    r"""ResidualCell should reuse the wrapped cell dimensions."""
    cell = GRUCell(3, 5)
    residual = ResidualCell(cell)

    y = torch.randn(7, 3)
    x = torch.randn(7, 5)

    assert residual.input_size == 3
    assert residual.hidden_size == 5
    assert isinstance(residual.gate, nn.Identity)
    torch.testing.assert_close(residual(y, x), x - cell(y, x))


def test_residual_cell_applies_gate_to_cell_output() -> None:
    r"""ResidualCell should apply the gate before adding the residual update."""
    cell = LinearCell(3, 5)
    gate = nn.Tanh()
    residual = ResidualCell(cell, gate)

    y = torch.randn(7, 3)
    x = torch.randn(7, 5)

    torch.testing.assert_close(residual(y, x), x - gate(cell(y, x)))
