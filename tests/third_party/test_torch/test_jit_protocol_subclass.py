r"""Test the `torch.jit` module with a protocol subclass."""

from tempfile import TemporaryFile
from typing import Protocol, runtime_checkable

import torch
from torch import Tensor, jit, nn


@runtime_checkable
class Cell(Protocol):
    input_size: int
    hidden_size: int


class MyCell(nn.Module, Cell):
    input_size: int
    hidden_size: int

    def __init__(self, input_size: int, hidden_size: int) -> None:
        nn.Module.__init__(self)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.cell = nn.RNNCell(input_size, hidden_size)

    def forward(self, y: Tensor, x: Tensor) -> Tensor:
        return self.cell(y, x)


def test_jit_cell() -> None:
    cell = MyCell(3, 3)
    scripted = jit.script(cell)
    with TemporaryFile() as f:
        jit.save(scripted, f)
        f.seek(0)
        reloaded = jit.load(f)

    print(MyCell.__mro__)

    assert isinstance(cell, nn.Module)
    assert isinstance(cell, MyCell)
    assert isinstance(cell, Cell)

    assert isinstance(scripted, nn.Module)
    # assert isinstance(scripted, MyCell)
    assert isinstance(scripted, Cell)

    assert isinstance(reloaded, nn.Module)
    assert isinstance(reloaded, MyCell)
    assert isinstance(reloaded, Cell)

    assert reloaded.input_size == 3
    assert reloaded.hidden_size == 3

    x = torch.randn(5, 3)
    y = torch.randn(5, 3)

    assert torch.equal(reloaded(y, x), cell(y, x))
