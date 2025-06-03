r"""Test the `torch.jit` module with a protocol subclass."""

from tempfile import TemporaryFile
from typing import Final, Protocol, runtime_checkable

import pytest
import torch
from torch import Tensor, jit, nn


@runtime_checkable
class Cell(Protocol):
    input_size: Final[int]  # type: ignore[misc]
    hidden_size: Final[int]  # type: ignore[misc]


class MyCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int) -> None:
        nn.Module.__init__(self)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.cell = nn.RNNCell(input_size, hidden_size)

    def forward(self, y: Tensor, x: Tensor) -> Tensor:
        return self.cell(y, x)


@pytest.mark.parametrize("stage", ["original", "scripted", "reloaded"])
def test_jit_cell(stage: str) -> None:
    cell = MyCell(3, 3)

    match stage:
        case "original":
            target: nn.Module = cell
        case "scripted":
            target = jit.script(cell)
        case "reloaded":
            target = jit.script(cell)
            with TemporaryFile() as f:
                jit.save(target, f)
                f.seek(0)
                target = jit.load(f)
        case _:
            raise ValueError(f"Invalid stage: {stage}")

    assert isinstance(target, nn.Module)
    assert target.input_size == 3
    assert target.hidden_size == 3

    x = torch.randn(5, 3)
    y = torch.randn(5, 3)
    assert torch.equal(target(y, x), cell(y, x))

    if stage == "original":
        assert isinstance(target, MyCell)
        assert isinstance(target, Cell)
    else:
        with pytest.raises(AssertionError):
            # Scripted classes do not subclass original class.
            assert isinstance(target, MyCell)
        with pytest.raises(AssertionError):
            # Scripted classes do not support getattr_static")
            assert isinstance(target, Cell)
