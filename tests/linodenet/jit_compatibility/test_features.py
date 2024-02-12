"""Test various things for JIT compatibility."""

import torch
from torch import Tensor, jit


def test_ternary():
    """Test JIT compatibility of ternary operations."""

    def all_pos(x: Tensor) -> bool:
        """Dummy function."""
        return bool(torch.all(x > 0))

    scripted = jit.script(all_pos)
    arg = torch.tensor([1, 2, 3])

    assert scripted(arg)


def test_zip_strict():
    """Test JIT compatibility of `zip(strict=True)`."""

    def sum_inner(x_vals: list[Tensor], y_vals: list[Tensor]) -> Tensor:
        """Sum inner products."""
        result = torch.tensor(0.0)
        for x, y in zip(x_vals, y_vals, strict=True):
            result = result + torch.dot(x, y)
        return result

    scripted = jit.script(sum_inner)
    xs = [torch.tensor([1, 2]), torch.tensor([3, 4])]
    ys = [torch.tensor([5, 6]), torch.tensor([7, 8])]
    scripted(xs, ys)
