r"""Tests for torch.func.grad."""

import torch
from torch import Tensor


def test_grad_with_tuple_argument() -> None:
    def inner_product(pair: tuple[Tensor, Tensor]) -> Tensor:
        x, y = pair
        return (x * y).sum()

    x = torch.tensor([1.0, 2.0, 3.0])
    y = torch.tensor([4.0, 5.0, 6.0])

    grad_x, grad_y = torch.func.grad(inner_product)((x, y))

    torch.testing.assert_close(grad_x, y)
    torch.testing.assert_close(grad_y, x)
