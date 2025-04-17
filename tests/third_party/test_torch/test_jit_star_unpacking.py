r"""Test JIT tuple unpacking and concatenation."""

import pytest
import torch
from torch import Tensor, jit


def tuple_unpack(x: Tensor) -> Tensor:
    new_shape = (*x.shape[:1], 1)
    return torch.zeros(new_shape)


def tuple_concat(x: Tensor) -> Tensor:
    new_shape = x.shape[:1] + (1,)
    return torch.zeros(new_shape)


@pytest.mark.xfail(strict=True, raises=RuntimeError)
def test_jit_tuple_unpack() -> None:
    jit.script(tuple_unpack)


def test_jit_tuple_concat() -> None:
    scripted_fn = jit.script(tuple_concat)
    x = torch.randn(5, 3, 3)
    y = scripted_fn(x)
    assert y.shape == (5, 1)
