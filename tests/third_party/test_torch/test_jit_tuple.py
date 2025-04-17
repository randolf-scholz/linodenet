r"""Test the `torch.jit` module with a protocol subclass."""

import pytest
import torch
from torch import jit


def scaled_norm(
    x: torch.Tensor,
    dim: tuple[int, ...] = (),
    p: float = 2.0,
    keepdim: bool = False,
) -> torch.Tensor:
    return torch.mean(x**p, dim=dim, keepdim=keepdim) ** (1 / p)


@pytest.mark.xfail(reason="https://github.com/pytorch/pytorch/issues/64700")
def test_jit_cell() -> None:
    jit.script(scaled_norm)
