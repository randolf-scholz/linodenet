r"""Implementations of activation functions.

Notes
-----
Contains activations in functional form.
  - See `linodenet.activations.modular` for modular implementations.
"""

__all__ = [
    # Functions
    "reglu",
    "geglu",
    "hard_bend",
]

import torch
from torch import Tensor, nn


def reglu(x: Tensor) -> Tensor:
    r"""Regularized gelu activation function."""
    a, b = x.chunk(2, dim=-1)
    return a * nn.functional.relu(b)


def geglu(x: Tensor) -> Tensor:
    r"""Gelu activation function."""
    a, b = x.chunk(2, dim=-1)
    return a * nn.functional.gelu(b)


def hard_bend(x: Tensor, a: float = 1, t: float = 1) -> Tensor:
    r"""Hard step activation function."""
    mask = x.abs() <= t / (torch.exp(a * t) - 1)
    return torch.where(mask, torch.exp(a * t) * x, x + torch.sign(x) * t)
