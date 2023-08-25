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


def custom(x: Tensor, a: float = 0.5) -> Tensor:
    zero = torch.zeros_like(x)
    ones = torch.ones_like(x)
    z = (1 + x / a) ** 2
    return torch.where(x >= -a, torch.minimum(z, ones), zero)


def h(x: Tensor, a: float = 0.5) -> Tensor:
    zero = torch.zeros_like(x)
    ones = torch.ones_like(x)

    y = torch.maximum(1 + x / a, zero)
    z = torch.minimum(y, ones)
    return z
