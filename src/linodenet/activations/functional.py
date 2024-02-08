r"""Implementations of activation functions.

Notes:
    Contains activations in functional form.
    - See `linodenet.activations.modular` for modular implementations.
"""

__all__ = [
    # Functions
    "geglu",
    "hard_bend",
    "reglu",
]

import torch
from torch import Tensor, nn


def reglu(x: Tensor) -> Tensor:
    r"""Regularized gelu activation function.

    .. math:: ϕ(x) = x₁ * relu(x₂)

    >>> result = reglu(torch.randn(4, 4))
    >>> assert result.shape == (2, 4)

    References:
        Noam Shazeer, "GLU Variants Improve Transformer", 2020
    """
    a, b = x.chunk(2, dim=-1)
    return a * nn.functional.relu(b)


def geglu(x: Tensor) -> Tensor:
    r"""Gelu activation function.

    .. math:: ϕ(x) = x₁ * gelu(x₂)

    >>> result = geglu(torch.randn(4, 4))
    >>> assert result.shape == (4, 2)

    References:
        Noam Shazeer, "GLU Variants Improve Transformer", 2020
    """
    a, b = x.chunk(2, dim=-1)
    return a * nn.functional.gelu(b)


def hard_bend(x: Tensor, a: float = 1, t: float = 1) -> Tensor:
    r"""Hard step activation function.

    .. math:: ϕ(x, a, t) =
        \begin{cases}
            x + t    &  x  >  \frac{t}{e^{at} - 1} \\
            e^{at} x & |x| ≤  \frac{t}{e^{at} - 1} \\
            x - t    &  x  < -\frac{t}{e^{at} - 1}
        \end{cases}
    """
    exp_at = torch.tensor(a * t, device=x.device, dtype=x.dtype).exp()
    mask = x.abs() <= t / (exp_at - 1)
    return torch.where(mask, exp_at * x, x + torch.sign(x) * t)
