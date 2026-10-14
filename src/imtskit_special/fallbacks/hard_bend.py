r"""Implementations of the hard bend activation function."""

__all__ = [
    "hard_contract",
    "hard_expand",
    "hard_bend",
]

import math

import torch
from torch import Tensor

from signatures import signature


@signature("[(...), (), (), ()] -> (...)")
def hard_bend(
    x: Tensor,
    /,
    a: Tensor | float = math.e**2,
    c: Tensor | float = 2.0,
    m: Tensor | float = 1.0,
) -> Tensor:
    r"""Apply a continuous piecewise linear activation with three regions.

    The middle region has slope $a$. The outer regions have slope $|m|$
    with the sign of $a$, and offsets $±|c|$.

    Args:
        x: The input tensor to be activated.
        a: The slope of the middle region. Defaults to $ℯ²$.
        c: The offset of the parallel lines, defaults to 2.0.
        m: The slope of the outer regions, defaults to 1.0.

    Note: Inversion formula
        For $a > 0$ and $m > 0$, the function is invertible:

        $$ y = f(x, a, c, m) ⟺ x = f(y, 1/a, c/m, 1/m) $$

    Note:
        An optimal transport from $N(0,1)$ to $½N(μ, σ²) + ½N(-μ, σ²)$
        can be approximated by $f(x, σℯ^{½μ²/σ²}, μ, σ)$ for $μ ≥ 0$ and $σ > 0$.

        The optimal transport from $½N(μ, σ²) + ½N(-μ, σ²)$ to $N(0,1)$
        can be approximated by $f(x, σ⁻¹ℯ^{−½μ²/σ²}, μ/σ, 1/σ)$.
    """
    a = torch.as_tensor(a, dtype=x.dtype, device=x.device)
    m = torch.as_tensor(m, dtype=x.dtype, device=x.device)
    c_abs = torch.as_tensor(c, dtype=x.dtype, device=x.device).abs()
    m = torch.copysign(m, a)
    z = (a - m) * x
    return torch.where(
        z.abs() <= c_abs,
        a * x,
        (m * x + z.sign() * c_abs),
    )


@signature("[(...), (), ()] -> (...)")
def hard_expand(x: Tensor, a: Tensor | float = 1.0, c: Tensor | float = 1.0) -> Tensor:
    r"""Apply the hard expand activation function.

    Args:
        x: The input tensor to be activated.
        a: The slope of the middle region. Must satisfy $a ≥ 1$.
        c: The offset of the parallel lines. Must satisfy $c > 0$.

    For $a > 1$, the transition occurs at $x⁎ = c/(a-1)$:

    $$
    f(x, a, c) = \begin{cases}
        x+c  & x > x⁎   \\
        ax   & |x| ≤ x⁎ \\
        x-c  & x < -x⁎
    \end{cases}
    $$

    When $a = 1$, the function is the identity.

    Note:
        `hard_expand(x, a, c)` is the inverse of `hard_contract(x, 1/a, c)`.
        `hard_expand` is a piecewise linear approximation of `gaussian_to_bimodal`.
    """
    return torch.where((a - 1) * x.abs() <= c, a * x, x + x.sign() * c)


@signature("[(...), (), ()] -> (...)")
def hard_contract(x: Tensor, a: Tensor | float = 1, c: Tensor | float = 1) -> Tensor:
    r"""Apply the hard contract activation, the inverse of hard expand.

    Args:
        x: The input tensor to be activated.
        a: The slope of the middle region. Must satisfy $0 < a ≤ 1$ for invertibility.
        c: The offset of the parallel lines. Must satisfy $c > 0$.

    For $0 < a < 1$, the transition occurs at $x⁎ = c/(1-a)$:

    $$
    f(x, a, c) = \begin{cases}
        x-c &   x > x⁎   \\
        ax  & |x| ≤ x⁎   \\
        x+c &   x < -x⁎
    \end{cases}
    $$

    When $a = 1$, the function is the identity.

    Note:
        `hard_contract(x, a, c)` is the inverse of `hard_expand(x, 1/a, c)`.
        `hard_contract` is a piecewise linear approximation of `bimodal_to_gaussian`.
    """
    return torch.where((1 - a) * x.abs() <= c, a * x, x - x.sign() * c)
