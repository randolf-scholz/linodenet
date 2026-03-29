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
    r"""Piecewise linear function (3 regions), close the origin: a*x, outside: mx±c.

    Args:
        x: The input tensor to be activated.
        a: The slope of the middle region, defaults to ℯ²
        c: The offset of the parallel lines, defaults to 2.0.
        m: The slope of the outer regions, defaults to 1.0.

    Note: Inversion formula
        $y = f(x, a, c, m) ⟺ x = f(y, 1/a, c, 1/m)$

    Note:
        The optimal transport from $½N(μ, σ²) + ½N(-μ, σ²)$ to $N(0,1)$
        can be approximated with hard_bend(x, σ⁻¹ℯ^{-½μ²/σ²}, μ/σ, 1/σ).

        An optimal transport from $N(0,1)$ to $½N(μ, σ²) + ½N(-μ, σ²)$ can be
        approximated with hard_bend(x, σℯ^{½μ²/σ²}, μ/σ, σ).
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
    r"""Hard expand activation function.

    Args:
        x: The input tensor to be activated.
        a: The slope of the linear region, must be ≥1.
        c: The offset of the parallel lines, must be >0.

    The critical value is $λx⁎ = x⁎+c ⟺ x⁎ = c/(λ-1)$.

    .. math:: Φ(x, λ, c) = {
           x+c  if   λx > x+c         (i.e. x > c/(λ-1))
           λx   if   x-c ≤ λx ≤ x+c   (i.e. x∈[-c/(λ-1), c/(λ-1)])
           x-c  if   λx < x-c         (i.e. x < -c/(λ-1))
        }

    Note:
        ``hard_expand(x, λ, c)`` is the inverse of ``hard_contract(x, 1/λ, c)``.
        `hard_expand` is a piecewise linear approximation of `gaussian_to_bimodal`.
    """
    assert a >= 1.0
    return torch.where((a - 1) * x.abs() <= c, a * x, x + x.sign() * c)


@signature("[(...), (), ()] -> (...)")
def hard_contract(x: Tensor, a: Tensor | float = 1, c: Tensor | float = 1) -> Tensor:
    r"""Inverse of the hard bend activation function.

    Args:
        x: The input tensor to be activated.
        a: The slope of the linear region, must be ≤1.
        c: The offset of the parallel lines, must be >0.

    the critical value is $λx⁎ = x⁎-c ⟺ x⁎=c/(1-λ)$

    .. math:: Φ⁻¹(x, λ, c) = {
           x-c  if   λx  < x-c         (i.e. x > c/(1-λ) )
           λx   if   y-c ≤ λx ≤ x+c    (i.e. x ∈ [-c/(1-λ), +c/(1-λ)] )
           x+c  if   λx  > y+c         (i.e. x < -c/(1-λ) )
        }

    Note:
        ``hard_contract(x, λ, c)`` is the inverse of ``hard_expand(x, 1/λ, c)``.
        `hard_contract` is a piecewise linear approximation of `bimodal_to_gaussian`.
    """
    assert a <= 1.0
    return torch.where((1 - a) * x.abs() <= c, a * x, x - x.sign() * c)
