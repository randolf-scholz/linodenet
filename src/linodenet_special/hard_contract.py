r"""Implementations of the hard bend activation function."""

__all__ = [
    "hard_contract",
    "hard_expand",
]

import torch
from torch import Tensor

from signatures import signature


@signature("[(...), (), ()] -> (...)")
def hard_expand(x: Tensor, a: Tensor | float = 1, c: Tensor | float = 1) -> Tensor:
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
        `hard_expand` is a piecewise linear approximation of `gaussian_to_twin`.
    """
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
        `hard_contract` is a piecewise linear approximation of `twin_to_gaussian`.
    """
    return torch.where((1 - a) * x.abs() <= c, a * x, x - x.sign() * c)
