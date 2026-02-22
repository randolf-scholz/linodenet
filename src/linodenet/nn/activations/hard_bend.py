r"""Implementations of the hard bend activation function."""

__all__ = ["hard_bend_exp", "HardBendExp"]

import torch
from torch import Tensor, nn


def hard_contract(x: Tensor, a: float = 1, c: float = 1) -> Tensor:
    r"""Inverse of the hard bend activation function.

    Args:
        x: The input tensor to be activated.
        a: The slope of the linear region, must be ≤1.
        c: The offset of the parallel lines, must be >0.

    the critical value is $λx⁎ = x⁎ - c$ ⟺ $x⁎=c/(1 - λ)$

    .. math:: Φ⁻¹(x, λ, t) = {
           x-c  if   λx  < x-c          (i.e. x > c/(1-λ) )
           λx   if   y-c < λx  < x+c    (i.e. x ∈ [-c/(1-λ), +c/(1-λ)] )
           x+c  if   λx  > y+c          (i.e. x < -c/(1-λ) )
        }

    Note:
        ``hard_contract(x, λ, c)`` is the inverse of ``hard_expand(x, 1/λ, c)``.
    """
    x = x / a
    z = x - torch.sign(x) * c
    return torch.where(x.abs() <= z.abs(), x, z)


def hard_expand(x: Tensor, a: float = 1, c: float = 1) -> Tensor:
    r"""Hard expand activation function.

    Args:
        x: The input tensor to be activated.
        a: The slope of the linear region, must be ≥1.
        c: The offset of the parallel lines, must be >0.

    The critical value is $λx⁎ = x⁎ + c$ ⟺ $x⁎ = c/(λ-1)$.

    .. math:: Φ(x, λ, t) = {
           x+c  if   λx > x+c         (i.e. x > c/(λ-1))
           λx   if   x-c ≤ λx ≤ x+c   (i.e. x∈[-c/(λ-1), c/(λ-1)])
           x-c  if   λx < x-c         (i.e. x < -c/(λ-1))
        }

    Note:
        ``hard_expand(x, λ, c)`` is the inverse of ``hard_contract(x, 1/λ, c)``.
    """
    y = a * x
    z = x + torch.sign(x) * c
    return torch.where(y.abs() <= z.abs(), y, z)


def hard_bend_exp(x: Tensor, a: float = 1, t: float = 1) -> Tensor:
    r"""Hard step activation function.

    .. math:: ϕ(x, a, t) =
        \begin{cases}
            x + t    &  x  >  \frac{t}{eᵃᵗ - 1} \\
            eᵃᵗ x    & |x| ≤  \frac{t}{eᵃᵗ - 1} \\
            x - t    &  x  < -\frac{t}{eᵃᵗ - 1}
        \end{cases}
    """
    exp_at = torch.tensor(a * t, device=x.device, dtype=x.dtype).exp()
    mask = x.abs() <= t / (exp_at - 1)
    return torch.where(mask, exp_at * x, x + torch.sign(x) * t)


class HardBendExp(nn.Module):
    r"""Optimized implementation of 2-parameter HardBend that precomputes the threshold and slope.

    .. math:: ϕ(x, a, t) = odesolve(u'(t) = a * tanh(u), t, u(0) = x) = sinh⁻¹(eᵃᵗ sinh(ax))

    The hard bend activation function is defined as:

    .. math:: ϕ(x, a, t) =
        \begin{cases}
            x + t    &  x  >  \frac{t}{eᵃᵗ - 1} \\
            eᵃᵗ x    & |x| ≤  \frac{t}{eᵃᵗ - 1} \\
            x - t    &  x  < -\frac{t}{eᵃᵗ - 1}
        \end{cases}
    """

    a: Tensor
    t: Tensor
    threshold: Tensor
    slope: Tensor

    def __init__(self, a: float = 1.0, t: float = 1.0) -> None:
        super().__init__()
        self.register_buffer("a", torch.tensor(a))
        self.register_buffer("t", torch.tensor(t))
        self.register_buffer("threshold", self.t / (torch.exp(self.a * self.t) - 1))
        self.register_buffer("slope", torch.exp(self.a * self.t))

    def forward(self, x: Tensor) -> Tensor:
        mask = x.abs() <= self.threshold
        return torch.where(mask, self.slope * x, x + torch.sign(x) * self.t)
