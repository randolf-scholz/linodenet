r"""Implementations of the hard bend activation function."""

__all__ = ["hard_bend", "HardBend"]

import torch
from torch import Tensor, nn


def hard_bend(x: Tensor, a: float = 1, t: float = 1) -> Tensor:
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


class HardBend(nn.Module):
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
