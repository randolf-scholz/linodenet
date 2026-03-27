r"""Optimal transport maps from a Gaussian to a mixture of Gaussians."""

__all__ = [
    "GaussianToMixture",
    "MixtureToGaussian",
    "GaussianToBimodal",
    "BimodalToGaussian",
]


import torch
from torch import Tensor, nn

from linodenet_special import (
    bimodal_to_gaussian,
    gaussian_to_bimodal,
    gaussian_to_mixture,
    mixture_to_gaussian,
)

from .base import TransformBase


class BimodalToGaussian(TransformBase):
    r"""Monotone transport from the symmetric bimodal mixture $½N(-μ,σ²)+½N(μ,σ²)$ to $N(0,1)$.

    .. math:: y = Φ⁻¹\Bigl(½Φ((x+μ)/σ) + ½Φ((x-μ)/σ)\Bigr)

    with `inverse` evaluating the corresponding Gaussian-to-bimodal map.

    Properties:
        - strictly increasing
        - smooth for $σ > 0$
        - derivative bounded by
          $\exp(-½(μ/σ)²)/σ ≤ ∂y/∂x ≤ 1/σ$
        - asymptotically affine in the tails:
          $y ≈ σ⁻¹(x-\sign(x)\abs{μ})$
    """

    def __init__(self) -> None:
        super().__init__()
        self.mean = nn.Parameter(torch.ones(()))
        self.log_std = nn.Parameter(torch.zeros(()))

    @property
    def stddev(self) -> Tensor:
        return self.log_std.exp()

    def encode(self, x: Tensor) -> Tensor:
        mu = self.mean
        sigma = self.log_std.exp()
        return bimodal_to_gaussian(x, mu, sigma)

    def decode(self, x: Tensor) -> Tensor:
        mu = self.mean
        sigma = self.log_std.exp()
        return gaussian_to_bimodal(x, mu, sigma)


class GaussianToBimodal(TransformBase):
    r"""Monotone transport from $N(0,1)$ to the symmetric bimodal mixture $½N(-μ,σ²)+½N(μ,σ²)$.

    The map is the inverse of

    .. math:: y = Φ⁻¹\Bigl(½Φ((x+μ)/σ) + ½Φ((x-μ)/σ)\Bigr)

    so `forward` returns the unique $x$ whose bimodal CDF matches $Φ(y)$.

    Properties:
        - strictly increasing
        - smooth for $σ > 0$
        - derivative bounded by
          $σ ≤ ∂x/∂y ≤ σ \exp(½(μ/σ)²)$
        - asymptotically affine in the tails:
          $x ≈ σy + \sign(y)\abs{μ}$
    """

    def __init__(self) -> None:
        super().__init__()
        self.mean = nn.Parameter(torch.ones(()))
        self.log_std = nn.Parameter(torch.zeros(()))

    @property
    def stddev(self) -> Tensor:
        return self.log_std.exp()

    def encode(self, x: Tensor) -> Tensor:
        mu = self.mean
        sigma = self.log_std.exp()
        return gaussian_to_bimodal(x, mu, sigma)

    def decode(self, x: Tensor) -> Tensor:
        mu = self.mean
        sigma = self.log_std.exp()
        return bimodal_to_gaussian(x, mu, sigma)


class MixtureToGaussian(TransformBase):
    r"""Monotone transport from the Gaussian mixture $∑ₖ ωₖ N(μₖ, σₖ²)$ to $N(0,1)$.

    .. math::  y = Φ⁻¹\Bigl(∑ₖ ωₖΦ((x-μₖ)/σₖ)\Bigr)

    where $Φ$ is the standard normal CDF.

    Properties:
        - strictly increasing in $x$
        - smooth for positive component scales
        - Jacobian
          $∂y/∂x = ∑ₖ (ωₖ/σₖ)\exp(\tfrac12(y²-zₖ²))$ is strictly positive
        - in the far tails it approaches the affine transport of the dominant
          component, so $y ≈ (x-μₖ)/σₖ$
    """

    def __init__(self, num_components: int) -> None:
        super().__init__()
        self.num_components = num_components
        self.weights = nn.Parameter(torch.rand(num_components))
        self.means = nn.Parameter(torch.ones(num_components))
        self.log_std = nn.Parameter(torch.zeros(num_components))

    @property
    def stddev(self) -> Tensor:
        return self.log_std.exp()

    def encode(self, x: Tensor) -> Tensor:
        w = self.weights.softmax(dim=-1)
        mu = self.means
        sigma = self.log_std.exp()
        return mixture_to_gaussian(x, w, mu, sigma)

    def decode(self, x: Tensor) -> Tensor:
        w = self.weights.softmax(dim=-1)
        mu = self.means
        sigma = self.log_std.exp()
        return gaussian_to_mixture(x, w, mu, sigma)


class GaussianToMixture(TransformBase):
    r"""Monotone transport from $N(0,1)$ to the Gaussian mixture $∑ₖ ωₖ N(μₖ, σₖ²)$.

    The map is defined implicitly as the inverse of

    .. math::  y = Φ⁻¹\Bigl(∑ₖ ωₖΦ((x-μₖ)/σₖ)\Bigr)

    hence `forward` returns the unique $x$ with mixture CDF equal to $Φ(y)$.

    Properties:
        - strictly increasing, since it is the inverse of a monotone CDF transport
        - smooth for positive component scales
        - Jacobian strictly positive, with bounds inherited from the reciprocal
          slope of the mixture-to-Gaussian map
        - asymptotically piecewise affine in the tails, matching the dominant
          component behavior $x ≈ μₖ + σₖ y$
    """

    def __init__(self, num_components: int) -> None:
        super().__init__()
        self.num_components = num_components
        self.weights = nn.Parameter(torch.rand(num_components))
        self.means = nn.Parameter(torch.ones(num_components))
        self.log_std = nn.Parameter(torch.zeros(num_components))

    @property
    def stddev(self) -> Tensor:
        return self.log_std.exp()

    def encode(self, x: Tensor) -> Tensor:
        w = self.weights.softmax(dim=-1)
        mu = self.means
        sigma = self.log_std.exp()
        return gaussian_to_mixture(x, w, mu, sigma)

    def decode(self, x: Tensor) -> Tensor:
        w = self.weights.softmax(dim=-1)
        mu = self.means
        sigma = self.log_std.exp()
        return mixture_to_gaussian(x, w, mu, sigma)
