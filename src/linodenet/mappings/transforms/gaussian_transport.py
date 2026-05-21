r"""Optimal transport maps from a Gaussian to a mixture of Gaussians."""

__all__ = [
    "GaussianToMixture",
    "MixtureToGaussian",
    "GaussianToBimodal",
    "BimodalToGaussian",
]


import torch
from torch import Tensor, nn

from linodenet.mappings.base import TransformBase
from linodenet_special import (
    bimodal_to_gaussian,
    bimodal_to_gaussian_value_and_grad,
    gaussian_to_bimodal,
    gaussian_to_bimodal_value_and_grad,
    gaussian_to_mixture,
    gaussian_to_mixture_value_and_grad,
    mixture_to_gaussian,
    mixture_to_gaussian_value_and_grad,
)


class BimodalToGaussian(TransformBase):
    r"""Monotone transport from the symmetric bimodal mixture $½N(-μ,σ²)+½N(μ,σ²)$ to $N(0,1)$.

    .. math:: y = Φ⁻¹\Bigl(½Φ((x+μ)/σ) + ½Φ((x-μ)/σ)\Bigr)

    with `inverse` evaluating the corresponding Gaussian-to-bimodal map.

    The transport is evaluated with numerically stable lower-tail and upper-tail
    formulas based on `log_ndtr` and `ndtri_exp`.

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

    def encode(self, x: Tensor, /) -> Tensor:
        mu = self.mean
        sigma = self.log_std.exp()
        return bimodal_to_gaussian(x, mu, sigma)

    def decode(self, x: Tensor, /) -> Tensor:
        mu = self.mean
        sigma = self.log_std.exp()
        return gaussian_to_bimodal(x, mu, sigma)

    def encode_and_logabsdet(self, x: Tensor, /) -> tuple[Tensor, Tensor]:
        mu = self.mean
        sigma = self.log_std.exp()
        y, grad = bimodal_to_gaussian_value_and_grad(x, mu, sigma)
        # Note: grad is guaranteed to be positive for these 1D transport maps
        return y, grad.log()

    def decode_and_logabsdet(self, y: Tensor, /) -> tuple[Tensor, Tensor]:
        mu = self.mean
        sigma = self.log_std.exp()
        x, grad = gaussian_to_bimodal_value_and_grad(y, mu, sigma)
        # Note: grad is guaranteed to be positive for these 1D transport maps
        return x, grad.log()


class GaussianToBimodal(TransformBase):
    r"""Monotone transport from $N(0,1)$ to the symmetric bimodal mixture $½N(-μ,σ²)+½N(μ,σ²)$.

    The map is the inverse of

    .. math:: y = Φ⁻¹\Bigl(½Φ((x+μ)/σ) + ½Φ((x-μ)/σ)\Bigr)

    so `forward` returns the unique $x$ whose bimodal CDF matches $Φ(y)$.
    The inverse map is not available in closed form and is computed with a
    safeguarded Newton iteration. Evaluation of the underlying transport uses
    numerically stable lower-tail and upper-tail formulas based on `log_ndtr`
    and `ndtri_exp`.

    Properties:
        - strictly increasing
        - smooth for $σ > 0$
        - derivative bounded by
          $σ ≤ ∂x/∂y ≤ σ\exp(½(μ/σ)²)$
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

    def encode(self, x: Tensor, /) -> Tensor:
        mu = self.mean
        sigma = self.log_std.exp()
        return gaussian_to_bimodal(x, mu, sigma)

    def decode(self, x: Tensor, /) -> Tensor:
        mu = self.mean
        sigma = self.log_std.exp()
        return bimodal_to_gaussian(x, mu, sigma)

    def encode_and_logabsdet(self, x: Tensor, /) -> tuple[Tensor, Tensor]:
        mu = self.mean
        sigma = self.log_std.exp()
        y, grad = gaussian_to_bimodal_value_and_grad(x, mu, sigma)
        # Note: grad is guaranteed to be positive for these 1D transport maps
        return y, grad.log()

    def decode_and_logabsdet(self, y: Tensor, /) -> tuple[Tensor, Tensor]:
        mu = self.mean
        sigma = self.log_std.exp()
        x, grad = bimodal_to_gaussian_value_and_grad(y, mu, sigma)
        # Note: grad is guaranteed to be positive for these 1D transport maps
        return x, grad.log()


class MixtureToGaussian(TransformBase):
    r"""Monotone transport from the Gaussian mixture $∑ₖ ωₖ N(μₖ, σₖ²)$ to $N(0,1)$.

    .. math::  y = Φ⁻¹\Bigl(∑ₖ ωₖΦ((x-μₖ)/σₖ)\Bigr)

    where $Φ$ is the standard normal CDF.
    The transport is evaluated with numerically stable lower-tail and upper-tail
    formulas based on `log_ndtr` and `ndtri_exp`.

    Properties:
        - strictly increasing in $x$
        - smooth for positive component scales
        - Jacobian
          $∂y/∂x = ∑ₖ (ωₖ/σₖ)\exp(½(y²-zₖ²))$ is strictly positive
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

    def encode(self, x: Tensor, /) -> Tensor:
        w = self.weights.softmax(dim=-1)
        mu = self.means
        sigma = self.log_std.exp()
        return mixture_to_gaussian(x, w, mu, sigma)

    def decode(self, x: Tensor, /) -> Tensor:
        w = self.weights.softmax(dim=-1)
        mu = self.means
        sigma = self.log_std.exp()
        return gaussian_to_mixture(x, w, mu, sigma)

    def encode_and_logabsdet(self, x: Tensor, /) -> tuple[Tensor, Tensor]:
        w = self.weights.softmax(dim=-1)
        mu = self.means
        sigma = self.log_std.exp()
        y, grad = mixture_to_gaussian_value_and_grad(x, w, mu, sigma)
        # Note: grad is guaranteed to be positive for these 1D transport maps
        return y, grad.log()

    def decode_and_logabsdet(self, y: Tensor, /) -> tuple[Tensor, Tensor]:
        w = self.weights.softmax(dim=-1)
        mu = self.means
        sigma = self.log_std.exp()
        x, grad = gaussian_to_mixture_value_and_grad(y, w, mu, sigma)
        # Note: grad is guaranteed to be positive for these 1D transport maps
        return x, grad.log()


class GaussianToMixture(TransformBase):
    r"""Monotone transport from $N(0,1)$ to the Gaussian mixture $∑ₖ ωₖ N(μₖ, σₖ²)$.

    The map is defined implicitly as the inverse of

    .. math::  y = Φ⁻¹\Bigl(∑ₖ ωₖΦ((x-μₖ)/σₖ)\Bigr)

    hence `forward` returns the unique $x$ with mixture CDF equal to $Φ(y)$.
    The inverse map is not available in closed form and is computed with a
    safeguarded Newton iteration. Evaluation of the underlying transport uses
    numerically stable lower-tail and upper-tail formulas based on `log_ndtr`
    and `ndtri_exp`.

    Properties:
        - strictly increasing, since it is the inverse of a monotone CDF transport
        - smooth for positive component scales
        - Jacobian strictly positive, with bounds inherited from the reciprocal
          slope of the mixture-to-Gaussian map
        - asymptotically piecewise affine in the tails, matching the dominant
          component behavior $x ≈ μₖ + σₖy$
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

    def encode(self, x: Tensor, /) -> Tensor:
        w = self.weights.softmax(dim=-1)
        mu = self.means
        sigma = self.log_std.exp()
        return gaussian_to_mixture(x, w, mu, sigma)

    def decode(self, x: Tensor, /) -> Tensor:
        w = self.weights.softmax(dim=-1)
        mu = self.means
        sigma = self.log_std.exp()
        return mixture_to_gaussian(x, w, mu, sigma)

    def encode_and_logabsdet(self, x: Tensor, /) -> tuple[Tensor, Tensor]:
        w = self.weights.softmax(dim=-1)
        mu = self.means
        sigma = self.log_std.exp()
        y, grad = gaussian_to_mixture_value_and_grad(x, w, mu, sigma)
        # Note: grad is guaranteed to be positive for these 1D transport maps
        return y, grad.log()

    def decode_and_logabsdet(self, y: Tensor, /) -> tuple[Tensor, Tensor]:
        w = self.weights.softmax(dim=-1)
        mu = self.means
        sigma = self.log_std.exp()
        x, grad = mixture_to_gaussian_value_and_grad(y, w, mu, sigma)
        # Note: grad is guaranteed to be positive for these 1D transport maps
        return x, grad.log()
