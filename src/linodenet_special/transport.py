r"""Learnable transport maps between special distributions."""

__all__ = [
    "GaussianToMixture",
    "GaussianToTwin",
    "MixtureToGaussian",
    "TwinToGaussian",
]


import torch
from torch import Tensor, nn

from linodenet_special.fallbacks import (
    bimodal_to_gaussian,
    gaussian_to_bimodal,
    gaussian_to_mixture,
    mixture_to_gaussian,
)


class GaussianToMixture(nn.Module):
    r"""Learnable transport map from a Gaussian distribution to a mixture of Gaussians."""

    def __init__(self, num_components: int) -> None:
        super().__init__()
        self.num_components = num_components
        self.weights = nn.Parameter(torch.rand(num_components))
        self.means = nn.Parameter(torch.randn(num_components))
        self.log_std = nn.Parameter(torch.randn(num_components))

    @property
    def stddev(self) -> Tensor:
        return self.log_std.exp()

    def forward(self, x: Tensor) -> Tensor:
        w = self.weights.softmax(dim=-1)
        mu = self.means
        sigma = self.log_std.exp()
        return gaussian_to_mixture(x, w, mu, sigma)

    def inverse(self, x: Tensor) -> Tensor:
        w = self.weights.softmax(dim=-1)
        mu = self.means
        sigma = self.log_std.exp()
        return mixture_to_gaussian(x, w, mu, sigma)


class MixtureToGaussian(nn.Module):
    r"""Learnable transport map from a mixture of Gaussians to a Gaussian distribution."""

    def __init__(self, num_components: int) -> None:
        super().__init__()
        self.num_components = num_components
        self.weights = nn.Parameter(torch.rand(num_components))
        self.means = nn.Parameter(torch.randn(num_components))
        self.log_std = nn.Parameter(torch.randn(num_components))

    @property
    def stddev(self) -> Tensor:
        return self.log_std.exp()

    def forward(self, x: Tensor) -> Tensor:
        w = self.weights.softmax(dim=-1)
        mu = self.means
        sigma = self.log_std.exp()
        return mixture_to_gaussian(x, w, mu, sigma)

    def inverse(self, x: Tensor) -> Tensor:
        w = self.weights.softmax(dim=-1)
        mu = self.means
        sigma = self.log_std.exp()
        return gaussian_to_mixture(x, w, mu, sigma)


class GaussianToTwin(nn.Module):
    r"""Learnable transport map from a Gaussian distribution to a bimodal distribution."""

    def __init__(self) -> None:
        super().__init__()
        self.mean = nn.Parameter(torch.randn(()))
        self.log_std = nn.Parameter(torch.randn(()))

    @property
    def stddev(self) -> Tensor:
        return self.log_std.exp()

    def forward(self, x: Tensor) -> Tensor:
        mu = self.mean
        sigma = self.log_std.exp()
        return gaussian_to_bimodal(x, mu, sigma)

    def inverse(self, x: Tensor) -> Tensor:
        mu = self.mean
        sigma = self.log_std.exp()
        return bimodal_to_gaussian(x, mu, sigma)


class TwinToGaussian(nn.Module):
    r"""Learnable transport map from a bimodal distribution to a Gaussian distribution."""

    def __init__(self) -> None:
        super().__init__()
        self.mean = nn.Parameter(torch.randn(()))
        self.log_std = nn.Parameter(torch.randn(()))

    @property
    def stddev(self) -> Tensor:
        return self.log_std.exp()

    def forward(self, x: Tensor) -> Tensor:
        mu = self.mean
        sigma = self.log_std.exp()
        return bimodal_to_gaussian(x, mu, sigma)

    def inverse(self, x: Tensor) -> Tensor:
        mu = self.mean
        sigma = self.log_std.exp()
        return gaussian_to_bimodal(x, mu, sigma)
