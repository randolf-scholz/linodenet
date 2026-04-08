__all__ = [
    "BimodalToGaussian",
    "GaussianToBimodal",
    "GaussianToMixture",
    "MixtureToGaussian",
    "bimodal_to_gaussian",
    "gaussian_to_mixture",
    "gaussian_to_bimodal",
    "hard_bend",
    "mixture_to_gaussian",
]

import torch
from torch import Tensor, nn

from linodenet_special import (
    bimodal_to_gaussian as bimodal_to_gaussian,
    gaussian_to_bimodal as gaussian_to_bimodal,
    gaussian_to_mixture as gaussian_to_mixture,
    hard_bend as hard_bend,
    mixture_to_gaussian as mixture_to_gaussian,
)


class BimodalToGaussian(nn.Module):
    r"""Wrap `bimodal_to_gaussian` as an `nn.Module`."""

    mu: Tensor
    sigma: Tensor

    def __init__(
        self,
        mu: Tensor | float = 2.0,
        sigma: Tensor | float = 1.0,
        *,
        learnable: bool = False,
    ) -> None:
        super().__init__()
        mu_tensor = torch.as_tensor(mu)
        sigma_tensor = torch.as_tensor(sigma)

        if learnable:
            self.mu = nn.Parameter(mu_tensor)
            self.sigma = nn.Parameter(sigma_tensor)
        else:
            self.register_buffer("mu", mu_tensor)
            self.register_buffer("sigma", sigma_tensor)

    def forward(self, x: Tensor, /) -> Tensor:
        return bimodal_to_gaussian(x, self.mu, self.sigma)


class GaussianToBimodal(nn.Module):
    r"""Wrap `gaussian_to_bimodal` as an `nn.Module`."""

    mu: Tensor
    sigma: Tensor

    def __init__(
        self,
        mu: Tensor | float = 2.0,
        sigma: Tensor | float = 1.0,
        *,
        learnable: bool = False,
    ) -> None:
        super().__init__()
        mu_tensor = torch.as_tensor(mu)
        sigma_tensor = torch.as_tensor(sigma)

        if learnable:
            self.mu = nn.Parameter(mu_tensor)
            self.sigma = nn.Parameter(sigma_tensor)
        else:
            self.register_buffer("mu", mu_tensor)
            self.register_buffer("sigma", sigma_tensor)

    def forward(self, x: Tensor, /) -> Tensor:
        return gaussian_to_bimodal(x, self.mu, self.sigma)


class MixtureToGaussian(nn.Module):
    r"""Wrap `mixture_to_gaussian` as an `nn.Module`."""

    weights: Tensor
    mus: Tensor
    sigmas: Tensor

    def __init__(
        self,
        weights: Tensor,
        mus: Tensor,
        sigmas: Tensor,
        *,
        learnable: bool = False,
    ) -> None:
        super().__init__()
        weights_tensor = torch.as_tensor(weights)
        mus_tensor = torch.as_tensor(mus)
        sigmas_tensor = torch.as_tensor(sigmas)

        if learnable:
            self.weights = nn.Parameter(weights_tensor)
            self.mus = nn.Parameter(mus_tensor)
            self.sigmas = nn.Parameter(sigmas_tensor)
        else:
            self.register_buffer("weights", weights_tensor)
            self.register_buffer("mus", mus_tensor)
            self.register_buffer("sigmas", sigmas_tensor)

    def forward(self, x: Tensor, /) -> Tensor:
        return mixture_to_gaussian(x, self.weights, self.mus, self.sigmas)


class GaussianToMixture(nn.Module):
    r"""Wrap `gaussian_to_mixture` as an `nn.Module`."""

    weights: Tensor
    mus: Tensor
    sigmas: Tensor

    def __init__(
        self,
        weights: Tensor,
        mus: Tensor,
        sigmas: Tensor,
        *,
        learnable: bool = False,
    ) -> None:
        super().__init__()
        weights_tensor = torch.as_tensor(weights)
        mus_tensor = torch.as_tensor(mus)
        sigmas_tensor = torch.as_tensor(sigmas)

        if learnable:
            self.weights = nn.Parameter(weights_tensor)
            self.mus = nn.Parameter(mus_tensor)
            self.sigmas = nn.Parameter(sigmas_tensor)
        else:
            self.register_buffer("weights", weights_tensor)
            self.register_buffer("mus", mus_tensor)
            self.register_buffer("sigmas", sigmas_tensor)

    def forward(self, x: Tensor, /) -> Tensor:
        return gaussian_to_mixture(x, self.weights, self.mus, self.sigmas)
