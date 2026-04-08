__all__ = [
    "BimodalToGaussian",
    "GaussianToBimodal",
    "bimodal_to_gaussian",
    "gaussian_to_bimodal",
    "hard_bend",
]

import torch
from torch import Tensor, nn

from linodenet_special import (
    bimodal_to_gaussian as bimodal_to_gaussian,
    gaussian_to_bimodal as gaussian_to_bimodal,
    hard_bend as hard_bend,
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
