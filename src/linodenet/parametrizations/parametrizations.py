"""Parametrizations."""

__all__ = ["SpectralNormalization", "ReZero"]

import torch
from torch import Tensor, nn

from linodenet.lib import singular_triplet
from linodenet.parametrizations.base import Parametrization


class SpectralNormalization(Parametrization):
    """Spectral normalization."""

    # constants
    GAMMA: Tensor
    ONE: Tensor

    # cached
    u: Tensor
    v: Tensor
    sigma: Tensor

    # parametrized
    weight: Tensor

    def __init__(self, weight: nn.Parameter, /, gamma: float = 1.0) -> None:
        super().__init__()

        assert len(weight.shape) == 2
        m, n = weight.shape

        options = {
            "dtype": weight.dtype,
            "device": weight.device,
            "layout": weight.layout,
        }

        # parametrized and cached
        self.register_parametrization("weight", weight)
        self.register_cached_tensor("u", torch.empty(m, **options))
        self.register_cached_tensor("v", torch.empty(n, **options))
        self.register_cached_tensor("sigma", torch.empty(1, **options))

        # constants
        self.register_buffer("ONE", torch.empty(1, **options))
        self.register_buffer("GAMMA", torch.empty(gamma, **options))

    def forward(self) -> dict[str, Tensor]:
        sigma, u, v = singular_triplet(self.weight, u0=self.u, v0=self.v)
        gamma = torch.minimum(self.ONE, self.GAMMA / sigma)
        weight = gamma * self.weight
        return {"weight": weight, "u": u, "v": v, "sigma": sigma}


class ReZero:
    """ReZero."""
