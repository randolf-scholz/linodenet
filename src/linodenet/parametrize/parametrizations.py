"""Parametrizations.

There are 2 types of parametrizations:

1. ad-hoc parametrizations
2. other parametrizations
"""

__all__ = ["SpectralNormalization", "ReZero"]

from typing import Any, Optional

import torch
from torch import Tensor, jit, nn

from linodenet.lib import singular_triplet
from linodenet.parametrize._parametrize import Parametrization


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
    maxiter: Optional[int]

    def __init__(
        self, weight: nn.Parameter, /, gamma: float = 1.0, maxiter: Optional[int] = None
    ) -> None:
        super().__init__()
        self.maxiter = maxiter

        assert len(weight.shape) == 2
        m, n = weight.shape

        options: Any = {
            "dtype": weight.dtype,
            "device": weight.device,
            "layout": weight.layout,
        }

        # parametrized and cached
        self.register_parametrized_tensor("weight", weight)
        self.register_cached_tensor("u", torch.randn(m, **options))
        self.register_cached_tensor("v", torch.randn(n, **options))
        self.register_cached_tensor("sigma", torch.ones((), **options))

        # constants
        self.register_buffer("ONE", torch.ones((), **options))
        self.register_buffer("GAMMA", torch.full_like(self.ONE, gamma, **options))

    def forward(self) -> dict[str, Tensor]:
        """Perform spectral normalization w ↦ w/‖w‖₂."""
        sigma, u, v = singular_triplet(
            self.weight, u0=self.u, v0=self.v, maxiter=self.maxiter
        )
        gamma = torch.minimum(self.ONE, self.GAMMA / sigma)
        weight = gamma * self.weight

        return {
            "weight": weight,
            "u": u,
            "v": v,
            "sigma": sigma,
        }


class ReZero(Parametrization):
    """ReZero."""

    scalar: Tensor

    def __init__(
        self,
        *,
        scalar: Optional[Tensor] = None,
        learnable: bool = True,
    ) -> None:
        super().__init__()
        self.learnable = learnable

        if scalar is None:
            initial_value = torch.tensor(0.0)
        else:
            initial_value = scalar

        if self.learnable:
            self.scalar = nn.Parameter(initial_value)
        else:
            self.scalar = initial_value

    @jit.export
    def forward(self, x: Tensor) -> Tensor:
        """.. Signature:: ``(...,) -> (...,)``."""
        return self.scalar * x
