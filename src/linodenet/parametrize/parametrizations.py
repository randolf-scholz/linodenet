"""Parametrizations.

There are 2 types of parametrizations:

1. ad-hoc parametrizations
2. other parametrizations
"""

__all__ = ["SpectralNormalization", "ReZero"]

from typing import Any, Final, Optional

import torch
from torch import Tensor, jit, nn

from linodenet.lib import singular_triplet
from linodenet.parametrize.base import Parametrization


class SpectralNormalization(Parametrization):
    """Spectral normalization."""

    # constants
    GAMMA: Tensor
    ONE: Tensor
    maxiter: Final[Optional[int]]

    # cached
    u: Tensor
    v: Tensor
    sigma: Tensor
    weight: Tensor

    # Parameters
    original_weight: Tensor

    def __init__(
        self, weight: Tensor, /, gamma: float = 1.0, maxiter: Optional[int] = None
    ) -> None:
        super().__init__()
        assert isinstance(weight, nn.Parameter), "weight must be a parameter"

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
        # IMPORTANT: Use the original weight, not the cached weight!
        # For auxiliary tensors, use the cached tensors.
        sigma, u, v = singular_triplet(
            self.original_weight,
            u0=self.u,
            v0=self.v,
            maxiter=self.maxiter,
        )
        gamma = torch.minimum(self.ONE, self.GAMMA / sigma)
        weight = gamma * self.original_weight

        return {
            "weight": weight,
            "sigma": sigma,
            "u": u,
            "v": v,
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
