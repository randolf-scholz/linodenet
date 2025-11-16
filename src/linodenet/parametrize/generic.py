r"""General parametrizations."""

__all__ = ["ReZero", "Identity"]

from typing import Optional

import torch
from torch import Tensor, jit, nn

from linodenet import projections
from linodenet.parametrize.base import ParametrizationBase


class ReZero(ParametrizationBase):
    r"""ReZero."""

    scalar: Tensor

    def __init__(
        self,
        tensor: Tensor,
        /,
        *,
        scalar: Optional[Tensor] = None,
        learnable: bool = True,
    ) -> None:
        super().__init__(tensor)
        self.learnable = learnable

        initial_value = torch.as_tensor(0.0 if scalar is None else scalar)
        self.scalar = nn.Parameter(initial_value) if self.learnable else initial_value

    @jit.export
    def forward(self, x: Tensor) -> Tensor:
        """.. Signature:: ``(...,) -> (...,)``."""
        return self.scalar * x

    def right_inverse(self, y: Tensor, /) -> Tensor:
        r""".. Signature:: ``(...,) -> (...,)``."""
        return y / self.scalar


class Identity(ParametrizationBase):
    r"""Parametrize a matriz as itself."""

    def forward(self, x: Tensor) -> Tensor:
        r""".. Signature:: ``... -> ...``."""
        return projections.identity(x)

    def right_inverse(self, y: Tensor) -> Tensor:
        r""".. Signature:: ``... -> ...``."""
        return y
