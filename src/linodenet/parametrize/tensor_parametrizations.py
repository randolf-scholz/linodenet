r"""Parametrizations for rank-3 and higher order tensors."""

__all__ = ["ReZero"]

from typing import Optional

import torch
from torch import Tensor, jit, nn

from linodenet.parametrize.base import Parametrization


class ReZero(Parametrization):
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
        super().__init__(tensor, unsafe=False)
        self.learnable = learnable

        initial_value = torch.as_tensor(0.0 if scalar is None else scalar)
        self.scalar = nn.Parameter(initial_value) if self.learnable else initial_value

    @jit.export
    def forward(self, x: Tensor) -> Tensor:
        """.. Signature:: ``(...,) -> (...,)``."""
        return self.scalar * x

    @jit.export
    def right_inverse(self, y: Tensor, /) -> Tensor:
        r""".. Signature:: ``(...,) -> (...,)``."""
        return y / self.scalar
