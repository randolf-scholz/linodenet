r"""Parametrizations for rank-3 and higher order tensors."""

__all__ = ["ReZero"]

from typing import Final, Optional

import torch
from torch import Tensor, jit, nn

from linodenet.parametrize.base import ParametrizationBase
from linodenet.signatures import signature


class ReZero(ParametrizationBase):
    r"""ReZero."""

    DOMAIN: Final[None] = None
    CODOMAIN: Final[None] = None

    scalar: Tensor
    r"""PARAM: The ReZero scalar."""

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
    @signature("(...) -> (...)")
    def forward(self, x: Tensor) -> Tensor:
        return self.scalar * x

    @jit.export
    @signature("(...) -> (...)")
    def right_inverse(self, y: Tensor, /) -> Tensor:
        return y / self.scalar
