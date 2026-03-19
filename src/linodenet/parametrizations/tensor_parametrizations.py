r"""Parametrizations for rank-3 and higher order tensors."""

__all__ = ["ReZero"]

from typing import Final, Optional

import torch
from torch import Tensor, nn

from linodenet.domains import TensorDomains
from linodenet.nn.parametrize import ParametrizationBase
from signatures import signature


class ReZero(ParametrizationBase):
    r"""ReZero."""

    DOMAIN: Final[TensorDomains] = TensorDomains.ANY
    CODOMAIN: Final[TensorDomains] = TensorDomains.ANY

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
        self.scalar = nn.Parameter(initial_value, requires_grad=self.learnable)

    @signature("(...) -> (...)")
    def forward(self, x: Tensor) -> Tensor:
        return self.scalar * x

    @signature("(...) -> (...)")
    def right_inverse(self, y: Tensor, /) -> Tensor:
        return y / self.scalar
