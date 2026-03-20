r"""ReZero layers.

References:
    - | ReZero is all you need: fast convergence at large depth.
      | Thomas Bachlechner, Bodhisattwa Prasad Majumder, Henry Mao, Gary Cottrell, Julian McAuley
      | Proceedings of the Thirty-Seventh Conference on Uncertainty in Artificial Intelligence, PMLR
      | https://proceedings.mlr.press/v161/bachlechner21a.html
"""

__all__ = [
    "ReZero",
    "ReZeroResNet",
]

from collections.abc import Iterable
from typing import Optional

import torch
from torch import Tensor, nn

from signatures import signature


class ReZero(nn.Module):
    r"""ReZero module.

    Simply multiplies the inputs by a scalar initialized to zero.
    """

    scalar: Tensor
    r"""PARAM: The scalar to multiply the inputs by."""
    scalar_map: nn.Module
    r"""MODULE: Map applied to the scalar before scaling the input."""

    @property
    def config(self) -> dict:
        return {
            "module": self.module,
            "scalar": self.scalar,
            "scalar_map": self.scalar_map,
            "learnable": self.learnable,
        }

    def __init__(
        self,
        module: nn.Module,
        *,
        scalar_map: Optional[nn.Module] = None,
        learnable: bool = True,
    ) -> None:
        super().__init__()
        initial_value = torch.as_tensor(0.0)
        self.learnable = learnable
        self.scalar = nn.Parameter(initial_value, requires_grad=self.learnable)
        self.scalar_map = nn.Identity() if scalar_map is None else scalar_map
        self.module = module

    @signature("(..., *xs) -> (..., *xs)")
    def forward(self, x: Tensor) -> Tensor:
        return self.scalar_map(self.scalar) * self.module(x)


class ReZeroResNet(nn.ModuleList):
    r"""A Residual Network with ReZero scalars."""

    @property
    def config(self) -> dict:
        return {"modules": list(self)}

    def __init__(self, modules: Iterable[nn.Module]) -> None:
        module_list = list(modules)

        for i, module in enumerate(module_list):
            # pass if already a ReZeroCell
            if isinstance(module, ReZero):
                continue
            module_list[i] = ReZero(module)

        super().__init__(module_list)

    def forward(self, x: Tensor) -> Tensor:
        for block in self:
            x = x + block(x)
        return x
