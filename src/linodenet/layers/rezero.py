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
from typing import Final, Optional

import torch
from torch import Tensor, jit, nn

from linodenet.signatures import signature


class ReZero(nn.Module):
    r"""ReZero module.

    Simply multiplies the inputs by a scalar initialized to zero.
    """

    HP = {
        "__name__": __qualname__,
        "__module__": __name__,
    }
    r"""The hyperparameter dictionary"""

    # CONSTANTS
    learnable: Final[bool]
    r"""CONST: Whether the scalar is learnable."""

    # PARAMETERS
    scalar: Tensor
    r"""The scalar to multiply the inputs by."""

    def __init__(
        self,
        module: Optional[nn.Module] = None,
        *,
        scalar: Optional[Tensor] = None,
        learnable: bool = True,
    ) -> None:
        super().__init__()
        initial_value = torch.as_tensor(0.0 if scalar is None else scalar)
        self.scalar = nn.Parameter(initial_value) if self.learnable else initial_value
        self.learnable = learnable
        self.module = module

    @jit.export
    @signature("(..., *xs) -> (..., *xs)")
    def forward(self, x: Tensor) -> Tensor:
        if self.module is None:
            return self.scalar * x
        return self.scalar * self.module(x)


class ReZeroResNet(nn.ModuleList):
    r"""A Residual Network with ReZero scalars."""

    HP = {
        "__name__": __qualname__,
        "__module__": __name__,
    }
    r"""The hyperparameter dictionary"""

    def __init__(self, modules: Iterable[nn.Module]) -> None:
        module_list = list(modules)

        for i, module in enumerate(module_list):
            # pass if already a ReZeroCell
            if isinstance(module, ReZero):
                continue
            module_list[i] = ReZero(module)

        super().__init__(module_list)

    @jit.export
    def forward(self, x: Tensor) -> Tensor:
        for block in self:
            x = x + block(x)
        return x
