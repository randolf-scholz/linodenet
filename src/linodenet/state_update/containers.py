r"""Containers for sequential application of filters."""

__all__ = [
    "UpdateList",
    "UpdateSequence",
    "ResidualUpdateSequence",
]

from abc import abstractmethod
from collections.abc import Iterable

import torch
from torch import Tensor, nn

from linodenet.nn import ModuleSequence
from signatures import signature

from .base import StateUpdater, StateUpdaterBase


class UpdateList[C: StateUpdaterBase](StateUpdaterBase, ModuleSequence[C]):
    r"""Base class for `nn.ModuleList` of state updaters that is itself a state updater.

    Note: This class takes care of tricky multiple inheritance issues with nn.Module.
    """

    def __init__(
        self, modules: Iterable[C] = (), /, *, input_size: int, hidden_size: int
    ) -> None:
        # ⚠️ multiple inheritance ⚠️
        # due to how nn.Module.__init__ works, it should only be ever called once
        # because it will overwrite internal state otherwise.
        # Therefore, we need to carefully manually reproduce the __init__ logic here.
        assert not hasattr(self, "_modules"), f"Module already initialized: {self}"
        ModuleSequence[C].__init__(self, modules)
        # Note: Need to call StateUpdater.__init__, not StateUpdaterBase.__init__
        #   Otherwise nn.Module.__init__ gets called twice!
        StateUpdater.__init__(self, input_size, hidden_size)

    @abstractmethod
    def forward(self, y_obs: Tensor, x_hat: Tensor, /) -> Tensor: ...


class UpdateSequence[C: StateUpdaterBase](UpdateList[C]):
    r"""Apply multiple state updaters sequentially.

    .. math:: xₖ₊₁ = Fₖ(y, xₖ)
    """

    def __init__(self, modules: Iterable[C] = (), /) -> None:
        cells = list(modules)

        if not cells:
            raise ValueError("At least one module must be given!")

        input_size = cells[0].input_size
        hidden_size = cells[0].hidden_size
        for module in cells:
            if module.input_size != input_size:
                raise ValueError(
                    "All modules must have the same input_size!"
                    f"Expected {input_size}, but {module=} has {module.input_size}"
                )
            if module.hidden_size != hidden_size:
                raise ValueError(
                    "All modules must have the same hidden_size!"
                    f"Expected {hidden_size}, but {module=} has {module.hidden_size}"
                )

        super().__init__(modules, input_size=input_size, hidden_size=hidden_size)

    @signature("[(..., m), (..., n)] -> (..., n)")
    def forward(self, y: Tensor, x: Tensor) -> Tensor:
        for cell in self:
            x = cell(y, x)
        return x


class ResidualUpdateSequence[C: StateUpdaterBase](UpdateSequence[C]):
    r"""Sequential state updater with residual connections.

    .. math:: xₖ₊₁ = xₖ + αₖ⋅Fₖ(y, xₖ)

    Args:
        modules: An iterable of state updater modules to be applied sequentially.
        use_rezero: Whether to use rezero (default: True)

    A regular ResNet is obtained by setting all αₖ=1.0 and making them non-learnable.
    """

    alpha: Tensor
    r"""PARAM: The residual scaling factors αₖ."""
    use_rezero: Tensor
    r"""Whether to use rezero"""

    def __init__(
        self,
        modules: Iterable[C] = (),
        /,
        *,
        use_rezero: bool = True,
    ) -> None:
        super().__init__(modules)
        self.alpha = nn.Parameter(
            torch.zeros(len(self)) if use_rezero else torch.ones(len(self)),
            requires_grad=use_rezero,
        )

    @signature("[(..., m), (..., n)] -> (..., n)")
    def forward(self, y: Tensor, x: Tensor) -> Tensor:
        for alpha, cell in zip(self.alpha, self, strict=True):
            x = x.addcmul(alpha, cell(y, x))  # xₖ₊₁ <- xₖ + αₖfₖ(y, xₖ)
        return x
