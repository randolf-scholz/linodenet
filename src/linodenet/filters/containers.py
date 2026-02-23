r"""Containers for sequential application of Cells and Filters."""

__all__ = [
    "CellList",
    "CellSequence",
    "ResidualCellSequence",
]

from abc import abstractmethod
from collections.abc import Iterable

import torch
from torch import Tensor, jit, nn

from linodenet.filters.base import Cell, CellBase
from linodenet.nn import ModuleSequence
from signatures import signature


class CellList[C: CellBase](CellBase, ModuleSequence[C]):
    r"""Base class for nn.ModuleList of Cells that's a Cell itself.

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
        # Note: Need to call Cell.__init__, not CellBase.__init__
        #   Otherwise nn.Module.__init__ gets called twice!
        Cell.__init__(self, input_size, hidden_size)

    @abstractmethod
    def forward(self, y_obs: Tensor, x_hat: Tensor, /) -> Tensor: ...


class CellSequence[C: CellBase](CellList[C]):
    r"""Apply multiple Cells sequentially.

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

    @jit.export
    @signature("[(..., m), (..., n)] -> (..., n)")
    def forward(self, y: Tensor, x: Tensor) -> Tensor:
        for cell in self:
            x = cell(y, x)
        return x


class ResidualCellSequence[C: CellBase](CellSequence[C]):
    r"""Sequential Cell with Residual connections.

    .. math:: xₖ₊₁ = xₖ + αₖ⋅Fₖ(y, xₖ)

    Args:
        modules: An iterable of Cell modules to be applied sequentially.
        alpha_learnable (default=True): If True, the residual scaling factors αₖ are learnable
        alpha (default=0.0): Initial value for the residual scaling factors αₖ.

    A regular ResNet is obtained by setting all αₖ=1.0 and making them non-learnable.
    """

    alpha: Tensor
    r"""PARAM: The residual scaling factors αₖ."""

    def __init__(
        self,
        modules: Iterable[C] = (),
        /,
        *,
        alpha_learnable: bool = True,
        alpha: float | list[float] | Tensor = 0.0,
    ) -> None:
        super().__init__(modules)
        alphas = torch.as_tensor(alpha).ravel()
        num = len(self)
        if alphas.numel() == 1:
            alphas = alphas.repeat(num)
        elif alphas.numel() != num:
            raise ValueError(
                f"alpha_value must be a scalar or have length {num}, but got {alphas.shape}"
            )
        assert alphas.shape == (num,)
        self.alpha = nn.Parameter(alphas, requires_grad=alpha_learnable)

    @jit.export
    @signature("[(..., m), (..., n)] -> (..., n)")
    def forward(self, y: Tensor, x: Tensor) -> Tensor:
        for alpha, cell in zip(self.alpha, self, strict=True):
            x = x + alpha * cell(y, x)
        return x
