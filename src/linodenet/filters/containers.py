r"""Containers for sequential application of Cells and Filters."""

__all__ = [
    "CellSequence",
    "FilterSequence",
    "ResidualCell",
    "ResidualFilter",
]

from collections.abc import Iterable

from torch import Tensor

from linodenet.filters.base import CellBase, CellList, FilterBase, FilterList


class CellSequence[C: CellBase](CellList[C]):
    r"""Apply multiple Cells sequentially."""

    def __init__(self, modules: Iterable[CellBase] = (), /) -> None:
        cells = list(modules)

        if not cells:
            raise ValueError("At least one module must be given!")

        input_size = cells[0].input_size
        hidden_size = cells[-1].hidden_size
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

    def forward(self, y: Tensor, x: Tensor) -> Tensor:
        r"""Signature: ``[(..., m), (..., n)] -> (..., n)``."""
        for layers in self:
            x = layers(y, x)
        return x


class ResidualCell[C: CellBase](CellSequence[C]):
    def forward(self, y: Tensor, x: Tensor) -> Tensor:
        r"""Signature: ``[(..., m), (..., n)] -> (..., n)``."""
        for layer in self:
            x = x + layer(y, x)
        return x


class FilterSequence[F: FilterBase](FilterList[F]):
    r"""Apply multiple Filters sequentially."""

    def __init__(self, modules: Iterable[FilterBase] = (), /) -> None:
        filters = list(modules)

        if not filters:
            raise ValueError("At least one module must be given!")

        input_size = filters[0].input_size
        for module in filters:
            if module.input_size != input_size:
                raise ValueError(
                    "All modules must have the same input_size!"
                    f"Expected {input_size}, but {module=} has {module.input_size}"
                )

        super().__init__(modules, input_size=input_size)

    def forward(self, y_obs: Tensor, y: Tensor) -> Tensor:
        r"""Signature: ``[(..., m), (..., n)] -> (..., n)``."""
        for layer in self:
            y = layer(y_obs, y)
        return y


class ResidualFilter[F: FilterBase](FilterSequence[F]):
    r"""Sequential Filter with Residual connections.

    .. math:: xₖ₊₁ = xₖ + Fₖ(y, xₖ)
    """

    def forward(self, y_obs: Tensor, y: Tensor) -> Tensor:
        r"""Signature: ``[(..., m), (..., n)] -> (..., n)``."""
        for layer in self:
            y = y + layer(y_obs, y)
        return y
