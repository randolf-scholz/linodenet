from collections.abc import Callable, Iterable, Mapping
from math import sqrt
from typing import Any, Final, Optional, Self, cast

import torch
from torch import Tensor, nn

from linodenet.activations import get_activation
from linodenet.constants import EMPTY_MAP
from linodenet.filters.base import CellBase, Filter, FilterBase


class ResidualCell(CellBase):
    r"""Non-Linear RNN Cell that performs a residual update.

    .. math:: x' = x - F⋅φ(Hy - x)

    Where $F$ is a learnable square matrix, and $H$ is either a learnable matrix or
    a fixed matrix and $φ$ is a non-linear activation function, applied element-wise.
    """

    def __init__(
        self,
        /,
        input_size: int,
        hidden_size: int,
        *,
        activation: Callable[[Tensor], Tensor] | str = "tanh",
    ) -> None:
        super().__init__(input_size=input_size, hidden_size=hidden_size)
        m = self.hidden_size
        n = self.input_size

        self.activation = get_activation(activation)
        self.F = nn.Parameter(torch.normal(0, 1 / sqrt(m), size=(m, m)))
        self.H = nn.Parameter(torch.normal(0, 1 / sqrt(n), size=(m, n)))

    def forward(self, y: Tensor, x: Tensor) -> Tensor:
        r = torch.einsum("ij, ...i -> ...j", self.H, y) - x
        return x - torch.einsum("ij, ...i -> ...j", self.F, r)


class ResNetFilter(nn.ModuleList):
    r"""Sequential Filter with residual connections.

    .. math:: xₖ₊₁ = xₖ + Fₖ(y, xₖ)
    """

    # CONSTANTS
    input_size: Final[int]
    r"""The size of the observable $y$."""
    hidden_size: Final[int]
    r"""The size of the hidden state $x$."""

    HP: dict = {
        "__name__": __qualname__,
        "__module__": __name__,
        "input_size": None,
        "hidden_size": None,
        "layers": [],
    }
    r"""The HyperparameterDict of this class."""

    @classmethod
    def from_config(cls, **kwargs: Any) -> Self:
        raise NotImplementedError

    def __init__(self, layers: Iterable[Filter], /) -> None:
        r"""Initialize from modules."""
        module_list: list[Filter] = list(layers)
        if not module_list:
            raise ValueError("At least one module must be given!")

        input_size = int(module_list[0].input_size)
        hidden_size = int(module_list[-1].hidden_size)

        for module in module_list:
            if not isinstance(module, Filter) or not isinstance(module, nn.Module):
                raise TypeError("All modules must be Filters!")
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

        super().__init__(cast("list[nn.Module]", module_list))
        self.input_size = input_size
        self.hidden_size = hidden_size

    def forward(self, y: Tensor, x: Tensor) -> Tensor:
        r"""Signature: ``[(..., m), (..., n)] -> (..., n)``."""
        for layer in self:
            x = x + layer(y, x)
        return x


class ReZeroFilter(nn.ModuleList):
    r"""Sequential Filter with ReZero connections.

    .. math:: xₖ₊₁ = xₖ + εₖ⋅Fₖ(y, xₖ)
    """

    # CONSTANTS
    input_size: Final[int]
    r"""The size of the observable $y$."""
    hidden_size: Final[int]
    r"""The size of the hidden state $x$."""

    # Parameters
    weight: Tensor

    def __init__(self, layers: Iterable[Filter], /) -> None:
        r"""Initialize from modules."""
        # TODO: Use intersection Type Filter & nn.Module
        module_list: list[Filter] = list(layers)

        if not module_list:
            raise ValueError("At least one module must be given!")

        self.input_size = int(module_list[0].input_size)
        self.hidden_size = int(module_list[-1].hidden_size)

        for module in module_list:
            if module.input_size != self.input_size:
                raise ValueError(
                    "All modules must have the same input_size!"
                    f"Expected {self.input_size}, but {module=} has {module.input_size}"
                )
            if module.hidden_size != self.hidden_size:
                raise ValueError(
                    "All modules must have the same hidden_size!"
                    f"Expected {self.hidden_size}, but {module=} has {module.hidden_size}"
                )
            assert isinstance(module, nn.Module)

        super().__init__(cast("list[nn.Module]", module_list))
        # add the weight last.
        self.weight = nn.Parameter(torch.zeros(len(self)))

    def forward(self, y: Tensor, x: Tensor) -> Tensor:
        r"""Signature: ``[(..., m), (..., n)] -> (..., n)``."""
        for w, layer in zip(self.weight, self, strict=True):
            x = x + w * layer(y, x)
        return x


class ResidualFilter(FilterBase):
    r"""Wraps an existing Filter to return the residual $x' = x - η⋅F(y，x)$.

    Attributes:
        input_size: The size of the observable $y$.
        hidden_size: The size of the hidden state $x$.
        filter (Filter): The wrapped Filter.
        decoder (Optional[nn.Module]): The observation model.
    """

    # SUBMODULES
    filter: Filter
    r"""The wrapped Filter."""
    decoder: Optional[nn.Module]
    r"""The observation model."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        *,
        filter_type: type[Filter],
        filter_kwargs: Mapping[str, Any] = EMPTY_MAP,
    ) -> None:
        super().__init__(input_size=input_size, hidden_size=hidden_size)
        options = dict(filter_kwargs) | {
            "input_size": input_size,
            "hidden_size": hidden_size,
        }
        self.filter = filter_type(**options)
        self.decoder = getattr(self.filter, "decoder", None)

    def forward(self, y: Tensor, x: Tensor) -> Tensor:
        r"""Signature: ``[(..., m), (..., n)] -> (..., n)``."""
        return x - self.filter(y, x)
