r"""Different Filter models to be used in conjunction with LinODENet.

A Filter takes two positional inputs:
- An input tensor y: the current measurement of the system
- An input tensor x: the current estimation of the state of the system

Sometimes, we have a third input, so called covariates $u$.
There are two types of covariates:

- Control inputs: These are external variables that influence the system
- Exogenous inputs: Sometimes, there are two coupled systems, and we have access to
  measurements / predictions of the other system (example: weather forecast).
  In this case we can treat these variables as part of the state.

These are external variables that influence the system,
but are not part of the state.

Example:
    The linear state space system is given by the equations (without noise):

    .. math::
        ẋ(t) &= A(t)x(t) + B(t)u(t) \\
        y(t) &= C(t)x(t) + D(t)u(t)

    Here $u$ is the control input.
"""

__all__ = [
    # ABCs & Protocols
    "Filter",
    "FilterBase",
    # Classes
    "ReZeroFilter",
    "ResNetFilter",
    "ResidualFilter",
    "SequentialFilter",
]

from abc import abstractmethod
from collections.abc import Iterable, Mapping
from typing import (
    Any,
    Final,
    Optional,
    Protocol,
    Self,
    cast,
    runtime_checkable,
)

import torch
from torch import Tensor, nn

from linodenet.constants import EMPTY_MAP
from linodenet.modules.filters.cells import Cell
from linodenet.torch_generics import ModuleSequence


@runtime_checkable
class Filter(Cell, Protocol):
    r"""Protocol for filter.

    Attributes:
        input_size: The size of the observable $y$.
        hidden_size: The size of the hidden state $x$.
    """

    @abstractmethod
    def __call__(self, y: Tensor, x: Tensor, /) -> Tensor:
        r"""Forward pass of the filter $x' = F(x, y)$.

        .. Signature: ``[(..., n), (..., m)] -> (..., n)``.
        """
        ...


class FilterBase(nn.Module):
    r"""Base class for all filters.

    All filters should have a signature of the form:

    .. math::  x' = x + ϕ(y-h(x))

    Where $x$ is the current state of the system, $y$ is the current measurement, and
    $x'$ is the new state of the system. $ϕ$ is a function that maps the measurement
    to the state of the system. $h$ is a function that maps the current state of the
    system to the measurement.

    Or multiple blocks of said form. In particular, we are interested in Filters
    satisfying the idempotence property: if $y=h(x)$, then $x'=x$.
    """

    input_size: Final[int]
    r"""The size of the observable $y$."""
    hidden_size: Final[int]
    r"""The size of the hidden state $x$."""

    def __init__(
        self,
        *,
        input_size: int,
        hidden_size: int,
    ) -> None:
        super().__init__()
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)

    @abstractmethod
    def forward(self, y: Tensor, x: Tensor, /) -> Tensor:
        r"""Forward pass of the filter.

        Args:
            y: The current measurement of the system.
            x: The current estimation of the state of the system.

        Returns:
            x̂: The updated state of the system.
        """
        ...


#  FIXME: https://github.com/python/typing/issues/213 (use intersection type)
class SequentialFilter(ModuleSequence[FilterBase], FilterBase):  # pyright: ignore[reportIncompatibleVariableOverride]
    r"""Multiple Filters passes applied sequentially.

    .. math:: xₖ₊₁ = Fₖ(y, xₖ)
    """

    HP: dict = {
        "__name__": __qualname__,
        "__module__": __name__,
        "input_size": None,
        "hidden_size": None,
        "layers": [],
    }
    r"""The HyperparameterDict of this class."""

    def __init__(self, filters: Iterable[FilterBase] = (), /) -> None:
        r"""Initialize from modules."""
        filter_list: list[Filter] = list(filters)

        if not filter_list:
            raise ValueError("At least one module must be given!")

        # ensure the input and hidden sizes are the same for all modules
        input_size = int(filter_list[0].input_size)
        hidden_size = int(filter_list[0].hidden_size)

        for module in filter_list:
            assert isinstance(module, nn.Module)
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

        FilterBase.__init__(self, input_size=input_size, hidden_size=hidden_size)
        ModuleSequence.__init__(self, filters)

    def forward(self, y: Tensor, x: Tensor) -> Tensor:
        r"""Signature: ``[(..., m), (..., n)] -> (..., n)``."""
        for layers in self:
            x = layers(y, x)
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

        super().__init__(cast(list[nn.Module], module_list))
        # add the weight last.
        self.weight = nn.Parameter(torch.zeros(len(self)))

    def forward(self, y: Tensor, x: Tensor) -> Tensor:
        r"""Signature: ``[(..., m), (..., n)] -> (..., n)``."""
        for w, layer in zip(self.weight, self, strict=True):
            x = x + w * layer(y, x)
        return x


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

        super().__init__(cast(list[nn.Module], module_list))
        self.input_size = input_size
        self.hidden_size = hidden_size

    def forward(self, y: Tensor, x: Tensor) -> Tensor:
        r"""Signature: ``[(..., m), (..., n)] -> (..., n)``."""
        for layer in self:
            x = x + layer(y, x)
        return x
