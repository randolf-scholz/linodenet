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
    "AbstractFilter",
    "AbstractCell",
    "Cell",
    "CellBase",
    "CellList",
    "Filter",
    "FilterBase",
    "FilterList",
]

from abc import abstractmethod
from collections.abc import Iterable
from typing import Final, Protocol, runtime_checkable

from torch import Tensor, nn

from linodenet.layers.containers import ModuleSequence


@runtime_checkable
class AbstractCell[X, Y](Protocol):
    r"""Abstract Protocol for cells.

    Currently unused and only included for documentation purposes.
    The cells we consider in practice take Tensors as inputs and outputs.
    In principle, however, one could consider more general types.

    .. math::  x' = F(y, x)
    """

    def __call__(self, y: Y, x: X, /) -> X: ...


@runtime_checkable
class AbstractFilter[Y](Protocol):
    r"""Abstract Protocol for filters.

    Currently unused and only included for documentation purposes.
    The filters we consider in practice take Tensors as inputs and outputs.
    In principle, however, one could consider more general types.

    .. math::  y' = F(y_obs, y_pred)
    """

    def __call__(self, y_obs: Y, y_pred: Y, /) -> Y: ...


@runtime_checkable
class Cell(AbstractCell[Tensor, Tensor], Protocol):
    r"""Protocol for cells.

    .. math::  x' = F(y, x)
    """

    input_size: Final[int]  # type: ignore[misc]
    hidden_size: Final[int]  # type: ignore[misc]

    def __init__(self, /, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)

    def __call__(self, y: Tensor, x: Tensor, /) -> Tensor: ...


@runtime_checkable
class Filter(AbstractFilter[Tensor], Protocol):
    r"""Protocol for filters.

    .. math::  y' = F(y_obs, y_pred)

    Note: Every Filter is a Cell with hidden_size = input_size.
    """

    input_size: Final[int]  # type: ignore[misc]
    hidden_size: Final[int]  # type: ignore[misc]

    def __init__(self, /, input_size: int) -> None:
        super().__init__()
        self.input_size = int(input_size)
        self.hidden_size = int(input_size)

    def __call__(self, y_obs: Tensor, y_pred: Tensor, /) -> Tensor: ...


# @implements(Cell[Tensor, Tensor])
class CellBase(nn.Module):
    r"""Base class for filter-cells.

    This base class is specialized to the case when X=Y=Tensor, and the arguments
    are vectors.

    .. math::  x' = F(y, x)
    .. Signature: ``[(..., d), (..., h)] -> (..., h)``.
    """

    input_size: Final[int]
    r"""CONST: The size of the observable $y$."""
    hidden_size: Final[int]
    r"""CONST: The size of the hidden state $x$."""

    def __init__(
        self,
        /,
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
            x: The current estimation of the hidden state of the system.

        Returns:
            The updated state of the system.
        """
        ...


# @implements(Filter[Tensor])
class FilterBase(CellBase):
    r"""Base class for filters.

    This base class is specialized to the case when X=Y=Tensor, and the arguments
    are vectors.

    .. math::  y' = F(y_obs, y_pred)
    .. Signature: ``[(..., d), (..., d)] -> (..., d)``.

    Where $x$ is the current state of the system, $y$ is the current measurement, and
    $x'$ is the new state of the system. $ϕ$ is a function that maps the measurement
    to the state of the system. $h$ is a function that maps the current state of the
    system to the measurement.

    Or multiple blocks of said form. In particular, we are interested in Filters
    satisfying the idempotence property: if $y=h(x)$, then $x'=x$.
    """

    def __init__(self, /, input_size: int) -> None:
        super().__init__(input_size=input_size, hidden_size=input_size)

    @abstractmethod
    def forward(self, y_obs: Tensor, y_hat: Tensor, /) -> Tensor:
        r"""Forward pass of the filter.

        Args:
            y_obs: The current measurement of the system.
            y_hat: The current estimation of the state of the system.

        Returns:
            The updated state of the system.
        """
        ...


class CellList[C: CellBase](CellBase, ModuleSequence[C]):
    r"""Base class for sequential Cells."""

    def __init__(
        self, modules: Iterable[CellBase] = (), /, *, input_size: int, hidden_size: int
    ) -> None:
        # ⚠️ multiple inheritance ⚠️
        # due to how nn.Module.__init__ works, it should only be ever called once
        # because it will overwrite internal state otherwise.
        # Therefore, we need to carefully manually reproduce the __init__ logic here.
        super(ModuleSequence, self).__init__(modules)
        self.input_size = int(input_size)  # type: ignore[misc] # pyright: ignore[reportGeneralTypeIssues]
        self.hidden_size = int(hidden_size)  # type: ignore[misc] # pyright: ignore[reportGeneralTypeIssues]

    @abstractmethod
    def forward(self, y_obs: Tensor, x_hat: Tensor, /) -> Tensor: ...


class FilterList[F: FilterBase](FilterBase, ModuleSequence[F]):
    r"""Multiple Filters passes applied sequentially.

    .. math:: xₖ₊₁ = Fₖ(y, xₖ)
    """

    HP: dict = {
        "__name__": __qualname__,
        "__module__": __name__,
        "input_size": None,
        "layers": [],
    }
    r"""The HyperparameterDict of this class."""

    def __init__(
        self, modules: Iterable[FilterBase] = (), /, *, input_size: int
    ) -> None:
        # ⚠️ multiple inheritance ⚠️
        # due to how nn.Module.__init__ works, it should only be ever called once
        # because it will overwrite internal state otherwise.
        # Therefore, we need to carefully manually reproduce the __init__ logic here.
        super(ModuleSequence, self).__init__(modules)
        self.input_size = int(input_size)  # type: ignore[misc] # pyright: ignore[reportGeneralTypeIssues]
        self.hidden_size = int(input_size)  # type: ignore[misc] # pyright: ignore[reportGeneralTypeIssues]

    @abstractmethod
    def forward(self, y_obs: Tensor, y_hat: Tensor, /) -> Tensor: ...
