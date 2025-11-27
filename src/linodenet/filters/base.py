r"""Different Filter models to be used in conjunction with LinODENet.

We distinguish between two main types of modules: Cells and Filters.

- A cell is anything that operationally is similar to torch.nn.RNNCell:
  - `__init__` takes two positional arguments: input_size and hidden_size
  - `forward` takes two positional inputs: $y$ (the current measurement) and
    $x$ (the current estimation of the hidden state), and returns the updated state of the system.
  - For example, a linear cell is of the form: $F(y, x) = Ax + By + b$
    Note that linear here means linear-affine in both inputs jointly, not separately, e.g.
    $F(y₁+y₂, x₁+x₂)$ and $F(y₁, x₁) + F(y₂, x₂)$ are equal up to a constant.
- A filter is a special case of a cell where `input_size == hidden_size`.
  - on this case, the same transformation is applied to both the measurement and the
    current estimation of the state.

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
    "Filter",
    "FilterBase",
    # functions
    "is_filter",
]

from abc import abstractmethod
from typing import Final, Protocol, TypeIs, runtime_checkable

from torch import Tensor, nn


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
    r"""Protocol for vector valued cells.

    .. math::  x' = F(y, x)
    """

    input_size: Final[int]  # type: ignore[misc]
    hidden_size: Final[int]  # type: ignore[misc]

    def __init__(self, /, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)

    def __call__(self, y: Tensor, x: Tensor, /) -> Tensor:
        r""".. Signature: ``[(..., d), (..., h)] -> (..., h)``."""
        ...


@runtime_checkable
class Filter(AbstractFilter[Tensor], Protocol):
    r"""Protocol for vector valued filters.

    .. math::  y' = F(y_obs, y_pred)

    Note: Every Filter is a Cell with hidden_size = input_size.
    """

    input_size: Final[int]  # type: ignore[misc]
    hidden_size: Final[int]  # type: ignore[misc]

    def __init__(self, /, input_size: int) -> None:
        super().__init__()
        self.input_size = int(input_size)
        self.hidden_size = int(input_size)

    def __call__(self, y_obs: Tensor, y_pred: Tensor, /) -> Tensor:
        r""".. Signature: ``[(..., d), (..., d)] -> (..., d)``."""
        ...


class CellBase(nn.Module, Cell):
    r"""Base class for filter-cells.

    This base class is specialized to the case when X=Y=Tensor, and the arguments
    are vectors.

    .. math::  x' = F(y, x)
    .. Signature: ``[(..., d), (..., h)] -> (..., h)``.
    """

    def __init__(self, /, input_size: int, hidden_size: int) -> None:
        # ⚠️ multiple inheritance ⚠️
        assert not hasattr(self, "_modules"), f"Module already initialized: {self}"
        nn.Module.__init__(self)
        Cell.__init__(self, input_size, hidden_size)

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
        super().__init__(input_size, input_size)

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


def is_filter(arg: object) -> TypeIs[Filter]:
    r"""Check whether an object is a Filter."""
    input_size = getattr(arg, "input_size", None)
    hidden_size = getattr(arg, "hidden_size", None)
    return (
        isinstance(input_size, int)
        and isinstance(hidden_size, int)
        and input_size == hidden_size
    )
