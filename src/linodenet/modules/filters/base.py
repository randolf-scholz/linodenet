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
    "Cell",
    "Filter",
    "ProbabilisticFilter",
    "FilterABC",
    # Classes
    "MissingValueFilter",
]

from abc import abstractmethod
from collections.abc import Mapping

import torch
from torch import Tensor, nn
from torch.distributions import Distribution
from typing_extensions import Any, Final, Optional, Protocol, Self, runtime_checkable

from linodenet.constants import EMPTY_MAP


@runtime_checkable
class Cell(Protocol):
    """Protocol for cells."""

    input_size: Final[int]  # type: ignore[misc]
    """The size of the observable $y$."""
    hidden_size: Final[int]  # type: ignore[misc]
    """The size of the hidden state $x$."""

    @abstractmethod
    def __init__(self, /, input_size: int, hidden_size: int) -> None:
        """Initialize the cell."""
        ...

    @abstractmethod
    def __call__(self, y: Tensor, x: Tensor, /) -> Tensor:
        """Forward pass of the filter $x' = F(y，x)$.

        .. Signature: ``[(..., n), (..., m)] -> (..., n)``.
        """
        ...


@runtime_checkable
class Filter(Protocol):
    """Protocol for filter.

    Additionally to the `Cell` protocol, a filter has knowledge of the observation model.
    This is a decoder that maps the hidden state to the observation space.

    .. math:: y = h(x)
    """

    # CONSTANTS
    input_size: Final[int]  # type: ignore[misc]
    """The size of the observable $y$."""
    hidden_size: Final[int]  # type: ignore[misc]
    """The size of the hidden state $x$."""

    # SUBMODULES
    decoder: nn.Module
    """The observation model."""

    @abstractmethod
    def __call__(self, y: Tensor, x: Tensor, /) -> Tensor:
        """Forward pass of the filter $x' = F(x, y)$.

        .. Signature: ``[(..., n), (..., m)] -> (..., n)``.
        """
        ...


@runtime_checkable
class ProbabilisticFilter(Protocol):
    """Protocol for probabilistic filters.

    The goal of a probabilistic filter is to update the distribution of the hidden state,
    given the current observation. This is done by updating the parameters of the distribution
    that represents the state.

    In practice, this is done by acting on the parameters that define the distribution model,
    rather than in function space itself.
    """

    decoder: Distribution
    """The model for the conditional distribution $p(y|x)$."""

    @abstractmethod
    def __call__(self, y: Distribution, x: Distribution, /) -> Distribution:
        """Forward pass of the filter $p' = F(q, p)$."""
        ...


class FilterABC(nn.Module):
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

    input_size: int
    """The size of the observable $y$."""
    hidden_size: int
    """The size of the hidden state $x$."""

    @abstractmethod
    def forward(self, y: Tensor, x: Tensor) -> Tensor:
        r"""Forward pass of the filter.

        Args:
            y: The current measurement of the system.
            x: The current estimation of the state of the system.

        Returns:
            x̂: The updated state of the system.
        """
        ...


class MissingValueCell(nn.Module):
    """Wraps an existing Cell to impute missing value.

    .. math:: x' = F([m ? s : y]，x)

    where $s$ is the substitute value.

    There are the following strategies for imputation:

    - "zero": Replace missing values with zeros.
    - "last": Replace missing values with the last observed value. (initialized with zero)
    """

    # CONSTANTS
    input_size: Final[int]
    """CONST: The size of the observable $y$."""
    hidden_size: Final[int]
    """CONST: The size of the hidden state $x$."""
    concat_mask: Final[bool]
    r"""CONST: Whether to concatenate the mask to the input or not."""
    imputation_strategy: Final[str]
    r"""CONST: The strategy to use for imputation."""

    # BUFFERS
    S: Tensor
    r"""BUFFER: The substitute."""

    @classmethod
    def from_cell(cls, cell: Cell, *, imputation_strategy: str = "zero") -> Self:
        r"""Initialize from a cell."""
        return cls(
            input_size=cell.input_size,
            hidden_size=cell.hidden_size,
            cell_type=cell,
            concat_mask=False,
            imputation_strategy=imputation_strategy,
        )

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        *,
        imputation_strategy: str = "zero",
        concat_mask: bool = True,
        cell_type: str | type[Cell] | Cell = "GRUCell",
        cell_kwargs: Mapping[str, Any] = EMPTY_MAP,
    ) -> None:
        super().__init__()
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.imputation_strategy = bool(imputation_strategy)
        self.concat_mask = bool(concat_mask)
        self.register_buffer("S", torch.tensor(0.0))

        # initialize cell
        options = dict(cell_kwargs)
        cell_input_size = self.input_size * (1 + self.concat_mask)
        options.update(input_size=cell_input_size, hidden_size=self.hidden_size)
        match cell_type:
            case Cell() as cell if isinstance(cell_type, nn.Module):
                self.cell = cell
            case type() as cls if issubclass(cls, nn.Module):
                self.cell = cls(**options)
            case str() as cell_name:
                cls = getattr(nn, cell_name)
                self.cell = cls(**options)
            case _:
                raise TypeError(f"Invalid cell type: {cell_type}")

    def forward(self, y: Tensor, x: Tensor) -> Tensor:
        r"""Signature: ``[(..., m), (..., n)] -> (..., n)``."""
        # impute missing value in observation with state estimate
        mask = torch.isnan(y)

        if self.imputation_strategy == "last":
            # update entries in S with observed values
            self.S = torch.where(mask, self.S, y)

        y = torch.where(mask, self.S, y)

        if self.concat_mask:
            y = torch.cat([y, mask], dim=-1)

        return self.cell(y, x)


class MissingValueFilter(nn.Module):
    r"""Wraps an existing Filter to impute missing values via the observation model.

    Given a filter $F$ with observation model $h$, this returns the filter defined by

    .. math:: x' = F([m ? h(x) : y]，x)
    """

    # CONSTANTS
    input_size: Final[int]
    """CONST: The size of the observable $y$."""
    hidden_size: Final[int]
    """CONST: The size of the hidden state $x$."""
    concat_mask: Final[bool]
    r"""CONST: Whether to concatenate the mask to the input or not."""

    # BUFFERS
    ZERO: Tensor
    """A buffer for the zero tensor."""

    HP = {
        "__name__": __qualname__,
        "__module__": __name__,
        "concat_mask": True,
        "input_size": None,
        "hidden_size": None,
        "autoregressive": True,
        "Cell": {
            "__name__": "GRUCell",
            "__module__": "torch.nn",
            "input_size": None,
            "hidden_size": None,
            "bias": True,
            "device": None,
            "dtype": None,
        },
    }
    r"""The HyperparameterDict of this class."""

    @classmethod
    def from_filter(cls, filter: Filter) -> Self:
        r"""Initialize from a filter."""
        return cls(
            input_size=filter.input_size,
            hidden_size=filter.hidden_size,
            concat_mask=False,
            filter_type=filter,
        )

    def __init__(
        self,
        input_size: Optional[int] = None,
        hidden_size: Optional[int] = None,
        *,
        concat_mask: bool = True,
        filter_type: str | type[Filter] | Filter = NotImplemented,
        filter_kwargs: Mapping[str, Any] = EMPTY_MAP,
    ) -> None:
        super().__init__()
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.concat_mask = bool(concat_mask)
        self.register_buffer("ZERO", torch.tensor(0.0))

        # initialize cell
        options = dict(filter_kwargs)
        filter_input_size = self.input_size * (1 + self.concat_mask)
        options.update(input_size=filter_input_size, hidden_size=self.hidden_size)
        match filter_type:
            case Cell() as cell if isinstance(filter_type, nn.Module):
                self.filter = cell
            case type() as cls if issubclass(cls, nn.Module):
                self.filter = cls(**options)
            case str() as filter_name:
                cls = getattr(nn, filter_name)
                self.filter = cls(**options)
            case _:
                raise TypeError(f"Invalid filter type: {filter_type}")

        self.decoder = self.filter.decoder

    def forward(self, y: Tensor, x: Tensor) -> Tensor:
        r"""Signature: ``[(..., m), (..., n)] -> (..., n)``."""
        mask = torch.isnan(y)
        y = torch.where(mask, self.decoder(x), y)

        if self.concat_mask:
            y = torch.cat([y, mask], dim=-1)

        # Unflatten return value
        return self.filter(y, x)


class ResidualCell(nn.Module):
    """Wraps an existing Cell with a residual connection."""
