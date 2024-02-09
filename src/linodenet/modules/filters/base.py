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
from collections.abc import Callable, Mapping

import torch
from torch import Tensor, jit, nn
from torch.distributions import Distribution
from typing_extensions import Any, Final, Optional, Protocol, runtime_checkable

from linodenet.constants import EMPTY_MAP


@runtime_checkable
class Cell(Protocol):
    """Protocol for cells."""

    input_size: Final[int]  # type: ignore[misc]
    """The size of the observable $y$."""
    hidden_size: Final[int]  # type: ignore[misc]
    """The size of the hidden state $x$."""

    def __init__(self, /, input_size: int, hidden_size: int) -> None:
        """Initialize the cell."""
        ...

    def __call__(self, y: Tensor, x: Tensor, /) -> Tensor:
        """Forward pass of the filter $x' = F(x, y)$.

        .. Signature: ``[(..., n), (..., m)] -> (..., n)``.
        """
        ...


@runtime_checkable
class Filter(Protocol):
    """Protocol for filter.

    Additionally to the `Cell` protocol, a filter has knowledge of the observation model.
    This is a decoder that maps the hidden state to the observation space.
    """

    # CONSTANTS
    input_size: Final[int]  # type: ignore[misc]
    """The size of the observable $y$."""
    hidden_size: Final[int]  # type: ignore[misc]
    """The size of the hidden state $x$."""
    direct_observation: Final[bool]  # type: ignore[misc]
    """Whether the observation model is the identity or not."""

    # SUBMODULES
    decoder: nn.Module
    """The observation model."""

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


class MissingValueFilter(nn.Module):
    r"""Wraps an existing Filter to impute missing values via the observation model.

    Given a filter $F$ and an observation model $h$, this returns the filter defined by

    .. math:: x' = F( [m ? h(x) : y]，x)
    """

    # CONSTANTS
    input_size: Final[int]
    """The size of the observable $y$."""
    hidden_size: Final[int]
    """The size of the hidden state $x$."""
    direct_observation: Final[bool]
    r"""CONST: Whether the filter is autoregressive or not."""
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

    def __init__(
        self,
        cell: Cell = NotImplemented,
        /,
        input_size: Optional[int] = None,
        hidden_size: Optional[int] = None,
        *,
        decoder: str | Callable[[Tensor], Tensor] = "zero",
        concat_mask: bool = True,
        cell_type: type[Cell] = nn.GRUCell,
        cell_kwargs: Mapping[str, Any] = EMPTY_MAP,
    ) -> None:
        super().__init__()
        self.concat_mask = concat_mask
        self.register_buffer("ZERO", torch.tensor(0.0))
        self._decoder = self.zero if decoder is None else decoder

        if cell is NotImplemented:
            # create configuration
            options = dict(cell_kwargs)
            if input_size is not None:
                assert "input_size" not in options
                options["input_size"] = (1 + concat_mask) * input_size
            if hidden_size is not None:
                assert "hidden_size" not in options
                options["hidden_size"] = hidden_size

            # attempt to initialize cell from hyperparameters
            try:
                self.cell = cell_type(**options)
            except TypeError as exc:
                raise RuntimeError(
                    f"Failed to initialize cell of type {cell_type} with inputs {options}"
                ) from exc
        else:
            self.cell = cell
            self.input_size = input_size if input_size is not None else cell.input_size
            self.hidden_size = (
                hidden_size if hidden_size is not None else cell.hidden_size
            )

        match decoder:
            case "zero":
                self.decoder = Constant(self.zero)
            case "linear":
                self.decoder = nn.Linear(self.hidden_size, self.input_size)
            case Callable() as decoder_func:
                self.decoder = decoder_func
            case _:
                raise TypeError(f"Invalid decoder type: {decoder}")

    @jit.export
    def forward(self, y: Tensor, x: Tensor) -> Tensor:
        r"""Signature: ``[(..., m), (..., n)] -> (..., n)``."""
        # impute missing value in observation with state estimate
        mask = torch.isnan(y)
        y = torch.where(mask, self.decoder(x), y)

        if self.concat_mask:
            y = torch.cat([y, mask], dim=-1)

        # Flatten for RNN-Cell
        y = y.view(-1, y.shape[-1])
        x = x.view(-1, x.shape[-1])

        # Apply filter
        result = self.cell(y, x)

        # Unflatten return value
        return result.view(mask.shape)
