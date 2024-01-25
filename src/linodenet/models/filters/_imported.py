"""Imported filters."""

__all__ = [
    "GRUFilter",
    "LSTMFilter",
    "RNNFilter",
]

from abc import abstractmethod
from typing import cast

from torch import Tensor, nn
from torch.nn import GRUCell, LSTMCell, RNNCell

from linodenet.models.filters.base import Filter


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


GRUFilter = cast(type[Filter], GRUCell)
"""Alias for the GRUCell filter."""
LSTMFilter = cast(type[Filter], LSTMCell)
"""Alias for the LSTMCell filter."""
RNNFilter = cast(type[Filter], RNNCell)
"""Alias for the RNNCell filter."""
