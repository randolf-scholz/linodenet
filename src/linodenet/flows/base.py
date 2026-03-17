r"""Shared protocols and base classes for dynamical flows."""

__all__ = [
    "Flow",
    "FlowBase",
]

from abc import abstractmethod
from typing import Final, Protocol

from torch import Tensor, nn

from signatures import signature


class Flow(Protocol):
    r"""Protocol for time-indexed state evolution operators."""

    input_shape: Final[tuple[int, ...]]  # type: ignore[misc]
    r"""CONST: The dimensionality of inputs."""

    @signature("[(..., $n_deltas), (..., *ds)] -> (..., $n_deltas, *ds)")
    def __call__(self, delta: Tensor, state: Tensor, /) -> Tensor:
        r"""Propagate the system state for the requested deltas."""
        ...


class FlowBase(nn.Module):
    r"""Abstract base class for time-indexed state evolution operators."""

    @abstractmethod
    @signature("[(..., $deltas), (..., *ds)] -> (..., $deltas, *ds)")
    def forward(self, delta: Tensor, state: Tensor, /) -> Tensor:
        r"""Propagate the system state for the requested deltas."""
        ...
