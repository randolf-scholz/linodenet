r"""Core protocols and base classes for state propagation.

This module defines the structural interface and abstract base class for
time-indexed operators that evolve a system state across one or more deltas.
"""

__all__ = ["Propagator", "PropagatorBase"]

from abc import abstractmethod
from typing import Final, Protocol

from torch import Tensor, nn

from signatures import signature


class Propagator(Protocol):
    r"""Protocol for time-indexed state evolution operators."""

    input_shape: Final[tuple[int, ...]]
    r"""CONST: The dimensionality of inputs."""

    @signature("[(..., $n_deltas), (..., *ds)] -> (..., $n_deltas, *ds)")
    def __call__(self, delta: Tensor, state: Tensor, /) -> Tensor:
        r"""Propagate the system state for the requested deltas."""
        ...


class PropagatorBase(nn.Module):
    r"""Abstract base class for time-indexed state evolution operators."""

    input_shape: Final[tuple[int, ...]]
    r"""CONST: The dimensionality of inputs."""

    def __init__(self, input_shape: tuple[int, ...]) -> None:
        super().__init__()
        self.input_shape = input_shape

    @abstractmethod
    @signature("[(..., $deltas), (..., *ds)] -> (..., $deltas, *ds)")
    def forward(self, delta: Tensor, state: Tensor, /) -> Tensor:
        r"""Propagate the system state for the requested deltas."""
        ...
