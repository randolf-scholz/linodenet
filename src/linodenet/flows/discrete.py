r"""Protocols and base classes for discrete-time flows."""

__all__ = [
    "DiscreteFlow",
    "DiscreteFlowBase",
]

from abc import abstractmethod
from typing import Protocol

from torch import Tensor

from signatures import signature

from .base import Flow, FlowBase


class DiscreteFlow(Flow, Protocol):
    r"""Protocol for discrete-time state evolution."""

    @signature("[(..., $n_steps), (..., *ds)] -> (..., $n_steps, *ds)")
    def __call__(self, num_steps: int | Tensor, state: Tensor, /) -> Tensor:
        r"""Propagate the system for the requested step counts."""
        ...


class DiscreteFlowBase(FlowBase):
    r"""Abstract base class for discrete-time state evolution."""

    @abstractmethod
    @signature("[(..., $n_steps), (..., *ds)] -> (..., $n_steps, *ds)")
    def forward(self, num_steps: Tensor, state: Tensor, /) -> Tensor:
        r"""Propagate the system for the requested step counts."""
        ...
