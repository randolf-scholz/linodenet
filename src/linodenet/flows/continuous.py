r"""Protocols and base classes for continuous-time flows."""

__all__ = [
    "ContinuousFlow",
    "ContinuousFlowBase",
]

from abc import abstractmethod
from typing import Protocol

from torch import Tensor

from linodenet.flows.base import Flow, FlowBase
from signatures import signature


class ContinuousFlow(Flow, Protocol):
    r"""Protocol for continuous-time state evolution.

    The first argument contains one or more time deltas at which the evolved state
    should be evaluated.
    """

    @signature("[(..., $deltas), (..., *ds)] -> (..., $deltas, *ds)")
    def __call__(self, timedeltas: float | Tensor, state: Tensor, /) -> Tensor:
        r"""Propagate the system for the requested time deltas."""
        ...


class ContinuousFlowBase(FlowBase):
    r"""Abstract base class for continuous-time state evolution."""

    @abstractmethod
    @signature("[(..., $deltas), (..., *ds)] -> (..., $deltas, *ds)")
    def forward(self, timedeltas: Tensor, state: Tensor, /) -> Tensor:
        r"""Propagate the system for the requested time deltas."""
        ...
