r"""Shared protocols and base classes for dynamical flows."""

__all__ = [
    # protocols
    "Flow",
    "ContinuousFlow",
    "DiscreteFlow",
    # base classes
    "FlowBase",
    "DiscreteFlowBase",
    "ContinuousFlowBase",
]

from abc import abstractmethod
from typing import Protocol

from torch import Tensor

from linodenet.state_propagation.base import Propagator, PropagatorBase
from signatures import signature


class Flow[State: Tensor | tuple[Tensor, ...]](Propagator, Protocol):
    r"""Flows are propagators satisfying a semigoup property.

    .. math::
        Φ(0, x) = x
        Φ(δ₁ + δ₂, x) = Φ(δ₁, Φ(δ₂, x))
    """

    @signature("[(..., $n_deltas), (..., *ds)] -> (..., $n_deltas, *ds)")
    def __call__(self, delta: Tensor, state: State, /) -> State:
        r"""Propagate the system state for the requested deltas."""
        ...


class DiscreteFlow[State: Tensor | tuple[Tensor, ...]](Flow, Protocol):
    r"""Protocol for discrete-time state evolution."""

    @signature("[(..., $n_steps), (..., *ds)] -> (..., $n_steps, *ds)")
    def __call__(self, num_steps: Tensor, state: State, /) -> State:
        r"""Propagate the system for the requested step counts."""
        ...


class ContinuousFlow[State: Tensor | tuple[Tensor, ...]](Flow, Protocol):
    r"""Protocol for continuous-time state evolution.

    The first argument contains one or more time deltas at which the evolved state
    should be evaluated.
    """

    @signature("[(..., $deltas), (..., *ds)] -> (..., $deltas, *ds)")
    def __call__(self, timedeltas: Tensor, state: State, /) -> State:
        r"""Propagate the system for the requested time deltas."""
        ...


class FlowBase[State: Tensor | tuple[Tensor, ...]](PropagatorBase):
    r"""Abstract base class for time-indexed state evolution operators."""

    @abstractmethod
    @signature("[(..., $deltas), (..., *ds)] -> (..., $deltas, *ds)")
    def forward(self, delta: Tensor, state: State, /) -> State:
        r"""Propagate the system state for the requested deltas."""
        ...


class DiscreteFlowBase[State: Tensor | tuple[Tensor, ...]](FlowBase):
    r"""Abstract base class for discrete-time state evolution."""

    @abstractmethod
    @signature("[(..., $n_steps), (..., *ds)] -> (..., $n_steps, *ds)")
    def forward(self, num_steps: Tensor, state: State, /) -> State:
        r"""Propagate the system for the requested step counts."""
        ...


class ContinuousFlowBase[State: Tensor | tuple[Tensor, ...]](FlowBase):
    r"""Abstract base class for continuous-time state evolution."""

    @abstractmethod
    @signature("[(..., $deltas), (..., *ds)] -> (..., $deltas, *ds)")
    def forward(self, timedeltas: Tensor, state: State, /) -> State:
        r"""Propagate the system for the requested time deltas."""
        ...
