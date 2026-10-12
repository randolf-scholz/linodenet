r"""Core protocols and base classes for state propagation.

This module defines protocols for operators that evolve a system state over
one or more time deltas or discrete step counts.
"""

__all__ = [
    # protocols
    "Propagator",
    "Flow",
    "ContinuousFlow",
    "DiscreteFlow",
]

from typing import Final, Protocol

from torch import Tensor

from signatures import signature

# TODO: Use intersection types with nn.Module


class Propagator[State: Tensor | tuple[Tensor, ...]](Protocol):
    r"""Protocol for time-indexed state evolution operators."""

    input_shape: Final[tuple[int, ...]]
    r"""CONST: Shape of the state coordinates, excluding batch and delta axes."""

    def __init__(self, input_shape: tuple[int, ...]) -> None:
        self.input_shape = input_shape

    @signature("[(..., $n_deltas), (..., *ds)] -> (..., $n_deltas, *ds)")
    def __call__(self, delta: Tensor, state: State, /) -> State:
        r"""Propagate the system state for the requested deltas."""
        ...


class Flow[State: Tensor | tuple[Tensor, ...]](Propagator[State], Protocol):
    r"""Protocol for propagators satisfying the semigroup property.

    $$\begin{aligned}
        Φ(0, x) &= x, \\
        Φ(δ₁ + δ₂, x) &= Φ(δ₁, Φ(δ₂, x))
    \end{aligned}$$

    Each requested delta is measured from the supplied initial state and is
    evaluated independently.
    """

    @signature("[(..., $n_deltas), (..., *ds)] -> (..., $n_deltas, *ds)")
    def __call__(self, delta: Tensor, state: State, /) -> State:
        r"""Propagate the system state for the requested deltas."""
        ...


class DiscreteFlow[State: Tensor | tuple[Tensor, ...]](Flow[State], Protocol):
    r"""Protocol for discrete-time state evolution."""

    @signature("[(..., $n_steps), (..., *ds)] -> (..., $n_steps, *ds)")
    def __call__(self, num_steps: Tensor, state: State, /) -> State:
        r"""Propagate the system for the requested step counts."""
        ...


class ContinuousFlow[State: Tensor | tuple[Tensor, ...]](Flow[State], Protocol):
    r"""Protocol for continuous-time state evolution.

    The first argument contains one or more time deltas at which the evolved state
    should be evaluated.
    """

    @signature("[(..., $deltas), (..., *ds)] -> (..., $deltas, *ds)")
    def __call__(self, timedeltas: Tensor, state: State, /) -> State:
        r"""Propagate the system for the requested time deltas."""
        ...
