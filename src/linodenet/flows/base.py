r"""Models for the latent dynamical system."""

__all__ = [
    # ABCs & Protocols
    "ContinuousFlow",
    "DiscreteFlow",
    "Flow",
    "FlowBase",
]

from abc import abstractmethod
from typing import Final, Protocol

from torch import Tensor, nn

from signatures import signature


class Flow(Protocol):
    r"""Protocol for dynamical flows."""

    input_shape: Final[tuple[int, ...]]  # type: ignore[misc]
    r"""CONST: The dimensionality of inputs."""

    @signature("[(..., $n_steps), (..., *ds)] -> (..., $n_steps, *ds)")
    def __call__(self, delta: Tensor, state: Tensor, /) -> Tensor:
        r"""Propagate the system state."""
        ...


class ContinuousFlow(Flow, Protocol):
    r"""Protocol for continuous-time flows.

    Note: in practice we may want a solve_ivp like interface instead:
    - y0: initial state
    - t0: initial time
    - t_eval: time steps to evaluate at

    Some libraries use t_eval[0] as t0.
    Some libraries also need a t_1.

    scipy: y0 + t_span + t_eval
    torchdiffeq: y0 + t_eval (t_eval[0] is t, no t1)
    diffrax: t0 + t1 + y0 + t_eval (called saveat)
    sdepy: y0 + (t0=0 implicit) + t_eval
    """

    @signature("[(..., $n_steps), (..., *ds)] -> (..., $n_steps, *ds)")
    def __call__(self, timedelta: float | Tensor, state: Tensor, /) -> Tensor:
        r"""Propagate the system for time-step `dt`."""
        ...


class DiscreteFlow(Flow, Protocol):
    r"""Protocol for discrete-time flows."""

    @signature("[(..., $n_steps), (..., *ds)] -> (..., $n_steps, *ds)")
    def __call__(self, num_steps: int | Tensor, state: Tensor, /) -> Tensor:
        r"""Propagate the system for `num_steps`.

        .. math:: step(𝐤, x₀) = (x(k₁), … x(kₙ))
        """
        ...


class FlowBase(nn.Module):
    r"""Abstract Base Class for dynamical flows."""

    input_shape: Final[tuple[int, ...]]  # type: ignore[misc]  # pyright: ignore[reportGeneralTypeIssues]
    r"""CONST: The dimensionality of inputs."""

    @abstractmethod
    @signature("[(..., $n_steps), (..., *ds)] -> (..., $n_steps, *ds)")
    def forward(self, delta: Tensor, state: Tensor, /) -> Tensor:
        r"""Forward pass of the system.

        Args:
            delta: The time-step to advance the system.
            state: The state estimate at time t.

        Returns:
            The updated state of the system at time t+∆t.
        """
        ...
