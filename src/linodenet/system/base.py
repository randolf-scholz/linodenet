r"""Models for the latent dynamical system."""

__all__ = [
    # ABCs & Protocols
    "ContinuousSystem",
    "DiscreteSystem",
    "SystemABC",
]

from abc import abstractmethod
from typing import Final, Protocol, runtime_checkable

from torch import Tensor, nn

from linodenet.signatures import signature


@runtime_checkable
class ContinuousSystem(Protocol):
    r"""Protocol for System Components."""

    input_size: Final[int]  # type: ignore[misc]
    r"""CONST: The dimensionality of inputs."""

    @signature("[(...), (..., d)] -> (..., d)")
    def __call__(self, dt: float | Tensor, z: Tensor, /) -> Tensor:
        r"""Propagate the system for time-step `dt`."""
        ...


class DiscreteSystem(Protocol):
    r"""Protocol for Discrete System Components."""

    input_size: Final[int]  # type: ignore[misc]
    r"""CONST: The dimensionality of inputs."""

    @signature("[int, (..., d)] -> (..., d)")
    def __call__(self, n_steps: int | Tensor, z: Tensor, /) -> Tensor:
        r"""Propagate the system for `n_steps`."""
        ...


class SystemABC(nn.Module):
    r"""Abstract Base Class for System components."""

    input_size: Final[int]  # type: ignore[misc]  # pyright: ignore[reportGeneralTypeIssues]
    r"""CONST: The dimensionality of inputs."""

    @abstractmethod
    def forward(self, dt: Tensor, z: Tensor, /) -> Tensor:
        r"""Forward pass of the system.

        Args:
            dt: The time-step to advance the system.
            z: The state estimate at time t.

        Returns:
            z': The updated state of the system at time t + ∆t.
        """
