r"""Models for the latent dynamical system."""

__all__ = [
    # ABCs & Protocols
    "ContinuousSystem",
    "SystemABC",
]

from abc import abstractmethod
from typing import Final, Protocol, runtime_checkable

from torch import Tensor, nn


@runtime_checkable
class ContinuousSystem(Protocol):
    r"""Protocol for System Components."""

    input_size: Final[int]  # type: ignore[misc]
    r"""CONST: The dimensionality of inputs."""

    def __call__(self, dt: Tensor, z: Tensor, /) -> Tensor:
        r"""Forward pass of the system.

        .. Signature: ``[∆t=(...,), x=(..., d)] -> (..., d)]``.
        """
        ...


class DiscreteSystem(Protocol):
    r"""Protocol for Discrete System Components."""

    input_size: Final[int]  # type: ignore[misc]
    r"""CONST: The dimensionality of inputs."""

    def __call__(self, n_steps: int, z: Tensor, /) -> Tensor:
        r"""Forward pass of the system.

        .. Signature: ``(…, d) -> (…, d)``.
        """
        ...


class SystemABC(nn.Module):
    r"""Abstract Base Class for System components."""

    @abstractmethod
    def forward(self, dt: Tensor, z: Tensor, /) -> Tensor:
        r"""Forward pass of the system.

        Args:
            dt: The time-step to advance the system.
            z: The state estimate at time t.

        Returns:
            z': The updated state of the system at time t + ∆t.
        """
