r"""Models for the latent dynamical system."""

__all__ = [
    # ABCs & Protocols
    "System",
    "SystemABC",
]

from abc import abstractmethod

from torch import Tensor, nn
from typing_extensions import Protocol, runtime_checkable


@runtime_checkable
class System(Protocol):
    """Protocol for System Components."""

    def __call__(self, dt: Tensor, z: Tensor, /) -> Tensor:
        """Forward pass of the system.

        .. Signature: ``[∆t=(...,), x=(..., d)] -> (..., d)]``.
        """
        ...


class SystemABC(nn.Module):
    """Abstract Base Class for System components."""

    @abstractmethod
    def forward(self, dt: Tensor, z: Tensor, /) -> Tensor:
        r"""Forward pass of the system.

        Args:
            dt: The time-step to advance the system.
            z: The state estimate at time t.

        Returns:
            z': The updated state of the system at time t + ∆t.
        """
