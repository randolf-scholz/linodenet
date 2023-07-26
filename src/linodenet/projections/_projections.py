r"""Projections."""

__all__ = [
    # Types
    "Projection",
    "ProjectionABC",
]

from abc import abstractmethod
from typing import Protocol, runtime_checkable

from torch import Tensor, nn


@runtime_checkable
class Projection(Protocol):
    """Protocol for Projection Components."""

    def __call__(self, z: Tensor, /) -> Tensor:
        """Forward pass of the projection.

        .. Signature: ``(..., d) -> (..., f)``.
        """
        ...


class ProjectionABC(nn.Module):
    """Abstract Base Class for Projection components."""

    @abstractmethod
    def forward(self, z: Tensor, /) -> Tensor:
        r"""Forward pass of the projection.

        .. Signature: ``(..., d) -> (..., f)``.

        Args:
            z: The input tensor to be projected.

        Returns:
            x: The projected tensor.
        """
