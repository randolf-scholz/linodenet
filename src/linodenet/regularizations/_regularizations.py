r"""Regularizations."""

__all__ = [
    "Regularization",
    "RegularizationABC",
]

from abc import abstractmethod
from typing import Protocol, runtime_checkable

from torch import Tensor, nn


@runtime_checkable
class Regularization(Protocol):
    """Protocol for Regularization Components."""

    def __call__(self, x: Tensor, /) -> Tensor:
        """Forward pass of the regularization."""


class RegularizationABC(nn.Module):
    """Abstract Base Class for Regularization components."""

    @abstractmethod
    def forward(self, x: Tensor, /) -> Tensor:
        r"""Forward pass of the regularization.

        .. Signature: ``... -> 1``.

        Args:
            x: The input tensor to be regularized.

        Returns:
            r: The (scalar) regularization value .
        """
