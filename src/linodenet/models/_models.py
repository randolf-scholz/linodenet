r"""Models of the LinODE-Net package."""

__all__ = [
    "Model",
    "ModelABC",
]

from abc import abstractmethod
from typing import Protocol

from torch import Tensor, nn

from linodenet.types import Nested, Scalar


class Model(Protocol):
    """Protocol for all models."""

    def __call__(
        self, *args: Nested[Tensor | Scalar], **kwargs: Nested[Tensor, Scalar]
    ) -> Nested[Tensor, Scalar]:
        """Forward pass of the model."""
        ...


class ModelABC(nn.Module):
    """Abstract Base Class for all models."""

    @abstractmethod
    def forward(
        self, *args: Nested[Tensor | Scalar], **kwargs: Nested[Tensor | Scalar]
    ) -> Nested[Tensor | Scalar]:
        """Forward pass of the model."""
