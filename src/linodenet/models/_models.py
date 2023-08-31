r"""Models of the LinODE-Net package."""

__all__ = [
    "Model",
    "ModelABC",
]

from abc import abstractmethod
from typing import Protocol

from torch import nn

from linodenet.utils.types import TensorLike


class Model(Protocol):
    """Protocol for all models."""

    def __call__(self, *args: TensorLike, **kwargs: TensorLike) -> TensorLike:
        """Forward pass of the model."""
        ...


class ModelABC(nn.Module):
    """Abstract Base Class for all models."""

    @abstractmethod
    def forward(self, *args: TensorLike, **kwargs: TensorLike) -> TensorLike:
        """Forward pass of the model."""
