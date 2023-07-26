r"""Models of the LinODE-Net package."""

__all__ = [
    "Model",
    "ModelABC",
]

from abc import abstractmethod
from typing import Protocol, TypeAlias, TypeVar

from torch import Tensor, nn

T = TypeVar("T")
Recursive: TypeAlias = T | list[T] | tuple[T, ...] | dict[str, T]
Scalars: TypeAlias = int | float | bool | str
TensorLike: TypeAlias = Recursive[Tensor | Scalars]


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
