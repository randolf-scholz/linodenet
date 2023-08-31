"""Types and Type Aliases."""

__all__ = [
    # Aliases
    "NestedTensor",
    "Recursive",
    "Scalars",
    "SizeLike",
    "TensorLike",
    "TensorLike",
    # Type Variables
    "NestedTensorVar",
    "T",
    "module_var",
    "return_co",
    "type_var",
]

from collections.abc import Iterable, Mapping
from typing import Any, TypeAlias, TypeVar

from torch import Tensor, nn

# region static type aliases -----------------------------------------------------------
Scalars: TypeAlias = int | float | bool | str
r"""Type hint for scalar types."""

SizeLike: TypeAlias = int | tuple[int, ...]
r"""Type hint for shape-like inputs."""

NestedTensor: TypeAlias = (
    Tensor | Mapping[Any, "NestedTensor"] | Iterable["NestedTensor"]
)
r"""Type hint for nested tensors."""
# endregion static type aliases --------------------------------------------------------


# region type variables ----------------------------------------------------------------
T = TypeVar("T")
"""Type Variable for generic types."""

type_var = TypeVar("type_var", bound=type)
r"""Type hint for classes."""

return_co = TypeVar("return_co", covariant=True)
r"""Type hint return value."""

module_var = TypeVar("module_var", bound=type[nn.Module])
r"""Type Variable for nn.Modules."""

NestedTensorVar = TypeVar("NestedTensorVar", bound=NestedTensor)
r"""Type Variable for NestedTensors."""
# endregion type variables -------------------------------------------------------------


# region generic type aliases ----------------------------------------------------------
Recursive: TypeAlias = T | list[T] | tuple[T, ...] | dict[str, T]
r"""Type hint for recursive types."""

TensorLike: TypeAlias = Recursive[Tensor | Scalars]
"""Type hint for tensor-like inputs."""
# endregion type aliases ---------------------------------------------------------------
