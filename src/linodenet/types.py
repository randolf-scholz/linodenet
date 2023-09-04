"""Types and Type Aliases."""

__all__ = [
    # Aliases
    "Nested",
    "Recursive",
    "Scalar",
    "SizeLike",
    "TensorLike",
    # Type Variables
    "NestedTensorVar",
    "T",
    "module_var",
    "return_co",
    "type_var",
]

from collections.abc import Iterable, Mapping
from typing import TypeAlias, TypeVar

from torch import Tensor, nn

# region static type aliases -----------------------------------------------------------
Scalar: TypeAlias = None | bool | int | float | str
r"""Type hint for scalar types allowed by torchscript."""

SizeLike: TypeAlias = int | tuple[int, ...]
r"""Type hint for shape-like inputs."""
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
# endregion type variables -------------------------------------------------------------


# region generic type aliases ----------------------------------------------------------
Nested: TypeAlias = T | Mapping[str, "Nested"] | Iterable["Nested"]
"""Type hint for nested types."""

Recursive: TypeAlias = T | list[T] | tuple[T, ...] | dict[str, T]
r"""Type hint for recursive types."""

TensorLike: TypeAlias = Recursive[Tensor | Scalar]
"""Type hint for tensor-like inputs."""
# endregion type aliases ---------------------------------------------------------------


# region type variables ----------------------------------------------------------------
NestedTensorVar = TypeVar("NestedTensorVar", bound=Nested[Tensor])
r"""Type Variable for NestedTensors."""
# endregion type variables -------------------------------------------------------------
