"""Types and Type Aliases."""

__all__ = [
    # Aliases
    "Nested",
    "Scalar",
    "SizeLike",
    # Type Variables
    "T",
    "module_var",
    "return_co",
    "type_var",
]

from collections.abc import Mapping, Sequence
from typing import TypeAlias, TypeVar

from torch import nn

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

module_var = TypeVar("module_var", bound=nn.Module)
r"""Type Variable for nn.Modules."""
# endregion type variables -------------------------------------------------------------


# region generic type aliases ----------------------------------------------------------
Nested: TypeAlias = T | Mapping[str, "Nested[T]"] | Sequence["Nested[T]"]
"""Type hint for nested types."""
# endregion type aliases ---------------------------------------------------------------
