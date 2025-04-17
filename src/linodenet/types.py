r"""Types and Type Aliases."""

__all__ = [
    # Protocols
    "HasHyperparameters",
    "Identity",
    "SelfMap",
    "SupportsLenAndGetItem",
    "SupportsSelfAdd",
    # Aliases
    "DeviceArg",
    "DtypeArg",
    "Nested",
    "NestedDict",
    "NestedMapping",
    "PathLike",
    "Range",
    "Scalar",
    "Shape",
    "Size",
]

from abc import abstractmethod
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Final, Protocol, Self, SupportsIndex, runtime_checkable

from torch import device, dtype

# region type aliases ------------------------------------------------------------------
type PathLike = str | Path
r"""Type hint for path-like objects."""
type Scalar = None | bool | int | float
r"""Type hint for scalar types allowed by torchscript."""
type DeviceArg = None | str | device  # Literal["cpu", "cuda"]
r"""Type hint for device arguments."""
type DtypeArg = None | dtype  # NOTE: no support for string dtypes!
r"""Type hint for dtype arguments."""
type Shape = tuple[int, ...]
r"""Type hint for shape-like inputs."""
type Size = tuple[int, ...] | list[int]
r"""Type hint for size-like inputs."""
type SizeArg = None | int | list[int] | tuple[int, ...]
r"""Type hint for size arguments."""
type NestedMapping[K, V] = Mapping[K, V | "NestedMapping[K, V]"]
r"""Generic Type Alias for nested `Mapping`."""
type NestedDict[K, V] = dict[K, V | "NestedDict[K, V]"]
r"""Generic Type Alias for nested `dict`."""
# endregion type aliases ---------------------------------------------------------------


# region Protocols ---------------------------------------------------------------------
class Identity(Protocol):
    r"""Protocol for the identity function."""

    def __call__[T](self, x: T, /) -> T: ...


class SelfMap[T](Protocol):
    r"""Protocol for generic function that map a type onto itself."""

    @abstractmethod
    def __call__(self, x: T, /) -> T: ...


class SupportsLenAndGetItem[T](Protocol):
    r"""Protocol for types that support `__len__` and `__getitem__`."""

    def __len__(self) -> int: ...
    def __getitem__(self, index: SupportsIndex, /) -> T: ...


class HasHyperparameters(Protocol):
    r"""Protocol for types that have hyperparameters."""

    HP: Final[str]  # type: ignore[misc]
    r"""Default hyperparameters of the type."""

    config: Final[dict]  # type: ignore[misc]
    r"""Concrete hyperparameters of an instance."""


@runtime_checkable
class SupportsSelfAdd(Protocol):
    r"""Protocol for types that support `__add__` with itself."""

    def __add__(self, other: Self, /) -> Self: ...


# endregion Protocol -------------------------------------------------------------------


# region generic type aliases ----------------------------------------------------------
type Range[T] = SupportsLenAndGetItem[T] | Iterable[T]
r"""Type hint for ranges of values."""
type Nested[T] = T | Mapping[str, Nested[T]] | Sequence[Nested[T]]
r"""Type hint for nested types."""
# endregion type aliases ---------------------------------------------------------------
