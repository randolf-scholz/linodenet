r"""Types and Type Aliases."""

__all__ = [
    # Protocols
    "SupportsLenAndGetItem",
    "SupportsSelfAdd",
    # Callbacks
    "Identity",
    "SelfMap",
    # Aliases
    "DeviceArg",
    "DtypeArg",
    "DimArg",
    "Makes",
    "Shape",
    "Nested",
    "NestedDict",
    "NestedMapping",
    "PathLike",
    "Range",
    "Scalar",
    # Tensor Aliases
    "BoolTensor",
    "IntTensor",
    "FloatTensor",
    "ComplexTensor",
]

import os
from abc import abstractmethod
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Protocol, Self, SupportsIndex, runtime_checkable

from torch import Tensor, device, dtype

# region type aliases ------------------------------------------------------------------
# NOTE: Tensor aliases intentionally do not use the type keyword.
BoolTensor = Tensor
IntTensor = Tensor
FloatTensor = Tensor
ComplexTensor = Tensor

type Makes[T] = dict
r"""Type hint for dictionaries that make objects of type T."""
type PathLike = str | Path | os.PathLike[str]
r"""Type hint for path-like objects."""
type Scalar = None | bool | int | float
r"""Type hint for scalar types allowed by torchscript."""
type DeviceArg = None | str | device  # Literal["cpu", "cuda"]
r"""Type hint for device arguments."""
type DtypeArg = None | dtype  # NOTE: no support for string dtypes!
r"""Type hint for dtype arguments."""
type Shape = tuple[int, ...]
r"""Type hint for shapes."""
type DimArg = None | int | Sequence[SupportsIndex]
r"""Type hint for dimension arguments."""
type NestedMapping[K, V] = Mapping[K, V | "NestedMapping[K, V]"]
r"""Generic Type Alias for nested `Mapping`."""
type NestedDict[K, V] = dict[K, V | "NestedDict[K, V]"]
r"""Generic Type Alias for nested `dict`."""
type Range[T] = SupportsLenAndGetItem[T] | Iterable[T]
r"""Type hint for ranges of values."""
type Nested[T] = T | Mapping[str, Nested[T]] | Sequence[Nested[T]]
r"""Type hint for nested types."""
# endregion type aliases ---------------------------------------------------------------


# region protocols ---------------------------------------------------------------------
class SupportsLenAndGetItem[T](Protocol):
    r"""Protocol for types that support `__len__` and `__getitem__`."""

    def __len__(self) -> int: ...
    def __getitem__(self, index: SupportsIndex, /) -> T: ...


@runtime_checkable
class SupportsSelfAdd(Protocol):
    r"""Protocol for types that support `__add__` with itself."""

    def __add__(self, other: Self, /) -> Self: ...


# endregion protocols ------------------------------------------------------------------


# region callback protocols ------------------------------------------------------------
class Identity(Protocol):
    r"""Protocol for the identity function."""

    def __call__[T](self, x: T, /) -> T: ...


class SelfMap[T](Protocol):
    r"""Protocol for generic functions that map a type onto itself."""

    @abstractmethod
    def __call__(self, x: T, /) -> T: ...


# endregion callback protocols ---------------------------------------------------------
