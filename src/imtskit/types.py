r"""Type aliases and protocols used throughout the library."""

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
r"""Configuration dictionary intended to construct an object of type `T`."""
type PathLike = str | Path | os.PathLike[str]
r"""Type hint for path-like objects."""
type Scalar = None | bool | int | float
r"""Scalar values represented by `None`, `bool`, `int`, or `float`."""
type DeviceArg = None | str | device  # Literal["cpu", "cuda"]
r"""Device argument given as `None`, a string, or a `torch.device`."""
type DtypeArg = None | dtype  # NOTE: no support for string dtypes!
r"""Dtype argument given as `None` or a `torch.dtype`; strings are not supported."""
type Shape = tuple[int, ...]
r"""Tensor shape represented by a tuple of integer dimensions."""
type DimArg = None | int | Sequence[SupportsIndex]
r"""Dimension argument given as `None`, an integer, or a sequence of indices."""
type Range[T] = SupportsLenAndGetItem[T] | Iterable[T]
r"""Iterable or indexable collection of values of type `T`."""
type Nested[T] = T | Mapping[str, Nested[T]] | Sequence[Nested[T]]
r"""Value of type `T` or recursively nested string-keyed mappings and sequences."""
# endregion type aliases ---------------------------------------------------------------


# region protocols ---------------------------------------------------------------------
class SupportsLenAndGetItem[T](Protocol):
    r"""Protocol for types that support `__len__` and `__getitem__`."""

    def __len__(self) -> int: ...
    def __getitem__(self, index: SupportsIndex, /) -> T: ...


@runtime_checkable
class SupportsSelfAdd(Protocol):
    r"""Protocol for addition between values of the same type, returning that type."""

    def __add__(self, other: Self, /) -> Self: ...


# endregion protocols ------------------------------------------------------------------


# region callback protocols ------------------------------------------------------------
class Identity(Protocol):
    r"""Protocol for callables that return their input unchanged."""

    def __call__[T](self, x: T, /) -> T: ...


class SelfMap[T](Protocol):
    r"""Protocol for callables whose input and output have type `T`."""

    @abstractmethod
    def __call__(self, x: T, /) -> T: ...


# endregion callback protocols ---------------------------------------------------------
