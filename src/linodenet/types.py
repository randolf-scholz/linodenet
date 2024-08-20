r"""Types and Type Aliases."""

__all__ = [
    # Protocols
    "HasHyperparameters",
    "SelfMap",
    "SupportsLenAndGetItem",
    # Aliases
    "Range",
    "DeviceArg",
    "DtypeArg",
    "Nested",
    "Scalar",
    "Shape",
]

from abc import abstractmethod
from collections.abc import Iterable, Mapping, Sequence
from typing import Final, Protocol, SupportsIndex

from torch import device, dtype

# region static type aliases -----------------------------------------------------------
type Scalar = None | bool | int | float | str
r"""Type hint for scalar types allowed by torchscript."""
type DeviceArg = None | str | device  # Literal["cpu", "cuda"]
r"""Type hint for device arguments."""
type DtypeArg = None | dtype  # NOTE: no support for string dtypes!
r"""Type hint for dtype arguments."""
type Shape = int | tuple[int, ...]
r"""Type hint for shape-like inputs."""
# endregion static type aliases --------------------------------------------------------


# region Protocols ---------------------------------------------------------------------
class SelfMap[T](Protocol):
    r"""Protocol for functions that map a type onto itself."""

    @abstractmethod
    def __call__(self, x: T, /) -> T:
        r"""Maps T -> T."""
        ...


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


# endregion Protocol -------------------------------------------------------------------

# region generic type aliases ----------------------------------------------------------
type Range[T] = SupportsLenAndGetItem[T] | Iterable[T]
r"""Type hint for ranges of values."""
type Nested[T] = T | Mapping[str, "Nested[T]"] | Sequence["Nested[T]"]
r"""Type hint for nested types."""
# endregion type aliases ---------------------------------------------------------------
