r"""Types and Type Aliases."""

__all__ = [
    # Type Variables
    "CLS",
    "F",
    "M",
    "R",
    "T",
    "S",
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
from collections.abc import Callable, Iterable, Mapping, Sequence

from torch import device, dtype, nn
from typing_extensions import Final, Protocol, SupportsInt, TypeAlias, TypeVar

# region static type aliases -----------------------------------------------------------
Scalar: TypeAlias = None | bool | int | float | str
r"""Type hint for scalar types allowed by torchscript."""

DeviceArg: TypeAlias = None | str | device  # Literal["cpu", "cuda"]
r"""Type hint for device arguments."""

DtypeArg: TypeAlias = None | dtype  # NOTE: no support for string dtypes!
r"""Type hint for dtype arguments."""

Shape: TypeAlias = int | tuple[int, ...]
r"""Type hint for shape-like inputs."""
# endregion static type aliases --------------------------------------------------------


# region type variables ----------------------------------------------------------------
S = TypeVar("S")
r"""Type Variable for generic types."""
T = TypeVar("T")
r"""Type Variable for generic types."""
T_co = TypeVar("T_co", covariant=True)
r"""Type Variable for generic types (always covariant)."""

CLS = TypeVar("CLS", bound=type)
r"""Type hint for classes."""

R = TypeVar("R", covariant=True)  # noqa: PLC0105
r"""Type hint return value (always covariant)."""

M = TypeVar("M", bound=nn.Module)
r"""Type Variable for nn.Modules."""

F = TypeVar("F", bound=Callable)
r"""Type Variable for callables."""


# endregion type variables -------------------------------------------------------------
# region Protocols ---------------------------------------------------------------------
class SelfMap(Protocol[T]):
    r"""Protocol for functions that map a type onto itself."""

    @abstractmethod
    def __call__(self, x: T, /) -> T:
        r"""Maps T -> T."""
        ...


class SupportsLenAndGetItem(Protocol[T_co]):
    r"""Protocol for types that support `__len__` and `__getitem__`."""

    def __len__(self) -> int: ...
    def __getitem__(self, index: SupportsInt, /) -> T_co: ...


class HasHyperparameters(Protocol):
    r"""Protocol for types that have hyperparameters."""

    HP: Final[str]  # type: ignore[misc]
    r"""Default hyperparameters of the type."""

    config: Final[dict]  # type: ignore[misc]
    r"""Concrete hyperparameters of an instance."""


# endregion Protocol -------------------------------------------------------------------

# region generic type aliases ----------------------------------------------------------
Range: TypeAlias = SupportsLenAndGetItem[T] | Iterable[T]
r"""Type hint for ranges of values."""
Nested: TypeAlias = T | Mapping[str, "Nested[T]"] | Sequence["Nested[T]"]
r"""Type hint for nested types."""
# endregion type aliases ---------------------------------------------------------------
