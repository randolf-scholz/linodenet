"""Types and Type Aliases."""

__all__ = [
    # Aliases
    "Nested",
    "Scalar",
    "Device",
    "Dtype",
    "Shape",
    # Type Variables
    "T",
    "callable_var",
    "module_var",
    "return_co",
    "type_var",
    # Protocols
    "SelfMap",
    "HasHyperparameters",
]

from abc import abstractmethod
from collections.abc import Callable, Mapping, Sequence
from typing import Final, Protocol, TypeAlias, TypeVar

from torch import device, dtype, nn

# region static type aliases -----------------------------------------------------------
Scalar: TypeAlias = None | bool | int | float | str
r"""Type hint for scalar types allowed by torchscript."""

Device: TypeAlias = None | device
r"""Type hint for device types allowed by torchscript."""

Dtype: TypeAlias = None | dtype
r"""Type hint for dtype types allowed by torchscript."""

Shape: TypeAlias = int | tuple[int, ...]
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

callable_var = TypeVar("callable_var", bound=Callable)
r"""Type Variable for callables."""
# endregion type variables -------------------------------------------------------------


# region generic type aliases ----------------------------------------------------------
Nested: TypeAlias = T | Mapping[str, "Nested[T]"] | Sequence["Nested[T]"]
"""Type hint for nested types."""
# endregion type aliases ---------------------------------------------------------------


# region Protocols ---------------------------------------------------------------------
class SelfMap(Protocol[T]):
    """Protocol for functions that map a type onto itself."""

    @abstractmethod
    def __call__(self, x: T, /) -> T:
        """Maps T -> T."""
        ...


class HasHyperparameters(Protocol):
    """Protocol for types that have hyperparameters."""

    HP: Final[str]  # type: ignore[misc]
    """Default hyperparameters of the type."""

    config: Final[dict]  # type: ignore[misc]
    """Concrete hyperparameters of an instance."""


# endregion Protocol -------------------------------------------------------------------
