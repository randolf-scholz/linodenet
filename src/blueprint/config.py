r"""Configuration protocols and utilities."""
# ruff: noqa: SIM103

__all__ = [
    "SupportsConfig",
    "SupportsDefaultConfig",
    "SupportsFromConfig",
    # functions
    "has_config",
    "has_default_config",
    "is_config",
    "is_key",
]

from typing import Any, ClassVar, Protocol, Self, TypeGuard, runtime_checkable

type Key = str


def is_key(arg: object, /) -> TypeGuard[Key]:
    return isinstance(arg, str) and arg.isidentifier()


class SupportsConfig[K: Key, V](Protocol):
    r"""Models that support a hyperparameter dictionary.

    A hyperparameter dictionary should be such that

    type(model)(**model.config)

    recovers the model.
    """

    @property
    def config(self) -> dict[K, V]: ...


def is_config(arg: object, /) -> TypeGuard[dict[Key, Any]]:
    if not isinstance(arg, dict):
        return False
    if not all(is_key(key) for key in arg):
        return False
    return True


def has_config(arg: object, /) -> TypeGuard[SupportsConfig]:
    if (config := getattr(arg, "config", None)) is None:
        return False
    return is_config(config)


class SupportsDefaultConfig(Protocol):
    r"""Models that have a default configuration dataclass."""

    DEFAULT_CONFIG: ClassVar[type]


def has_default_config(arg: object, /) -> TypeGuard[SupportsDefaultConfig]:
    if not isinstance(arg, type):
        arg = type(arg)
    default_config = getattr(arg, "DEFAULT_CONFIG", None)
    return default_config is not None


@runtime_checkable
class SupportsFromConfig(Protocol):
    r"""Models that can be explicitly initialized from a configuration dictionary."""

    @classmethod
    def from_config(cls, config: dict[str, Any], /) -> Self: ...
