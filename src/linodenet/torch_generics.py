r"""Generic types for PyTorch modules."""

__all__ = [
    "Config",
    "ModuleMapping",
    "ModuleSequence",
    "SupportsFromConfig",
    # Functions
    "initialize_from_dict",
    "is_config",
]

import logging
from collections.abc import (
    ItemsView,
    Iterable,
    Iterator,
    KeysView,
    Mapping,
    Sequence,
    ValuesView,
)
from importlib import import_module
from typing import (
    TYPE_CHECKING,
    Any,
    Never,
    NewType,
    Protocol,
    Self,
    TypeIs,
    overload,
    runtime_checkable,
)

from torch.nn import Module, ModuleDict, ModuleList

__logger__ = logging.getLogger(__name__)

Config = NewType("Config", Mapping[str, Any])


def is_config(obj: object, /) -> TypeIs[Config]:
    r"""Check if the object is a configuration dictionary.

    A configuration is mapping with the following keys:

    - `__module__` (`str`): The module name.
    - `__name__` (`str`): The class name.
    - `__args__` (`Sequence`, optional): The positional arguments for the class.
    - any extra keys that are valid, non-private identifiers.

    """
    if not isinstance(obj, Mapping):
        return False
    if not isinstance(obj.get("__module__"), str):
        return False
    if not isinstance(obj.get("__name__"), str):
        return False
    if not isinstance(obj.get("__args__"), None | Sequence):
        return False

    # check that extra keys are valid identifiers
    for key in obj:
        if not isinstance(key, str):
            return False
        if key in {"__module__", "__name__", "__args__"}:
            continue
        if not key.isidentifier() or key.startswith("_"):
            return False

    # check JSON serializability
    return True


@runtime_checkable
class SupportsFromConfig(Protocol):
    r"""Protocol for classes that can be initialized from a configuration dictionary."""

    @classmethod
    def from_config(cls, config: Mapping[str, Any], /) -> Self: ...

    # @classmethod
    # def from_config(cls, cfg: Mapping[str, Any] = EMPTY_MAP, /, **kwargs: Any) -> Self:
    #     r"""Initialize from hyperparameters."""
    #     config = cls.HP | dict(cfg, **kwargs)
    #     return cls(**config)  # type: ignore[arg-type]


def initialize_from_dict(cfg: Mapping[str, Any], /) -> Module:
    r"""Initialize a class from a dictionary.

    Args:
        cfg: A dictionary containing the default configuration of the class.

    Note:
        The configuration must provide the keys `__module__` and `__name__`.
        The function will attempt to import the module and class and initialize it.
    """
    config = dict(cfg)
    __logger__.debug("Initializing model from config %s", config)

    if (lib_name := config.pop("__module__", None)) is None:
        raise ValueError(f"Expected {config=} to contain '__module__'")
    if (cls_name := config.pop("__name__", None)) is None:
        raise ValueError(f"Expected {config=} to contain '__name__'")

    try:  # import the module
        library = import_module(lib_name)
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(f"Failed to import {lib_name}") from exc

    try:  # import the class from the module
        cls = getattr(library, cls_name)
    except AttributeError as exc:
        raise AttributeError(f"Failed to import {cls_name} from {lib_name}") from exc
    if not issubclass(cls, Module):
        raise TypeError(f"Expected a subclass of {Module}, but got {cls}")

    # attempt to initialize the class
    #  check if classmethod from_config is available
    if issubclass(cls, SupportsFromConfig):
        try:
            module = cls.from_config(config)
        except Exception as exc:
            raise RuntimeError(f"Failed to initialize {cls} with {config=}") from exc
        return module

    try:
        module = cls(**config)
    except Exception as exc:
        raise RuntimeError(f"Failed to initialize {cls} with {config=}") from exc
    return module


class ModuleSequence[M: Module](ModuleList, Sequence[M]):
    r"""Wrapper for ModuleList to make it a generic Sequence type."""

    @classmethod
    def from_config(cls, config: Mapping[str, Any], /) -> ModuleSequence:
        r"""Initialize from hyperparameters."""
        layers: list[Module] = []
        for layer_cfg in config["layers"]:
            layer = initialize_from_dict(layer_cfg)
            layers.append(layer)

        return ModuleSequence(layers)

    @classmethod
    def from_modules(cls, modules: Iterable[M], /) -> ModuleSequence[M]:
        r"""Initialize from an iterable of modules."""
        return ModuleSequence(modules)

    if TYPE_CHECKING:

        @overload
        def __init__(self: ModuleSequence[Never], /) -> None: ...
        @overload
        def __init__(self, modules: Iterable[M], /) -> None: ...

        def __iter__(self) -> Iterator[M]: ...
        @overload  # type: ignore[override]
        def __getitem__(self, index: int, /) -> M: ...
        @overload
        def __getitem__(self, index: slice, /) -> Self: ...  # pyright: ignore[reportIncompatibleMethodOverride]


class ModuleMapping[M: Module](ModuleDict, Mapping[str, M]):
    r"""Wrapper for ModuleDict to make it a generic Mapping type."""

    def __hash__(self) -> int:
        # NOTE: fixes https://github.com/pytorch/pytorch/issues/110959
        return hash(tuple(self.items()))

    if TYPE_CHECKING:

        @overload
        def __init__(self: ModuleMapping[Never], /) -> None: ...
        @overload
        def __init__(self, modules: Mapping[str, M], /) -> None: ...

        def __iter__(self) -> Iterator[str]: ...
        def __getitem__(self, key: str, /) -> M: ...  # pyright: ignore[reportIncompatibleMethodOverride]
        def keys(self) -> KeysView[str]: ...
        def values(self) -> ValuesView[M]: ...
        def items(self) -> ItemsView[str, M]: ...
