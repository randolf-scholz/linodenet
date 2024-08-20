r"""Utility functions."""

__all__ = [
    # Functions
    "autojit",
    "deep_dict_update",
    "deep_keyval_update",
    "initialize_from_dict",
    "initialize_from_type",
    "is_dunder",
    "is_private",
    "pad",
    "try_initialize_from_config",
]

import logging
from collections.abc import Mapping
from copy import deepcopy
from functools import wraps
from importlib import import_module
from typing import Any, Self

import torch
from torch import Tensor, jit, nn

from linodenet.config import CONFIG

__logger__ = logging.getLogger(__name__)


def autojit[M: nn.Module](base_class: type[M], /) -> type[M]:
    r"""Class decorator that enables automatic jitting of nn.Modules upon instantiation.

    Makes it so that

    >>> class MyModule: ...
    >>> model = jit.script(MyModule())

    and

    >>> class MyModule: ...
    >>> model = MyModule()

    are (roughly?) equivalent.
    """
    if not isinstance(base_class, type):
        raise TypeError("Expected a class.")
    if not issubclass(base_class, nn.Module):
        raise TypeError("Expected a subclass of nn.Module.")

    @wraps(base_class, updated=())
    class WrappedClass(base_class):  # type: ignore[valid-type,misc]
        r"""A simple Wrapper."""

        def __new__(cls, *args: Any, **kwargs: Any) -> Self:
            # Note: If __new__() does not return an instance of cls,
            #   then the new instance's __init__() method will not be invoked.
            instance = base_class(*args, **kwargs)

            if CONFIG.autojit:
                scripted = jit.script(instance)
                return scripted  # type: ignore[return-value]
            return instance  # type: ignore[return-value]

    if not isinstance(WrappedClass, type):
        raise TypeError(f"Expected a class, got {WrappedClass}.")
    if not issubclass(WrappedClass, base_class):
        raise TypeError(f"Expected {WrappedClass} to be a subclass of {base_class}.")

    return WrappedClass


def deep_dict_update(d: dict, new: Mapping, /, *, inplace: bool = False) -> dict:
    r"""Update nested dictionary recursively in-place with new dictionary.

    References:
        https://stackoverflow.com/a/30655448/9318372
    """
    if not inplace:
        d = deepcopy(d)

    for key, value in new.items():
        d[key] = (
            deep_dict_update(d.get(key, {}), value)
            if isinstance(value, Mapping)
            else value
        )
    return d


def deep_keyval_update(d: dict, /, **new_kv: Any) -> dict:
    r"""Update nested dictionary recursively in-place with key-value pairs.

    References:
        https://stackoverflow.com/a/30655448/9318372
    """
    for key, value in d.items():
        if isinstance(value, Mapping) and value:
            d[key] = deep_keyval_update(d.get(key, {}), **new_kv)
        elif key in new_kv:
            d[key] = new_kv[key]
    return d


def is_dunder(s: str, /) -> bool:
    r"""Check if name is a dunder method."""
    return s.isidentifier() and s.startswith("__") and s.endswith("__")


def is_private(s: str, /) -> bool:
    r"""Check if name is a private method."""
    return s.isidentifier() and s.startswith("_") and not s.endswith("__")


def get_module(obj_ref: object, /) -> str:
    return obj_ref.__module__.rsplit(".", maxsplit=1)[-1]


def initialize_from_dict(defaults: Mapping[str, Any], /, **kwargs: Any) -> nn.Module:
    r"""Initialize a class from a dictionary.

    Parameters:
        defaults: A dictionary containing the default configuration of the class.
        kwargs: Additional keyword arguments to override the configuration.

    Note:
        The configuration must provide the keys `__module__` and `__name__`.
        The function will attempt to import the module and class and initialize it.
    """
    config = dict(defaults, **kwargs)
    assert "__name__" in config, f"__name__ not found in {config=}"
    assert "__module__" in config, f"__module__ not found in {config=}"
    __logger__.debug("Initializing model from config %s", config)
    library_name: str = config.pop("__module__")
    class_name: str = config.pop("__name__")

    try:  # import the module
        library = import_module(library_name)
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(f"Failed to import {library_name}") from exc

    try:  # import the class from the module
        module_type = getattr(library, class_name)
    except AttributeError as exc:
        raise AttributeError(
            f"Failed to import {class_name} from {library_name}"
        ) from exc

    try:  # attempt to initialize the class
        module = module_type(**config)
    except TypeError as exc:
        raise TypeError(f"Failed to initialize {module_type} with {config=}") from exc
    else:
        assert isinstance(module, nn.Module)

    return module


def initialize_from_type(module_type: type[nn.Module], /, **kwargs: Any) -> nn.Module:
    r"""Initialize a class from a dictionary."""
    default_config = getattr(module_type, "HP", {})
    config = dict(default_config, **kwargs)
    __logger__.debug("Initializing model_type %s from config %s", module_type, config)

    try:  # attempt to initialize the class
        return module_type(**config)
    except TypeError as exc:
        raise TypeError(f"Failed to initialize {module_type} with {config=}") from exc


def try_initialize_from_config(
    obj: nn.Module | type[nn.Module] | Mapping[str, Any], /, **kwargs: Any
) -> nn.Module:
    r"""Try to initialize a module from a config."""
    match obj:
        case nn.Module() as module:
            assert not kwargs, f"Unexpected {kwargs=}"
            return module
        case type() as module_type if issubclass(module_type, nn.Module):
            return initialize_from_type(module_type, **kwargs)
        case Mapping() as config:
            return initialize_from_dict(config, **kwargs)
        case _:
            raise TypeError(f"Unsupported type {type(obj)}")


@jit.script
def pad(
    x: Tensor,
    value: float,
    pad_width: int,
    dim: int = -1,
    prepend: bool = False,
) -> Tensor:
    r"""Pad a tensor with a constant value along a given dimension."""
    shape = list(x.shape)
    shape[dim] = pad_width
    z = torch.full(shape, value, dtype=x.dtype, device=x.device)

    if prepend:
        return torch.cat((z, x), dim=dim)
    return torch.cat((x, z), dim=dim)
