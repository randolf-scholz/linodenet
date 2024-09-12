r"""Utility functions."""

__all__ = [
    # Types
    "NestedMapping",
    "NestedDict",
    # Functions
    "autojit",
    "deep_dict_update",
    "deep_keyval_update",
    "initialize_from_dict",
    "initialize_from_type",
    "is_allcaps",
    "is_dunder",
    "is_private",
    "flatten_dict",
    "unflatten_dict",
    "pad",
    "try_initialize_from_config",
]

import logging
from collections.abc import Callable, Iterable, Mapping
from copy import deepcopy
from functools import wraps
from importlib import import_module
from typing import Any, Self, cast, overload

import torch
from torch import Tensor, jit, nn

from linodenet.config import CONFIG

__logger__ = logging.getLogger(__name__)
type NestedMapping[K, V] = Mapping[K, V | "NestedMapping[K, V]"]
r"""Generic Type Alias for nested `Mapping`."""
type NestedDict[K, V] = dict[K, V | "NestedDict[K, V]"]
r"""Generic Type Alias for nested `dict`."""


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


def is_allcaps(s: str, /) -> bool:
    r"""Check if a string is all caps."""
    return s.isidentifier() and s.isupper() and s.isalpha()


def is_dunder(s: str, /) -> bool:
    r"""Check if name is a dunder method."""
    return s.isidentifier() and s.startswith("__") and s.endswith("__")


def is_private(s: str, /) -> bool:
    r"""Check if name is a private method."""
    return s.isidentifier() and s.startswith("_") and not s.endswith("__")


def get_module(obj_ref: object, /) -> str:
    return obj_ref.__module__.rsplit(".", maxsplit=1)[-1]


def flatten_dict[K, K2](
    d: NestedMapping[K, Any],
    /,
    *,
    recursive: bool | int = True,
    join_fn: Callable[[Iterable[K]], K2] = ".".join,  # type: ignore[assignment]
    split_fn: Callable[[K2], Iterable[K]] = lambda s: s.split("."),  # type: ignore[attr-defined]
) -> dict[K2, Any]:
    r"""Flatten dictionaries recursively.

    Args:
        d: dictionary to flatten
        recursive: whether to flatten recursively. If `recursive` is an integer,
            then it flattens recursively up to `recursive` levels.
        join_fn: function to join keys
        split_fn: function to split keys

    Examples:
        Using ``join_fn = ".".join`` and ``split_fn = lambda s: s.split(".")``
        will combine string keys like ``"a"`` and ``"b"`` into ``"a.b"``.

        >>> flatten_dict({"a": {"b": 1, "c": 2}})
        {'a.b': 1, 'a.c': 2}

        >>> flatten_dict({"a": {"b": {"x": 2}, "c": 2}})
        {'a.b.x': 2, 'a.c': 2}

        >>> flatten_dict({"a": {"b": 1, "c": 2}}, recursive=False)
        {'a': {'b': 1, 'c': 2}}

        Using ``join_fn = tuple`` and ``split_fn = lambda s: s`` will combine
        keys like ``("a", "b")`` and ``("a", "c")`` into ``("a", "b", "c")``.

        >>> flatten_dict({"a": {"b": 1, "c": 2}}, join_fn=tuple, split_fn=lambda x: x)
        {('a', 'b'): 1, ('a', 'c'): 2}

        >>> flatten_dict(
        ...     {"a": {"b": {"x": 2}, "c": 2}}, join_fn=tuple, split_fn=lambda x: x
        ... )
        {('a', 'b', 'x'): 2, ('a', 'c'): 2}

        >>> flatten_dict({"a": {17: "foo", 18: "bar"}}, join_fn=tuple, split_fn=lambda x: x)
        {('a', 17): 'foo', ('a', 18): 'bar'}

        When trying to flatten a partially flattened dictionary, setting recursive=<int>.

        >>> flatten_dict(
        ...     {"a": {(1, True): "foo", (2, False): "bar"}},
        ...     join_fn=tuple,
        ...     split_fn=lambda x: x,
        ... )
        {('a', (1, True)): 'foo', ('a', (2, False)): 'bar'}

        >>> flatten_dict(
        ...     {"a": {(1, True): "foo", (2, False): "bar"}},
        ...     join_fn=tuple,
        ...     split_fn=lambda x: x,
        ...     recursive=1,
        ... )
        {('a', 1, True): 'foo', ('a', 2, False): 'bar'}
    """
    if not recursive:
        return cast(dict[K2, Any], dict(d))

    recursive = recursive if isinstance(recursive, bool) else recursive - 1
    result: dict[K2, Any] = {}
    for key, item in d.items():
        if isinstance(item, Mapping):
            for subkey, subitem in flatten_dict(
                item,
                recursive=recursive,
                join_fn=join_fn,
                split_fn=split_fn,
            ).items():
                new_key = join_fn((key, *split_fn(subkey)))
                result[new_key] = subitem
        else:
            new_key = join_fn((key,))
            result[new_key] = item
    return result


def unflatten_dict[K, K2](
    d: Mapping[K2, Any],
    /,
    *,
    recursive: bool | int = True,
    join_fn: Callable[[Iterable[K]], K2] = ".".join,  # type: ignore[assignment]
    split_fn: Callable[[K2], Iterable[K]] = lambda s: s.split("."),  # type: ignore[attr-defined]
) -> NestedDict[K, Any]:
    r"""Unflatten dictionaries recursively.

    Examples:
        Using ``join_fn = ".".join`` and ``split_fn = lambda s: s.split(".")``
        will split up string keys like ``"a.b.c"`` into ``{"a": {"b": {"c": ...}}}``.

        >>> unflatten_dict({"a.b": 1, "a.c": 2})
        {'a': {'b': 1, 'c': 2}}

        >>> unflatten_dict({"a.b": 1, "a.c": 2}, recursive=False)
        {'a.b': 1, 'a.c': 2}

        Using ``join_fn = tuple`` and ``split_fn = lambda s: s`` will split up
        keys like ``("a", "b", "c")`` into ``{"a": {"b": {"c": ...}}}``.

        >>> unflatten_dict(
        ...     {("a", 17): "foo", ("a", 18): "bar"}, join_fn=tuple, split_fn=lambda x: x
        ... )
        {'a': {17: 'foo', 18: 'bar'}}
    """
    if not recursive:
        return cast(dict[K, Any], dict(d))

    recursive = recursive if isinstance(recursive, bool) else recursive - 1
    result: dict[K, Any] = {}
    for key, item in d.items():
        outer_key, *inner_keys = split_fn(key)
        if inner_keys:
            result.setdefault(outer_key, {})
            result[outer_key] |= unflatten_dict(
                {join_fn(inner_keys): item},
                recursive=recursive,
                split_fn=split_fn,
                join_fn=join_fn,
            )
        else:
            result[outer_key] = item
    return result


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

    if "__name__" not in config:
        raise ValueError(f"Expected {config=} to contain '__name__'")
    if "__module__" not in config:
        raise ValueError(f"Expected {config=} to contain '__module__'")

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
        if not isinstance(module, nn.Module):
            raise TypeError(f"Expected {module_type} to be a subclass of nn.Module")

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


@overload
def try_initialize_from_config[M: nn.Module](module: M, /) -> M: ...
@overload
def try_initialize_from_config[M: nn.Module](cls: type[M], /, **kwargs: Any) -> M: ...
@overload
def try_initialize_from_config(config: Mapping[str, Any], /) -> nn.Module: ...
def try_initialize_from_config[M: nn.Module](
    obj: M | type[M] | Mapping[str, Any], /, **kwargs: Any
) -> M:
    r"""Try to initialize a module from a config."""
    match obj:
        case nn.Module() as module:
            if kwargs:
                raise ValueError(f"Unexpected {kwargs=}")
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
