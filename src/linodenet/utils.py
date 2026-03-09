r"""Utility functions."""

__all__ = [
    # Types
    "NestedDict",
    "NestedMapping",
    # Functions
    "deep_dict_update",
    "flatten_dict",
    "get_module",
    "is_allcaps",
    "is_dunder",
    "is_private",
    "unflatten_dict",
]

from collections.abc import Callable, Iterable, Mapping
from copy import deepcopy
from typing import Any, cast, overload


def is_allcaps(s: str, /) -> bool:
    r"""Check if a string is all caps."""
    return s.isidentifier() and s.isupper() and s.isalpha()


def is_dunder(s: str, /) -> bool:
    r"""Check if name is a dunder method."""
    return s.isidentifier() and s.startswith("__") and s.endswith("__")


def is_private(s: str, /) -> bool:
    r"""Check if name is a private method."""
    return s.isidentifier() and s.startswith("_") and not s.startswith("__")


def get_module(obj_ref: object, /) -> str:
    return obj_ref.__module__.rsplit(".", maxsplit=1)[-1]


def deep_dict_update(d: dict, new: Mapping, /, *, inplace: bool = False) -> dict:
    r"""Update nested dictionary recursively in-place with new dictionary.

    References:
        https://stackoverflow.com/a/30655448/9318372
    """
    if not inplace:
        d = deepcopy(d)

    for key, value in new.items():
        match value:
            # recurse on non-empty mapping
            case Mapping() as mapping if mapping:  # non-empty mapping
                subdict = d.get(key, {})
                d[key] = deep_dict_update(subdict, mapping, inplace=True)
            # update value for the given key
            case _:
                d[key] = value
    return d


type NestedDict[K, V] = dict[K, V | "NestedDict[K, V]"]
r"""Generic Type Alias for nested `dict`."""
type NestedMapping[K, V] = Mapping[K, V | "NestedMapping[K, V]"]
r"""Generic Type Alias for nested `Mapping`."""


@overload
def flatten_dict(
    d: NestedMapping[str, Any],
    /,
    *,
    join_fn: Callable[[Iterable[str]], str] = ...,
    split_fn: Callable[[str], Iterable[str]] = ...,
    recursive: bool | int = ...,
) -> dict[str, Any]: ...
@overload
def flatten_dict[K, K2](
    d: NestedMapping[K, Any],
    /,
    *,
    join_fn: Callable[[Iterable[K]], K2],
    split_fn: Callable[[K2], Iterable[K]],
    recursive: bool | int = ...,
) -> dict[K2, Any]: ...
def flatten_dict[K, K2](
    d: NestedMapping[K, Any],
    /,
    *,
    join_fn: Callable[[Iterable[K]], K2] = cast("Any", ".".join),  # noqa: B008
    split_fn: Callable[[K2], Iterable[K]] = cast("Any", lambda s: s.split(".")),  # noqa: B008
    recursive: bool | int = True,
) -> dict[K2, Any]:
    r"""Flatten dictionaries recursively.

    Args:
        d: dictionary to flatten
        recursive: whether to flatten recursively.
            If `recursive` is an integer, flattens that many levels.
        join_fn: function to join keys
            Defaults to ``'.'.join``, implicitly assuming that all keys are strings.
        split_fn: function to split keys
            Defaults to ``str.split('.')``, implicitly assuming that all keys are strings.

    Example: flattening with string keys.
        When ``join_fn`` and ``split_fn`` are not provided, they default to
        ``join_fn = ".".join`` and ``split_fn = lambda s: s.split(".")``,
        implicitly assuming that all keys are strings.
        This will combine string keys like ``"a"`` and ``"b"`` into ``"a.b"``.

        >>> flatten_dict({"a": {"b": 1, "c": 2}})
        {'a.b': 1, 'a.c': 2}

        >>> flatten_dict({"a": {"b": {"x": 2}, "c": 2}})
        {'a.b.x': 2, 'a.c': 2}

    Example: flattening with custom key functions.
        Using ``join_fn = tuple`` and ``split_fn = lambda s: s`` will combine
        keys like ``("a", "b")`` and ``("a", "c")`` into ``("a", "b", "c")``.
        This choice works for arbitrary key types.

        >>> flatten_dict({"a": {"b": 1, "c": 2}}, join_fn=tuple, split_fn=lambda x: x)
        {('a', 'b'): 1, ('a', 'c'): 2}

        >>> flatten_dict(
        ...     {"a": {"b": {"x": 2}, "c": 2}},
        ...     join_fn=tuple,
        ...     split_fn=lambda x: x,
        ... )
        {('a', 'b', 'x'): 2, ('a', 'c'): 2}

    Example: partial flattening with ``recursive``.
        >>> flatten_dict({"a": {"i": {"x": 0}, "b": {"y": 1}}})
        {'a.i.x': 0, 'a.b.y': 1}

        >>> flatten_dict(
        ...     {"a": {"i": {"x": 0}, "b": {"y": 1}}},
        ...     recursive=2,
        ... )
        {'a.i': {'x': 0}, 'a.b': {'y': 1}}

        >>> flatten_dict(
        ...     {"a": {"i": {"x": 0}, "b": {"y": 1}}},
        ...     recursive=1,
        ... )
        {'a': {'i': {'x': 0}, 'b': {'y': 1}}}
    """
    recursive = recursive if isinstance(recursive, bool) else recursive - 1
    result: dict[K2, Any] = {}
    for key, item in d.items():
        if recursive and isinstance(item, Mapping):
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


@overload
def unflatten_dict(
    d: Mapping[str, Any],
    /,
    *,
    join_fn: Callable[[Iterable[str]], str] = ...,
    split_fn: Callable[[str], Iterable[str]] = ...,
    recursive: bool | int = ...,
) -> NestedDict[str, Any]: ...
@overload
def unflatten_dict[K, K2](
    d: Mapping[K2, Any],
    /,
    *,
    join_fn: Callable[[Iterable[K]], K2],
    split_fn: Callable[[K2], Iterable[K]],
    recursive: bool | int = ...,
) -> NestedDict[K, Any]: ...
def unflatten_dict[K, K2](
    d: Mapping[K2, Any],
    /,
    *,
    recursive: bool | int = True,
    join_fn: Callable[[Iterable[K]], K2] = cast("Any", ".".join),  # noqa: B008
    split_fn: Callable[[K2], Iterable[K]] = cast("Any", lambda s: s.split(".")),  # noqa: B008
) -> NestedDict[K, Any]:
    r"""Unflatten dictionaries recursively.

    Example: Unflattening with string keys.
        When ``join_fn`` and ``split_fn`` are not provided, they default to
        ``join_fn = ".".join`` and ``split_fn = lambda s: s.split(".")``,
        implicitly assuming that all keys are strings.
        This will split up keys like ``"a.b"`` into ``{"a": {"b": ...}}``.
        >>> unflatten_dict({"a.b": 1, "a.c": 2})
        {'a': {'b': 1, 'c': 2}}

    Example: unflattening with custom join function.
        Using ``join_fn = tuple`` and ``split_fn = lambda s: s`` will split up
        keys like ``("a", "b", "c")`` into ``{"a": {"b": {"c": ...}}}``.
        >>> unflatten_dict(
        ...     {("a", 17): "foo", ("a", 18): "bar"},
        ...     join_fn=tuple,
        ...     split_fn=lambda x: x,
        ... )
        {'a': {17: 'foo', 18: 'bar'}}

    Example: partial unflattening with ``recursive``.
        >>> unflatten_dict({"a.b.c.d": 0, "a.x.y.z": 1})
        {'a': {'b': {'c': {'d': 0}}, 'x': {'y': {'z': 1}}}}

        >>> unflatten_dict({"a.b.c.d": 0, "a.x.y.z": 1}, recursive=2)
        {'a': {'b': {'c.d': 0}, 'x': {'y.z': 1}}}

        >>> unflatten_dict({"a.b.c.d": 0, "a.x.y.z": 1}, recursive=1)
        {'a': {'b.c.d': 0, 'x.y.z': 1}}
    """
    recursive = recursive if isinstance(recursive, bool) else recursive - 1
    result: dict[K, Any] = {}
    for key, item in d.items():
        outer_key, *inner_keys = split_fn(key)
        if inner_keys:
            result.setdefault(outer_key, {})
            if not isinstance(result[outer_key], dict):
                raise KeyError(f"Key conflict at {outer_key}! Cannot unflatten.")
            if recursive:
                result[outer_key] |= unflatten_dict(
                    {join_fn(inner_keys): item},
                    recursive=recursive,
                    split_fn=split_fn,
                    join_fn=join_fn,
                )
            else:
                result[outer_key] |= {join_fn(inner_keys): item}
        elif outer_key in result:
            raise KeyError(f"Key conflict at {outer_key}! Cannot unflatten.")
        else:
            result[outer_key] = item
    return result


r"""Progress bar update interval in milliseconds."""
