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
    "normalize_name",
    "resolve_name",
]

from collections.abc import Callable, Iterable, Mapping
from copy import deepcopy
from typing import Any, cast, overload


def is_allcaps(s: str, /) -> bool:
    r"""Check whether a string is an uppercase alphabetic Python identifier.

    Digits and underscores are rejected, even in otherwise valid identifiers.
    """
    return s.isidentifier() and s.isupper() and s.isalpha()


def is_dunder(s: str, /) -> bool:
    r"""Check whether an identifier starts and ends with two underscores."""
    return s.isidentifier() and s.startswith("__") and s.endswith("__")


def is_private(s: str, /) -> bool:
    r"""Check whether an identifier starts with exactly one underscore."""
    return s.isidentifier() and s.startswith("_") and not s.startswith("__")


def get_module(obj_ref: object, /) -> str:
    return obj_ref.__module__.rsplit(".", maxsplit=1)[-1]


def deep_dict_update(d: dict, new: Mapping, /, *, inplace: bool = False) -> dict:
    r"""Recursively merge a mapping into a dictionary.

    Args:
        d: The dictionary to update. Existing values corresponding to nonempty
            mappings in `new` must be dictionaries that can be updated recursively.
        new: Values to merge. Nonempty mappings are merged recursively; all other
            values, including empty mappings, replace the existing value.
        inplace: Whether to modify `d` directly. Otherwise, deep-copy `d` before
            merging. Replacement values from `new` are assigned without copying.

    Returns:
        The updated dictionary.

    References:
        [Recursive dictionary update](https://stackoverflow.com/a/30655448/9318372).
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
r"""Dictionary with keys of type `K` and values of type `V` or nested dictionaries."""
type NestedMapping[K, V] = Mapping[K, V | "NestedMapping[K, V]"]
r"""Mapping with keys of type `K` and values of type `V` or nested mappings."""


@overload
def flatten_dict(  # pyrefly: ignore[inconsistent-overload-default]
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
    join_fn: Callable[[Iterable[K]], K2] = cast("Any", ".".join),  # ruff: ignore[B008]
    split_fn: Callable[[K2], Iterable[K]] = cast("Any", lambda s: s.split(".")),  # ruff: ignore[B008]
    recursive: bool | int = True,
) -> dict[K2, Any]:
    r"""Flatten dictionaries recursively.

    Args:
        d: The mapping to flatten.
        recursive: Whether to flatten all nested mappings. `False` preserves
            nested values. A positive integer limits the number of key components
            in each path: `1` preserves nested values, and `2` flattens one level.
        join_fn: Function combining key components. Defaults to `".".join`,
            which requires string keys.
        split_fn: Function recovering key components from a joined key. Defaults
            to splitting a string at `"."`.

    Returns:
        A dictionary whose keys represent paths through `d`.

    Note:
        Joined paths must be unique; collisions overwrite earlier values.
        Empty nested mappings disappear when traversed recursively. For a
        reversible conversion, `split_fn` must invert `join_fn` on key paths.

    Examples:
        **String keys.**

        When `join_fn` and `split_fn` are not provided, they default to
        `join_fn = ".".join` and `split_fn = lambda s: s.split(".")`,
        implicitly assuming that all keys are strings.
        This combines successive keys `"a"` and `"b"` into the path `"a.b"`.

        ```pycon
        >>> flatten_dict({"a": {"b": 1, "c": 2}})
        {'a.b': 1, 'a.c': 2}

        >>> flatten_dict({"a": {"b": {"x": 2}, "c": 2}})
        {'a.b.x': 2, 'a.c': 2}

        ```

        **Custom key functions.**

        Using `join_fn = tuple` and `split_fn = lambda s: s` stores each path as
        a tuple of keys. This choice works for arbitrary hashable key types.

        ```pycon
        >>> flatten_dict({"a": {"b": 1, "c": 2}}, join_fn=tuple, split_fn=lambda x: x)
        {('a', 'b'): 1, ('a', 'c'): 2}

        >>> flatten_dict(
        ...     {"a": {"b": {"x": 2}, "c": 2}},
        ...     join_fn=tuple,
        ...     split_fn=lambda x: x,
        ... )
        {('a', 'b', 'x'): 2, ('a', 'c'): 2}

        ```

        **Partial flattening with `recursive`.**

        ```pycon
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

        ```
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
def unflatten_dict(  # pyrefly: ignore[inconsistent-overload-default]
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
    join_fn: Callable[[Iterable[K]], K2] = cast("Any", ".".join),  # ruff: ignore[B008]
    split_fn: Callable[[K2], Iterable[K]] = cast("Any", lambda s: s.split(".")),  # ruff: ignore[B008]
) -> NestedDict[K, Any]:
    r"""Unflatten dictionaries recursively.

    Args:
        d: The mapping whose keys represent paths to expand.
        recursive: Whether to expand paths fully. `False` splits each path once;
            a positive integer limits the number of splits along each path.
        join_fn: Function combining any unexpanded key components. Defaults to
            `".".join`, which requires string keys.
        split_fn: Function splitting a key into path components. Defaults to
            splitting a string at `"."`. Each key must yield at least one component.

    Returns:
        A nested dictionary reconstructed from the key paths.

    Raises:
        KeyError: If a path needs to descend through an existing non-dictionary
            value or a terminal key is already present.

    Examples:
        **String keys.**

        When `join_fn` and `split_fn` are not provided, they default to
        `join_fn = ".".join` and `split_fn = lambda s: s.split(".")`,
        implicitly assuming that all keys are strings.
        This expands paths like `"a.b"` into `{"a": {"b": ...}}`.

        ```pycon
        >>> unflatten_dict({"a.b": 1, "a.c": 2})
        {'a': {'b': 1, 'c': 2}}

        ```

        **Custom key functions.**

        Using `join_fn = tuple` and `split_fn = lambda s: s` expands tuple paths
        like `("a", "b", "c")` into `{"a": {"b": {"c": ...}}}`.

        ```pycon
        >>> unflatten_dict(
        ...     {("a", 17): "foo", ("a", 18): "bar"},
        ...     join_fn=tuple,
        ...     split_fn=lambda x: x,
        ... )
        {'a': {17: 'foo', 18: 'bar'}}

        ```

        **Partial unflattening with `recursive`.**

        ```pycon
        >>> unflatten_dict({"a.b.c.d": 0, "a.x.y.z": 1})
        {'a': {'b': {'c': {'d': 0}}, 'x': {'y': {'z': 1}}}}

        >>> unflatten_dict({"a.b.c.d": 0, "a.x.y.z": 1}, recursive=2)
        {'a': {'b': {'c.d': 0}, 'x': {'y.z': 1}}}

        >>> unflatten_dict({"a.b.c.d": 0, "a.x.y.z": 1}, recursive=1)
        {'a': {'b.c.d': 0, 'x.y.z': 1}}

        ```
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


def _camel_to_snake(name: str, /) -> str:
    r"""Convert `CamelCase` names to `snake_case`."""
    return "".join(["_" + c.lower() if c.isupper() else c for c in name]).lstrip("_")


def _snake_to_kebab(string: str) -> str:
    return string.replace("_", "-").lower()


def normalize_name(name: str, /) -> str:
    r"""Normalize a name to lowercase kebab-case after removing a leading `is_`.

    Each uppercase letter starts a new component, so acronyms are split into
    individual letters. Underscores become hyphens.
    """
    name = name.removeprefix("is_")
    name = _camel_to_snake(name)
    name = _snake_to_kebab(name)
    return name


def resolve_name[T](mapping: Mapping[str, T], name: str, /) -> T:
    canonical = normalize_name(name)

    for key, value in mapping.items():
        if normalize_name(key) == canonical:
            return value
    available = sorted(normalize_name(key) for key in mapping)
    raise KeyError(
        f"Item {name!r} ({canonical!r}) not found in mapping."
        f" Avaible canonical names: {available}."
    )
