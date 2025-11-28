r"""Utility functions."""

__all__ = [
    # types
    "SelfMap",
    # Classes
    # Functions
    "deep_dict_update",
    "flatten_dict",
    "get_module",
    "implements",
    "is_allcaps",
    "is_dunder",
    "is_private",
    "unflatten_dict",
    "signature",
]

from collections.abc import Callable, Iterable, Mapping
from copy import deepcopy
from typing import (
    Any,
    Protocol,
    TypeIs,
    cast,
    get_protocol_members,
    is_protocol,
    overload,
)

from typing_extensions import TypeForm

from linodenet.signature import parse_signature
from linodenet.types import NestedDict, NestedMapping


class SelfMap[T](Protocol):
    r"""A callable that returns the same type as its argument."""

    # TODO: make this generic and upper-bound T to the generic type.
    # alternatively, use signature def[S](T & S) -> (T & S)
    def __call__[S](self, arg: S, /) -> S: ...


def signature(sig: str, /) -> SelfMap:
    r"""To be used as a no-op decorator for annotating function signatures."""

    def decorator[Fn: Callable](fn: Fn) -> Fn:
        fn.signature = parse_signature(sig)  # type: ignore[attr-defined]  # pyright: ignore[reportFunctionMemberAccess]
        if isinstance(fn.__doc__, str):
            fn.__doc__ = fn.__doc__ + f"\n.. Signature:: ``{sig}``"
        return fn

    return cast("SelfMap", decorator)


@overload
def implements[T](protocol: TypeForm[T], /) -> SelfMap[type[T]]: ...
@overload
def implements[T](obj: object, protocol: type[T], /) -> TypeIs[T]: ...
def implements[T](
    obj: object, protocol: None | type[T] = None
) -> TypeIs[T] | SelfMap[T]:
    r"""Check if an object implements a protocol.

    Args:
        obj: object to check
        protocol: protocol class
    """
    if protocol is None:
        return lambda x: x

    if not isinstance(protocol, type) or not is_protocol(protocol):
        raise TypeError(f"{protocol} is not a protocol class.")

    return all(hasattr(obj, member) for member in get_protocol_members(protocol))


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


def flatten_dict[K, K2](
    d: NestedMapping[K, Any],
    /,
    *,
    recursive: bool | int = True,
    join_fn: Callable[[Iterable[K]], K2] = ".".join,  # type: ignore[assignment]
    split_fn: Callable[[K2], Iterable[K]] = lambda s: s.split("."),  # type: ignore[attr-defined]  # pyright: ignore[reportAttributeAccessIssue]
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

        >>> flatten_dict(
        ...     {"a": {17: "foo", 18: "bar"}}, join_fn=tuple, split_fn=lambda x: x
        ... )
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


def unflatten_dict[K, K2](
    d: Mapping[K2, Any],
    /,
    *,
    recursive: bool | int = True,
    join_fn: Callable[[Iterable[K]], K2] = ".".join,  # type: ignore[assignment]
    split_fn: Callable[[K2], Iterable[K]] = lambda s: s.split("."),  # type: ignore[attr-defined]  # pyright: ignore[reportAttributeAccessIssue]
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
        ...     {("a", 17): "foo", ("a", 18): "bar"},
        ...     join_fn=tuple,
        ...     split_fn=lambda x: x,
        ... )
        {'a': {17: 'foo', 18: 'bar'}}
    """
    recursive = recursive if isinstance(recursive, bool) else recursive - 1
    result: dict[K, Any] = {}
    for key, item in d.items():
        outer_key, *inner_keys = split_fn(key)
        if recursive and inner_keys:
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
