r"""Implements the BaseConfig class."""

__all__ = [
    # Classes
    "Config",
    "ConfigMetaclass",
]

from abc import ABCMeta
from collections.abc import Iterator, Mapping
from dataclasses import KW_ONLY, dataclass, field
from typing import Any, Self

from linodenet.utils._utils import is_allcaps, is_dunder


class ConfigMetaclass(ABCMeta):
    r"""Metaclass for `BaseConfig`."""

    _FORBIDDEN_FIELDS = {
        "clear",       # Removes all the elements from the dictionary
        "copy",        # Returns a copy of the dictionary
        "fromkeys",    # Returns a dictionary with the specified keys and value
        "get",         # Returns the value of the specified key
        "items",       # Returns a list containing a tuple for each key value pair
        "keys",        # Returns a list containing the dictionary's keys
        "pop",         # Removes the element with the specified key
        "popitem",     # Removes the last inserted key-value pair
        "setdefault",  # Returns the value of the specified key or set default
        "update",      # Updates the dictionary with the specified key-value pairs
        "values",      # Returns a list of all the values in the dictionary
    }  # fmt: skip

    def __new__(
        cls,
        name: str,
        bases: tuple[type, ...],
        attrs: dict[str, Any],
        /,
        **kwds: Any,
    ) -> type:
        r"""Create a new class, patch in dataclass fields, and return it."""
        if "__annotations__" not in attrs:
            attrs["__annotations__"] = {}

        config_type = super().__new__(cls, name, bases, attrs, **kwds)
        FIELDS = set(attrs["__annotations__"])

        # check forbidden fields
        FORBIDDEN_FIELDS = cls._FORBIDDEN_FIELDS & FIELDS
        if FORBIDDEN_FIELDS:
            raise ValueError(
                f"Fields {cls._FORBIDDEN_FIELDS!r} are not allowed! "
                f"Found {FORBIDDEN_FIELDS!r}"
            )

        # check for dunder fields
        DUNDER_FIELDS = {key for key in FIELDS if is_dunder(key)}
        if DUNDER_FIELDS:
            raise ValueError(f"Dunder fields are not allowed!Found {DUNDER_FIELDS!r}.")

        # check all caps fields
        ALLCAPS_FIELDS = {key for key in FIELDS if is_allcaps(key)}
        if ALLCAPS_FIELDS:
            raise ValueError(f"ALLCAPS fields are reserved!Found {ALLCAPS_FIELDS!r}.")

        NAME = config_type.__qualname__.rsplit(".", maxsplit=1)[0]
        patched_fields = [
            ("_", KW_ONLY),
            ("NAME", str, field(default=NAME)),
            ("MODULE", str, field(default=attrs["__module__"])),
        ]

        for key, hint, *value in patched_fields:
            config_type.__annotations__[key] = hint
            if value:
                setattr(config_type, key, value[0])

        return dataclass(config_type, eq=False, frozen=True)  # type: ignore[call-overload]


class Config(Mapping[str, Any], metaclass=ConfigMetaclass):
    r"""Base Config."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        r"""Initialize the dictionary."""
        self.__dict__.update(*args, **kwargs)

    def __iter__(self) -> Iterator[str]:
        r"""Return an iterator over the keys of the dictionary."""
        return iter(self.__dict__)

    def __getitem__(self, key: str) -> Any:
        r"""Return the value of the specified key."""
        return self.__dict__[key]

    def __len__(self) -> int:
        r"""Return the number of items in the dictionary."""
        return len(self.__dict__)

    def __hash__(self) -> int:
        r"""Return permutation-invariant hash on `items()`."""
        # Alternative:
        # https://stackoverflow.com/questions/1151658/python-hashable-dicts
        # return hash((frozenset(self), frozenset(self.itervalues())))
        return hash(frozenset(self.items()))

    def __or__(self, other: Mapping[str, Any]) -> Self:
        r"""Return a new dictionary with the keys from both dictionaries."""
        res: dict = {}
        res.update(self)
        res.update(other)
        return self.__class__(**res)
