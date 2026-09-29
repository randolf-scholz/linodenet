from collections.abc import Mapping, Sequence
from typing import ClassVar, Protocol, Self, SupportsFloat, SupportsIndex

# mutable version of JSON
type JSON_SCALAR = None | bool | int | float | str
type JSON_ARRAY[T: JSON = JSON] = list[T]
type JSON_DICT[T: JSON = JSON] = dict[str, T]
type JSON = JSON_SCALAR | JSON_ARRAY[JSON] | JSON_DICT

# read-only version of JSON
type CFG_SCALAR = None | bool | int | float | str
type CFG_ARRAY[T: CFG] = Sequence[T]
type CFG_DICT[T: CFG] = Mapping[str, T]
type CFG = CFG_SCALAR | CFG_ARRAY[CFG] | CFG_DICT[CFG]


# no inductive types in python ...
type JSON_TENSOR = (
    list[float]
    | list[list[float]]
    | list[list[list[float]]]
    | list[list[list[list[float]]]]
    | list[list[list[list[list[float]]]]]
    | list[list[list[list[list[list[float]]]]]]
    | list[list[list[list[list[list[list[float]]]]]]]
    | list[list[list[list[list[list[list[list[float]]]]]]]]
)


class Config(Protocol):
    r"""Base class for configurable modules."""

    def __getitem__(self, key: str, /) -> JSON: ...
    def __or__(self, other: JSON, /) -> Self: ...
    def __ror__(self, other: JSON, /) -> Self: ...
    def export(self) -> JSON: ...


class ConfigurableModule(Protocol):
    CFG: ClassVar[type[Config]]
    config: CFG

    @classmethod
    def from_config(cls, config: CFG, /) -> Self: ...


# implementation


def export_value(value: object, /) -> JSON:
    r"""Export a value to a JSON-compatible format."""
    match value:
        case Sequence() as seq:
            return [export_value(item) for item in seq]
        case Mapping() as mapping:
            if not all(isinstance(_k, str) for _k in {type(key) for key in mapping}):
                raise TypeError("Keys of a mapping must be strings for JSON export.")
            return {str(key): export_value(val) for key, val in mapping.items()}
        case None:
            return None
        case bool(mask):
            return bool(mask)
        case SupportsIndex():
            return int(value)
        case SupportsFloat():
            return float(value)
        case str(name):
            return str(name)
        case _:
            raise TypeError(f"Unsupported value type: {type(value)}")
