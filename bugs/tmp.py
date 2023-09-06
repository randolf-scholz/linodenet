import typing
from collections.abc import Iterable, Mapping, Sequence
from typing import TypeAlias, TypeVar, Union, overload

T = TypeVar("T")
Nested: TypeAlias = T | Mapping[str, "Nested[T]"] | Iterable["Nested[T]"]


def foo(x: float) -> int:
    return 42


# fmt: off
@overload
def sum_recursive(x: int) -> int: ...
@overload
def sum_recursive(x: Mapping[str, Nested[int]]) -> int: ...
@overload
def sum_recursive(x: Iterable[Nested[int]]) -> int: ...
# fmt: on
def sum_recursive(x):
    total: int = 0
    reveal_type(x)
    match x:
        case int() as number:
            total += number
        case Mapping() as mapping:
            reveal_type(mapping)
            for key in mapping:
                total += sum_recursive(mapping[key])  # error
        case Iterable() as iterable if not isinstance(iterable, Mapping):
            for item in iterable:
                total += sum_recursive(item)

    return total


#
# Union[
#     Mapping[
#         str,
#         Union[
#             T @ Nested,
#             Mapping[str, Nested],
#             Iterable[Nested],
#         ],
#     ],
#     Mapping[
#         Union[
#             int,
#             Mapping[str, T @ Nested | Mapping[str, Nested] | Iterable[Nested]],
#             Iterable[T @ Nested | Mapping[str, Nested] | Iterable[Nested]],
#         ],
#         Unknown,
#     ],
# ]


#
# Mapping[str, T @ Nested | Mapping[str, Nested] | Iterable[Nested]] | Mapping[
#     Tensor
#     | Mapping[str, T @ Nested | Mapping[str, Nested] | Iterable[Nested]]
#     | Iterable[T @ Nested | Mapping[str, Nested] | Iterable[Nested]],
#     Unknown,
# ]
