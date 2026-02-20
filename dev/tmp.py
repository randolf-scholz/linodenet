from typing import TypedDict, assert_never, assert_type


class D(TypedDict):
    name: str
    value: int


def test_match_key(x: D | int) -> None:
    match x:
        case {"name": _}:
            # equivalent to (isinstance(x, Mapping) and "name" in x)
            assert_type(x, D)
        case _:
            assert_type(x, int)


def test_match_key_and_value(x: D | int) -> None:
    match x:
        case {"name": str()}:
            # equivalent to (isinstance(x, Mapping)
            # and "name" in x and isinstance(x["name"], str))
            assert_type(x, D)
        case _:
            assert_type(x, int)


def test_non_match(x: D | int) -> None:
    match x:
        case {"value": str()}:
            # equivalent to (isinstance(x, Mapping)
            # and "value" in x and isinstance(x["value"], str))
            assert_never(x)
        case _:
            assert_type(x, D | int)
