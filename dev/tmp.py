from collections.abc import Callable, Sequence
from typing import Any


def test(x: int | Callable[[], int]) -> int:
    if isinstance(x, int):
        return x
    if callable(x):
        return x()
    raise TypeError


def test2(x: Any) -> int:
    if isinstance(x, int):
        return x
    if callable(x):
        return reveal_type(x())
    raise TypeError
