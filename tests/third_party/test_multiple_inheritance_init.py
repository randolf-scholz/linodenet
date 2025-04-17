r"""Test multiple inheritance init."""

from typing import Any

import pytest


class A:
    def __init__(self, *, a: int) -> None:
        self.a = a


class B:
    def __init__(self, *, b: int) -> None:
        self.b = b


class C(A, B):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)


def test_init() -> None:
    assert C.__mro__ == (C, A, B, object)

    # This should work, super().__init__ picks second member of __mro__
    C(a=1)

    # This should NOT work
    with pytest.raises(TypeError):
        C(b=1)

    # This should NOT work
    with pytest.raises(TypeError):
        C(a=1, b=1)
