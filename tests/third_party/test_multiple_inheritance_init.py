r"""Test multiple inheritance init."""

from typing import Any

import pytest


def test_init_order() -> None:
    r"""Test multiple inheritance init."""

    class A:
        def __init__(self) -> None:
            print("Entering A.__init__")
            super().__init__()
            print("Exiting A.__init__")

    class B:
        def __init__(self) -> None:
            print("Entering B.__init__")
            super().__init__()
            print("Exiting B.__init__")

    class C(A, B):
        def __init__(self) -> None:
            print("Entering C.__init__")
            super().__init__()
            print("Exiting C.__init__")

    class D(B, A):
        def __init__(self) -> None:
            print("Entering D.__init__")
            super().__init__()
            print("Exiting D.__init__")

    print("\n\nTesting C:")
    C()

    print("\n\nTesting D:")
    D()


def test_init() -> None:

    class A:
        def __init__(self, *, a: int) -> None:
            print("hello from A")
            self.a = a

    class B:
        def __init__(self, *, b: int) -> None:
            print("hello from B")
            self.b = b

    class C(A, B):
        def __init__(self, **kwargs: Any) -> None:
            super().__init__(**kwargs)

    assert C.__mro__ == (C, A, B, object)

    # This should work, super().__init__ picks second member of __mro__
    C(a=1)

    # This should NOT work
    with pytest.raises(TypeError):
        C(b=1)

    # This should NOT work
    with pytest.raises(TypeError):
        C(a=1, b=1)
