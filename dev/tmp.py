from collections.abc import Callable
from typing import Any, Protocol, reveal_type, runtime_checkable


@runtime_checkable
class AnyCallable(Protocol):
    # equivalent to Callable[..., Any]
    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...


class ConcreteCallable:
    def __call__(self, x: object, /) -> int: ...


class TestObject:  # test with argument of type `object`
    def test_callable(self, x: object) -> None:
        if callable(x):
            reveal_type(x)

    def test_isinstance_callable(self, x: object) -> None:
        if isinstance(x, Callable):
            reveal_type(x)

    def test_isinstance_protocol(self, x: object) -> None:
        if isinstance(x, AnyCallable):
            reveal_type(x)

    def test_match_callable(self, x: object) -> None:
        match x:
            case Callable() as fn:
                reveal_type(fn)

    def test_match_protocol(self, x: object) -> None:
        match x:
            case AnyCallable() as fn:
                reveal_type(fn)


class TestCallable:  # test with argument of type `Callable[[object], int]`
    def test_callable(self, x: Callable[[object], int]) -> None:
        if callable(x):
            reveal_type(x)

    def test_isinstance_callable(self, x: Callable[[object], int]) -> None:
        if isinstance(x, Callable):
            reveal_type(x)

    def test_isinstance_protocol(self, x: Callable[[object], int]) -> None:
        if isinstance(x, AnyCallable):
            reveal_type(x)

    def test_match_callable(self, x: Callable[[object], int]) -> None:
        match x:
            case Callable() as fn:
                reveal_type(fn)

    def test_match_protocol(self, x: Callable[[object], int]) -> None:
        match x:
            case AnyCallable() as fn:
                reveal_type(fn)


class TestConcreteFunction:  # test with argument of type `ConcreteCallable`
    def test_callable(self, x: ConcreteCallable) -> None:
        if callable(x):
            reveal_type(x)

    def test_isinstance_callable(self, x: ConcreteCallable) -> None:
        if isinstance(x, Callable):
            reveal_type(x)

    def test_isinstance_protocol(self, x: ConcreteCallable) -> None:
        if isinstance(x, AnyCallable):
            reveal_type(x)

    def test_match_callable(self, x: ConcreteCallable) -> None:
        match x:
            case Callable() as fn:
                reveal_type(fn)

    def test_match_protocol(self, x: ConcreteCallable) -> None:
        match x:
            case AnyCallable() as fn:
                reveal_type(fn)
