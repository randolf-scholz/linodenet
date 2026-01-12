from collections.abc import Iterable, Sequence
from typing import Never, overload


class Foo: ...


class Bar(Foo): ...


class FooSequence[F: Foo](Foo, Sequence[F]):
    @overload
    def __init__(self: "FooSequence[Never]", /) -> None: ...

    @overload
    def __init__(self, modules: Iterable[F], /) -> None: ...

    def __init__(self, modules: Iterable[F] = (), /) -> None: ...


class BarSequence[B: Bar](Bar, FooSequence[B]):
    def __init__(self, modules: Iterable[B] = (), /) -> None:
        FooSequence[B].__init__(self, modules)
