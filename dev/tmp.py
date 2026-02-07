from typing import reveal_type


class Foo[T]:
    def __init__(self) -> None:
        pass


class Bar[S](Foo[S]):
    def __init__(self) -> None:
        reveal_type(Foo.__init__)
        reveal_type(Foo[S].__init__)
        reveal_type(Foo[int].__init__)  # E: BoundMethod
        Foo[S].__init__(self)
