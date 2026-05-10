from dataclasses import dataclass

class Foo[T: int | float | str]:

    def __init__(self, value: T) -> None:
        self.value = value

    @classmethod
    def as_int_cls(cls) -> Foo[int]:
        return cls(int())

    def as_int(self) -> Foo[int]:
        reveal_type(type(self))
        reveal_type(self.__class__)
