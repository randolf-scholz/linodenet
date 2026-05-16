r"""test whether module with custom metaclass works with ``torch.jit.script``."""

from typing import Any, Never

import torch
from torch import Tensor, jit, nn


class AddPostInitMeta(type):
    r"""A metaclass that adds a __post_init__ method if not present."""

    @staticmethod
    def __post_init__(_: Never, /) -> None:
        pass

    def __call__[T](cls: type[T], *args: Any, **kwargs: Any) -> T:
        instance = super().__call__(*args, **kwargs)
        instance.__post_init__()
        return instance

    def __new__(
        cls,
        name: str,
        bases: tuple[type, ...],
        namespace: dict[str, Any],
        /,
        **kwargs: Any,
    ) -> type:
        new: type[Any] = super().__new__(cls, name, bases, namespace, **kwargs)
        if getattr(new, "__post_init__", None) is None:
            namespace["__post_init__"] = AddPostInitMeta.__post_init__
            new = super().__new__(cls, name, bases, namespace, **kwargs)

        return new


class WithoutPostInit(nn.Module, metaclass=AddPostInitMeta):
    r"""Dummy module that adds a constant value to the input."""

    value: float

    def __init__(self, value: float) -> None:
        super().__init__()
        self.value = value

    def forward(self, x: Tensor) -> Tensor:
        return x + self.value

    def __post_init__(self) -> None:
        pass


class WithPostInit(WithoutPostInit):
    def __post_init__(self) -> None:
        self.value *= 2.0


def test_my_module() -> None:
    without_double = WithoutPostInit(5.0)
    with_double = WithPostInit(5.0)
    x = torch.tensor(0.0)
    assert without_double(x).item() == x + 5.0
    assert with_double(x).item() == x + 10.0

    scripted_without_double = jit.script(WithoutPostInit(5.0))
    scripted_with_double = jit.script(WithPostInit(5.0))
    x = torch.tensor(0.0)
    assert scripted_without_double(x).item() == x + 5.0
    assert scripted_with_double(x).item() == x + 10.0
