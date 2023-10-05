"""Some tests regarding the JIT compiler."""


from typing import Any

import pytest
import torch
from torch import Tensor, jit, nn
from torch.nn.utils import parametrize as torch_parametrize

from linodenet.testing import check_combined


class ModuleWithSlots(nn.Module):
    """Module with slots."""

    __slots__ = ("foo", "bar")

    foo: Tensor
    bar: Tensor

    def __init__(self, foo, bar):
        super().__init__()
        self.foo = foo
        self.bar = bar

    def forward(self, x: Tensor) -> Tensor:
        return self.foo * x + self.bar


class ModuleWithoutSlots(nn.Module):
    """Module without slots."""

    foo: Tensor
    bar: Tensor

    def __init__(self, foo, bar):
        super().__init__()
        self.foo = foo
        self.bar = bar

    def forward(self, x: Tensor) -> Tensor:
        return self.foo * x + self.bar


class AddPostInit(type):
    """Adds a post-init hook to the class."""

    def __init__(
        cls,
        name: str,
        bases: tuple[type, ...],
        namespace: dict[str, Any],
        /,
        **kwds: Any,
    ) -> None:
        super().__init__(name, bases, namespace, **kwds)
        if not hasattr(cls, "__post_init__") and "__post_init__" not in namespace:
            raise AttributeError(f"Class {cls} does not have a __post_init__ method")

    def __call__(cls, *args, **kwargs):
        obj = super().__call__(*args, **kwargs)
        obj.__post_init__()
        return obj


class ModuleWithMetaclass(nn.Module, metaclass=AddPostInit):
    """A module with a metaclass."""

    weight: Tensor

    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.layer = nn.Linear(input_size, output_size)
        self.weight = self.layer.weight

    def __post_init__(self):
        if self.weight.shape[-1] != self.weight.shape[-2]:
            raise ValueError("Weight matrix must be square!")

    def forward(self, x: Tensor) -> Tensor:
        return self.layer(x)


class SubclassWithMetaclass(ModuleWithMetaclass, metaclass=AddPostInit):
    """A subclass."""


def test_post_init_jit() -> None:
    """Checks that models with meta classes can be scripted."""
    model = ModuleWithMetaclass(3, 3)
    check_combined(model, input_args=(torch.randn(2, 3),), test_jit=True)


def test_post_init() -> None:
    """Checks that metaclasses work as expected."""
    ModuleWithMetaclass(4, 4)

    with pytest.raises(ValueError):
        ModuleWithMetaclass(4, 5)

    SubclassWithMetaclass(4, 4)

    with pytest.raises(ValueError):
        SubclassWithMetaclass(4, 5)


def test_jit_error_slots() -> None:
    """Tests that a module with slots cannot be scripted."""
    foo = torch.randn(())
    bar = torch.randn(())

    # check that the module without slots can be scripted
    model_without_slots = ModuleWithoutSlots(foo, bar)
    jit.script(model_without_slots)

    # check that the module with slots cannot be scripted
    model_with_slots = ModuleWithSlots(foo, bar)
    with pytest.raises(RuntimeError):
        jit.script(model_with_slots)


def test_jit_errors_parametrize() -> None:
    """Tests that scripting fails for torch builtin parametrization."""

    class Symmetric(nn.Module):
        def forward(self, x: Tensor) -> Tensor:
            return x.triu() + x.triu(1).transpose(-1, -2)

    model = nn.Linear(5, 5)

    torch_parametrize.register_parametrization(model, "weight", Symmetric())

    scripted = jit.script(model)
    inputs = torch.randn(7, 5)

    with pytest.raises(jit.Error):  # type: ignore[type-var]
        scripted(inputs)
