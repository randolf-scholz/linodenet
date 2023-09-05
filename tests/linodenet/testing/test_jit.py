"""Some tests regarding the JIT compiler."""


import pytest
import torch
from torch import Tensor, jit, nn
from torch.nn.utils import parametrize as torch_parametrize


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

    with pytest.raises(jit.Error):
        scripted(inputs)
