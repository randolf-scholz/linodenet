r"""Test if typing_extensions is compatible with torch.jit.script."""

from tempfile import TemporaryDirectory
from typing import Optional

import pytest
import torch
from torch import Tensor, jit, nn
from typing_extensions import Optional as OptionalExt


def foo(x: Tensor, y: Optional[Tensor] = None) -> Tensor:
    r"""Dummy function."""
    if y is None:
        return x
    return x + y


def bar(x: Tensor, y: OptionalExt[Tensor] = None) -> Tensor:
    r"""Dummy function."""
    if y is None:
        return x
    return x + y


class Foo(nn.Module):
    r"""Dummy module."""

    def forward(self, x: Tensor, y: Optional[Tensor] = None) -> Tensor:
        if y is None:
            return x
        return x + y


class Bar(nn.Module):
    r"""Dummy module."""

    def forward(self, x: Tensor, y: OptionalExt[Tensor] = None) -> Tensor:
        if y is None:
            return x
        return x + y


@pytest.mark.xfail(reason="https://github.com/pytorch/pytorch/issues/119192.")
def test_function_typing():
    x = torch.randn(3, 4)
    y = torch.randn(3, 4)
    z = x + y

    assert torch.equal(foo(x), x)
    assert torch.equal(foo(x, y), z)

    # test with scripted
    scripted_foo = jit.script(foo)
    assert isinstance(scripted_foo, jit.ScriptFunction)
    assert torch.equal(scripted_foo(x), x)
    assert torch.equal(scripted_foo(x, y), z)

    # serialize and deserialize
    with TemporaryDirectory() as folder:
        fname = f"{folder}/foo.pt"
        jit.save(scripted_foo, fname)
        foo_loaded = jit.load(fname)

    # assert isinstance(foo_loaded, torch.jit.ScriptFunction)  # ScriptModule!
    assert torch.equal(foo_loaded(x), x)
    assert torch.equal(foo_loaded(x, y), z)


def test_module_typing():
    x = torch.randn(3, 4)
    y = torch.randn(3, 4)
    z = x + y

    module = Foo()

    assert torch.equal(module(x), x)
    assert torch.equal(module(x, y), z)

    # test with scripted
    scripted_foo = jit.script(module)
    assert isinstance(scripted_foo, jit.ScriptModule)
    assert torch.equal(scripted_foo(x), x)
    assert torch.equal(scripted_foo(x, y), z)

    # serialize and deserialize
    with TemporaryDirectory() as folder:
        fname = f"{folder}/foo.pt"
        jit.save(scripted_foo, fname)
        foo_loaded = jit.load(fname)

    assert isinstance(foo_loaded, jit.ScriptModule)
    assert torch.equal(foo_loaded(x), x)
    assert torch.equal(foo_loaded(x, y), z)


@pytest.mark.xfail(reason="https://github.com/pytorch/pytorch/issues/119192.")
def test_function_typing_extensions():
    x = torch.randn(3, 4)
    y = torch.randn(3, 4)
    z = x + y

    assert torch.equal(bar(x), x)
    assert torch.equal(bar(x, y), x + y)

    # test with scripted
    scripted_bar = jit.script(bar)
    assert torch.equal(scripted_bar(x), x)
    assert torch.equal(scripted_bar(x, y), z)

    # serialize and deserialize
    with TemporaryDirectory() as folder:
        fname = f"{folder}/bar.pt"
        jit.save(scripted_bar, fname)
        bar_loaded = jit.load(fname)

    assert torch.equal(bar_loaded(x), x)
    assert torch.equal(bar_loaded(x, y), x + y)


def test_module_typing_extensions():
    x = torch.randn(3, 4)
    y = torch.randn(3, 4)
    z = x + y

    module = Bar()

    assert torch.equal(module(x), x)
    assert torch.equal(module(x, y), x + y)

    # test with scripted
    scripted_bar = jit.script(module)
    assert torch.equal(scripted_bar(x), x)
    assert torch.equal(scripted_bar(x, y), z)

    # serialize and deserialize
    with TemporaryDirectory() as folder:
        fname = f"{folder}/bar.pt"
        jit.save(scripted_bar, fname)
        bar_loaded = jit.load(fname)

    assert torch.equal(bar_loaded(x), x)
    assert torch.equal(bar_loaded(x, y), x + y)
