from tempfile import TemporaryFile
from typing import Final

import pytest
import torch
from torch import Tensor, jit, nn


def check_getitem(module: nn.ModuleList) -> None:
    # check __getitem__(int)
    assert isinstance(module[0], nn.Module)
    # check __getitem__(slice)
    assert isinstance(module[2:], nn.Module)


def check_iter(module: nn.ModuleList) -> None:
    # check __iter__
    for m in module:
        assert isinstance(m, nn.Module)


class MyModuleList(nn.ModuleList): ...


class UsesModuleList(nn.Module):
    def __init__(self, modules: list[nn.Module]) -> None:
        super().__init__()
        self.components = nn.ModuleList(modules)

    def forward(self, x: Tensor) -> Tensor:
        for m in self.components:
            x = m(x)
        return x


class UsesMyModuleList(nn.Module):
    def __init__(self, modules: list[nn.Module]) -> None:
        super().__init__()
        self.components = MyModuleList(modules)

    def forward(self, x: Tensor) -> Tensor:
        for m in self.components:
            x = m(x)
        return x


@pytest.mark.parametrize("interface", ["getitem", "iter"])
@pytest.mark.parametrize("stage", ["initialized", "scripted", "reloaded"])
@pytest.mark.parametrize("cls", [nn.ModuleList, MyModuleList])
def test_jit_modulelist(cls: type, stage: str, interface: str) -> None:
    # simple MLP
    modules = [nn.Linear(3, 3), nn.ReLU(), nn.Linear(3, 3), nn.ReLU()]
    module = cls(modules)
    scripted = jit.script(module)
    x = torch.randn(5, 3)

    # save and load
    with TemporaryFile() as f:
        jit.save(scripted, f)
        f.seek(0)
        reloaded = jit.load(f)

    match stage:
        case "initialized":
            target = module
        case "scripted":
            target = scripted
        case "reloaded":
            target = reloaded

    match interface:
        case "getitem":
            check_getitem(target)
        case "iter":
            check_iter(target)
        case "forward":
            assert torch.equal(target(x), module(x))


@pytest.mark.parametrize("interface", ["forward"])
@pytest.mark.parametrize("stage", ["initialized", "scripted", "reloaded"])
@pytest.mark.parametrize("cls", [UsesModuleList, UsesMyModuleList])
def test_jit_iter_modulelist_in_scripted_forward(
    cls: type, stage: str, interface: str
) -> None:
    # simple MLP
    modules = [nn.Linear(3, 3), nn.ReLU(), nn.Linear(3, 3), nn.ReLU()]
    module = cls(modules)
    scripted = jit.script(module)
    x = torch.randn(5, 3)

    # save and load
    with TemporaryFile() as f:
        jit.save(scripted, f)
        f.seek(0)
        reloaded = jit.load(f)

    match stage:
        case "initialized":
            target = module
        case "scripted":
            target = scripted
        case "reloaded":
            target = reloaded

    match interface:
        case "getitem":
            check_getitem(target)
        case "iter":
            check_iter(target)
        case "forward":
            assert torch.equal(target(x), module(x))


class Mixin(nn.Module):
    input_size: Final[int]
    hidden_size: Final[int]

    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.layer = nn.Linear(input_size, hidden_size)

    def forward(self, x: Tensor) -> Tensor:
        return self.layer(x)


# NOTE: Always put nn.ModuleList last, otherwise __init__ won't work, as
#   nn.ModuleList.__init__ tries to call Mixin.__init__ with the wrong arguments.
#   This is because


class MySequential(nn.ModuleList, Mixin):
    def __init__(self, modules: list[nn.Module], *, shape: tuple[int, int]) -> None:
        nn.ModuleList.__init__(self, modules)
        Mixin.__init__(self, input_size=shape[0], hidden_size=shape[1])
        # Mixin.__init__(self, input_size=shape[0], hidden_size=shape[1])

    def forward(self, x: Tensor) -> Tensor:
        for m in self:
            x = m(x)
        return self.layer(x)


def test_jit_mixin() -> None:
    # simple MLP
    modules = [nn.Linear(3, 3), nn.ReLU(), nn.Linear(3, 3), nn.ReLU()]
    print(f"{[t.__name__ for t in MySequential.__mro__]}")
    module = MySequential(modules, shape=(3, 3))
    scripted = jit.script(module)
    x = torch.randn(5, 3)
    # ['MySequential', 'Mixin', 'ModuleList', 'Module', 'object']
    # ['MySequential', 'ModuleList', 'Mixin', 'Module', 'object']
    # save and load
    with TemporaryFile() as f:
        jit.save(scripted, f)
        f.seek(0)
        reloaded = jit.load(f)

    assert torch.equal(reloaded(x), module(x))

    assert isinstance(reloaded.layer, nn.Module)
    assert isinstance(reloaded.input_size, int)
    assert isinstance(reloaded.hidden_size, int)
