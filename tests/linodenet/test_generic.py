r"""Tests for the generic types."""

from collections.abc import ItemsView, Iterator, KeysView, ValuesView
from typing import TYPE_CHECKING, assert_type

import torch
from torch import Tensor, nn

from linodenet.generic import ModuleMapping, ModuleSequence
from linodenet.testing import check_jit_serializable
from linodenet.testing._testing import assert_jit_compatible

BATCH_SIZE = 5


def test_module_sequence_types() -> None:
    D = 3
    DIM_OUT = 2
    m1 = nn.Linear(D, D)
    m2 = nn.Linear(D, DIM_OUT)
    m = ModuleSequence([m1, m2])

    if TYPE_CHECKING:
        assert_type(m, ModuleSequence[nn.Linear])
        assert_type(m[0], nn.Linear)
        assert_type(m[1:], ModuleSequence[nn.Linear])
        assert_type(iter(m), Iterator[nn.Linear])

        for module in m:
            assert_type(module, nn.Linear)
        for module in reversed(m):
            assert_type(module, nn.Linear)
    else:
        assert type(m) is ModuleSequence
        assert type(m[0]) is nn.Linear
        assert type(m[1:]) is ModuleSequence

        for module in m:
            assert type(module) is nn.Linear
        for module in reversed(m):
            assert type(module) is nn.Linear


def test_module_mapping_types() -> None:
    D = 3
    DIM_OUT = 2
    m1 = nn.Linear(D, D)
    m2 = nn.Linear(D, DIM_OUT)
    m = ModuleMapping({"m1": m1, "m2": m2})

    if TYPE_CHECKING:
        assert_type(m, ModuleMapping[nn.Linear])
        assert_type(m["m1"], nn.Linear)
        assert_type(m["m2"], nn.Linear)
        assert_type(iter(m), Iterator[str])
        assert_type(m.keys(), KeysView[str])
        assert_type(m.values(), ValuesView[nn.Linear])
        assert_type(m.items(), ItemsView[str, nn.Linear])

        for key in m:
            assert_type(key, str)
        for key in m.keys():  # noqa: SIM118
            assert_type(key, str)
        for module in m.values():
            assert_type(module, nn.Linear)
        for key, module in m.items():
            assert_type(key, str)
            assert_type(module, nn.Linear)
    else:
        assert type(m) is ModuleMapping
        assert type(m["m1"]) is nn.Linear
        assert type(m["m2"]) is nn.Linear

        assert type(iter(m)) is type(iter({}))
        assert type(m.keys()) is type({}.keys())
        assert type(m.values()) is type({}.values())
        assert type(m.items()) is type({}.items())

        assert isinstance(iter(m), Iterator)
        assert isinstance(m.keys(), KeysView)
        assert isinstance(m.values(), ValuesView)
        assert isinstance(m.items(), ItemsView)

        for key in m:
            assert type(key) is str
        for key in m.keys():  # noqa: SIM118
            assert type(key) is str
        for module in m.values():
            assert type(module) is nn.Linear
        for key, module in m.items():
            assert type(key) is str
            assert type(module) is nn.Linear


def test_sequence_jit() -> None:
    class Foo(ModuleSequence):
        def forward(self, x: Tensor) -> Tensor:
            for module in self:
                x = module(x)
            return x

    DIM_IN = 5
    DIM_OUT = 2
    model = Foo([nn.Linear(DIM_IN, DIM_OUT), nn.Linear(DIM_OUT, DIM_OUT)])
    x = torch.randn(BATCH_SIZE, DIM_IN)

    assert_jit_compatible(model, args=(x,), kwargs={})


def test_mapping_jit() -> None:
    class Bar(ModuleMapping):
        def forward(self, x: Tensor) -> Tensor:
            outputs: list[Tensor] = []
            for module in self.values():
                outputs.append(module(x))  # noqa: PERF401
            # average the outputs
            return torch.stack(outputs, dim=-1).mean(dim=-1)

    DIM_IN = 5
    DIM_OUT = 2
    model = Bar({"m1": nn.Linear(DIM_IN, DIM_OUT), "m2": nn.Linear(DIM_IN, DIM_OUT)})
    x = torch.randn(BATCH_SIZE, DIM_IN)

    assert_jit_compatible(model, args=(x,), kwargs={})
