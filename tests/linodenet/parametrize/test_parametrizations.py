r"""Test parametrization of modules."""

import pytest
import torch
from torch import nn

from linodenet.parametrize import PARAMETRIZATIONS, Parametrization


@pytest.mark.parametrize("name", PARAMETRIZATIONS)
def test_parametrization(name: str) -> None:
    r"""Test parametrization."""
    cls = PARAMETRIZATIONS[name]
    tensor = nn.Parameter(torch.randn(3, 3))
    parametrization = cls(tensor)
    assert isinstance(parametrization, Parametrization)


@pytest.mark.xfail(reason="https://github.com/python/typing/discussions/1941")
def test_foo() -> None:
    from typing import Protocol, runtime_checkable

    @runtime_checkable
    class HasWeight(Protocol):
        weight: nn.Parameter

    model = nn.Linear(3, 4)
    # fmt: off
    assert hasattr(model, "weight")                   # ✅
    assert isinstance(model.weight, nn.Parameter)     # ✅
    assert "weight" in model.__static_attributes__    # ✅
    assert "weight" in model.__annotations__          # ✅
    assert isinstance(model, HasWeight)               # ❌
    # fmt: on
