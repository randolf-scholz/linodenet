r"""Tests for torch.parametrizations module."""

import pytest
import torch
from torch import Tensor, nn
from torch.nn.utils import parametrizations, parametrize
from torch.optim import SGD

ATOL = 1e-6
RTOL = 1e-5


class Symmetric(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.triu() + x.triu(1).transpose(-1, -2)


class MyLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_features, out_features))

    def forward(self, x: Tensor) -> Tensor:
        return nn.functional.linear(x, self.weight)


def test_torch_parametrize_demo() -> None:
    layer = nn.Linear(3, 3)
    print(f"Unparametrized:\n{layer}")

    # apply parametrization
    # This injects a nn.ModuleDict called `parametrizations` into the layer,
    # which holds the parametrization modules.
    # It also changes layer.__class__ in order to inject the new property.
    parametrize.register_parametrization(layer, "weight", Symmetric())
    print(f"\nParametrized:\n{layer}")
    assert isinstance(layer.parametrizations, nn.ModuleDict)
    assert isinstance(layer, nn.Linear)
    assert type(layer) is not nn.Linear


def test_torch_parametrize_basic() -> None:
    n = 4
    model = nn.Linear(n, n)
    x = torch.randn(2, n)

    parametrized = parametrizations.orthogonal(model)
    _ = parametrized(x)
    Q = parametrized.weight
    assert isinstance(Q, Tensor)
    assert torch.allclose(Q @ Q.T, torch.eye(n), atol=ATOL, rtol=RTOL)


def test_torch_parametrize_cache_basic() -> None:
    pass


def test_torch_parametrize_compile() -> None:
    n = 4
    model = nn.Linear(n, n)
    x = torch.randn(2, n)

    # test parametrized version
    parametrized = parametrizations.orthogonal(model)
    y = parametrized(x)
    Q = parametrized.weight
    assert isinstance(Q, Tensor)
    assert torch.allclose(Q @ Q.T, torch.eye(n), atol=ATOL, rtol=RTOL)

    # test compiled version
    compiled = torch.compile(parametrized)
    assert torch.allclose(compiled(x), y)
    Q = compiled.weight  # type: ignore[attr-defined]
    assert isinstance(Q, Tensor)
    assert torch.allclose(Q @ Q.T, torch.eye(n), atol=ATOL, rtol=RTOL)

    # test training step
    optim = SGD(compiled.parameters(), lr=1e-3)  # type: ignore[attr-defined]
    loss_before = compiled(x).sum()
    loss_before.backward()
    optim.step()
    Q = compiled.weight  # type: ignore[attr-defined]
    assert isinstance(Q, Tensor)
    assert torch.allclose(Q @ Q.T, torch.eye(n), atol=ATOL, rtol=RTOL)

    loss_after = compiled(x).sum()
    assert loss_after < loss_before


def test_torch_parametrize_export() -> None:
    n = 4
    model = nn.Linear(n, n, bias=False)
    I = torch.eye(n, n)  # Hack to inspect the parametrized weight export.

    parametrized = model
    parametrize.register_parametrization(parametrized, "weight", Symmetric())
    y = parametrized(I)
    assert torch.allclose(model.weight, model.weight.T)

    exported = torch.export.export(model, args=(I,)).module()
    assert torch.allclose(exported(I), y)
    Q = exported(I)
    assert torch.allclose(Q, Q.T)

    # test training step
    optim = SGD(exported.parameters(), lr=1e-3)
    loss_before = exported(I).sum()
    loss_before.backward()
    optim.step()
    Q = exported(I)
    assert torch.allclose(Q, Q.T)

    loss_after = exported(I).sum()
    assert loss_after < loss_before


@pytest.mark.xfail(
    raises=torch.jit.Error, reason="TorchScript does not support parametrizations yet."
)
def test_torch_parametrize_jit() -> None:
    n = 4
    model = nn.Linear(n, n)
    x = torch.randn(2, n)

    parametrized = model
    parametrize.register_parametrization(parametrized, "weight", Symmetric())
    y = parametrized(x)
    assert torch.allclose(model.weight, model.weight.T)

    compiled = torch.jit.script(model)
    assert torch.allclose(compiled(x), y)
    assert torch.allclose(compiled.weight, compiled.weight.T)
