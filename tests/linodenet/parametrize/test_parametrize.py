r"""Test parametrization of modules."""

import pytest
import torch
from torch import Tensor, nn
from torch._dynamo import OptimizedModule
from torch.linalg import matrix_norm
from torch.nn.functional import mse_loss
from torch.optim import SGD

from linodenet.mappings.projections import LipschitzBounded, Symmetric
from linodenet.parametrizations import (
    UpperTriangular,
    cached,
    get_parametrizations,
    is_parametrization,
    parametrized,
    register_optimizer_hook,
    register_parametrization,
    update_parametrizations,
)
from linodenet.testing import (
    all_close,
    assert_model_ok,
    is_symmetric,
    is_upper_triangular,
)
from linodenet.testing.utils import get_norm


def check_optimization(
    model: nn.Module, *, args: tuple[Tensor, ...], target: Tensor
) -> None:
    optimizer = SGD(model.parameters(), lr=0.1)

    with torch.no_grad():
        original_weights = [w.clone().detach() for w in model.parameters()]
        original_outputs = model(*args)
        original_loss = mse_loss(original_outputs, target)
        original_params = [w.clone().detach() for w in model.parameters()]

    # crucial, otherwise no update in the first iteration!
    update_parametrizations(model)

    # perform 1 training step
    model.zero_grad(set_to_none=True)
    outputs = model(*args)
    loss = get_norm(outputs)
    loss.backward()
    assert all(w.grad is not None for w in model.parameters() if w.requires_grad)
    assert loss.isfinite()
    optimizer.step()
    update_parametrizations(model)
    assert isinstance(model.weight, Tensor)
    assert is_upper_triangular(model.weight)
    assert not all_close(original_weights, list(model.parameters()))

    # check that the loss has decreased
    assert loss < original_loss
    # check that the outputs are different
    new_outputs = model(*args)
    assert not torch.allclose(new_outputs, original_outputs)
    # check that the parameters are different
    for x, y in zip(model.parameters(), original_params, strict=True):
        assert not torch.allclose(x, y)


def test_register_parametrization() -> None:
    dim_in, dim_out = 3, 5
    model = nn.Linear(in_features=dim_in, out_features=dim_out, bias=False)
    register_parametrization(model, "weight", UpperTriangular())
    assert is_upper_triangular(model.weight)

    ps = get_parametrizations(model)
    assert is_parametrization(ps["weight"])


def test_optimization() -> None:
    torch.manual_seed(42)
    batch_size, dim_in, dim_out = 3, 5, 4
    x = torch.randn(batch_size, dim_in)
    y = torch.randn(batch_size, dim_out)

    model = nn.Linear(in_features=dim_in, out_features=dim_out, bias=False)
    register_parametrization(model, "weight", UpperTriangular())

    check_optimization(model, args=(x,), target=y)


def test_optimization_compile() -> None:
    r"""Tests the optimization of a JIT-compiled model."""
    torch.manual_seed(42)
    batch_size, dim_in, dim_out = 3, 5, 4
    x = torch.randn(batch_size, dim_in)
    y = torch.randn(batch_size, dim_out)

    model = nn.Linear(in_features=dim_in, out_features=dim_out, bias=False)
    register_parametrization(model, "weight", UpperTriangular())

    compiled_model = torch.compile(model)
    assert isinstance(compiled_model, OptimizedModule)
    check_optimization(compiled_model, args=(x,), target=y)
    assert is_upper_triangular(compiled_model.weight)


@pytest.mark.xfail(reason="export not supported yet", strict=True)
def test_optimization_export() -> None:
    r"""Tests the optimization of a JIT-compiled model."""
    torch.manual_seed(42)
    batch_size, dim_in, dim_out = 3, 5, 4
    x = torch.randn(batch_size, dim_in)
    y = torch.randn(batch_size, dim_out)

    model = nn.Linear(in_features=dim_in, out_features=dim_out, bias=False)
    register_parametrization(model, "weight", UpperTriangular())

    exported_model = torch.export.export(model, args=(x,)).module()
    check_optimization(exported_model, args=(x,), target=y)
    assert isinstance(exported_model.weight, Tensor)
    assert is_upper_triangular(exported_model.weight)


def test_optimization_missing() -> None:
    r"""Checks that if parametrization is not updated, loss does not change."""
    torch.manual_seed(42)
    batch_size, dim_in, dim_out = 3, 5, 4
    x = torch.randn(batch_size, dim_in)
    y = torch.randn(batch_size, dim_out)

    model = nn.Linear(in_features=dim_in, out_features=dim_out, bias=False)
    register_parametrization(model, "weight", UpperTriangular())

    with torch.no_grad():
        optimizer = SGD(model.parameters(), lr=0.1)
        original_loss = mse_loss(model(x), y)
        loss = original_loss

    for _ in range(5):
        model.zero_grad(set_to_none=True)
        loss = mse_loss(model(x), y)
        loss.backward()
        assert loss.isfinite()
        assert model.weight.isnan().all()  # cache poisoned after backward
        optimizer.step()
        update_parametrizations(model)
        assert is_upper_triangular(model.weight)  # cache restored

    assert loss < original_loss


def test_update_parametrization() -> None:
    torch.manual_seed(42)
    batch_size, dim_in, dim_out = 3, 5, 4
    x = torch.randn(batch_size, dim_in)
    y = torch.randn(batch_size, dim_out)

    model = nn.Linear(in_features=dim_in, out_features=dim_out, bias=False)
    register_parametrization(model, "weight", UpperTriangular())

    with torch.no_grad():
        optimizer = SGD(model.parameters(), lr=0.1)
        original_loss = mse_loss(model(x), y)
        loss = original_loss

    for _ in range(5):
        model.zero_grad(set_to_none=True)
        loss = mse_loss(model(x), y)
        loss.backward()
        assert loss.isfinite()
        optimizer.step()
        update_parametrizations(model)
        assert is_upper_triangular(model.weight)

    assert loss < original_loss


def test_optimizer_hook() -> None:
    torch.manual_seed(42)
    batch_size, dim_in, dim_out = 3, 5, 4
    x = torch.randn(batch_size, dim_in)
    y = torch.randn(batch_size, dim_out)

    model = nn.Linear(in_features=dim_in, out_features=dim_out, bias=False)
    register_parametrization(model, "weight", UpperTriangular())

    with torch.no_grad():
        optimizer = SGD(model.parameters(), lr=0.1)
        register_optimizer_hook(optimizer, model)
        original_loss = mse_loss(model(x), y)
        loss = original_loss

    for _ in range(5):
        model.zero_grad(set_to_none=True)
        loss = mse_loss(model(x), y)
        loss.backward()
        assert loss.isfinite()
        optimizer.step()
        assert is_upper_triangular(model.weight)

    assert loss < original_loss


def test_optimization_cached() -> None:
    r"""Tests the `cached` context manager."""
    torch.manual_seed(42)
    batch_size, dim_in, dim_out = 3, 5, 4
    x = torch.randn(batch_size, dim_in)
    y = torch.randn(batch_size, dim_out)

    model = nn.Linear(in_features=dim_in, out_features=dim_out, bias=False)
    register_parametrization(model, "weight", UpperTriangular())

    with torch.no_grad():
        optimizer = SGD(model.parameters(), lr=0.1)
        register_optimizer_hook(optimizer, model)
        original_loss = mse_loss(model(x), y)
        loss = original_loss

    for _ in range(5):
        with cached(model):
            model.zero_grad(set_to_none=True)
            loss = mse_loss(model(x), y)
            loss.backward()
            assert loss.isfinite()
            optimizer.step()
            assert is_upper_triangular(model.weight)

    assert loss < original_loss


def test_surgery() -> None:
    # create model, parametrization and inputs
    m, n = 3, 3
    inputs = torch.randn(2, 3)
    model = nn.Linear(m, n)
    spec = parametrized(model.weight, LipschitzBounded(0.95))
    # cloned_model = deepcopy(model)

    # register the parametrization
    model.register_module("spec", spec)
    # remove the weight attribute (it still exists on the parametrization)
    del model.weight

    # register the parametrization's weight-buffer as a buffer
    model.register_buffer("weight", spec.cached_parameter.clone().detach())
    assert not model.weight.requires_grad
    # copy the parametrized weight to the buffer.
    model.weight.copy_(spec.original_parameter)
    assert model.weight.requires_grad

    # register the parametrization's weight as a parameter (optional)
    model.parametrized_weight = spec.original_parameter  # type: ignore[unreachable]

    # perform forward and backward pass
    r = model(inputs)
    r.norm().backward()
    assert model.parametrized_weight.grad is not None
    assert model.weight.grad is None


def test_surgery_extended() -> None:
    # create model, parametrization and inputs
    batch_size, dim_in, dim_out = 4, 2, 3

    torch.randn(batch_size, dim_in)
    model = nn.Linear(dim_in, dim_out)

    # plant specific weights
    weight = torch.arange(dim_in * dim_out).reshape(dim_out, dim_in).float()

    with torch.no_grad():
        model.weight.copy_(weight)
        assert matrix_norm(model.weight, ord=2) > 1

    spec = parametrized(model.weight, LipschitzBounded(0.95))
    spec.update_parametrization()
    assert matrix_norm(spec.cached_parameter, ord=2) <= 1.0
    spec.original_parameter.norm().backward()
    spec.zero_grad(set_to_none=True)


def test_parametrized() -> None:
    torch.manual_seed(42)
    batch_size, dim_in, dim_out = 4, 3, 3

    x = torch.randn(batch_size, dim_in)

    # setup reference model
    reference_model = nn.Linear(dim_in, dim_out, bias=False)
    symmetric = Symmetric()
    symmetrized_weight = symmetric(reference_model.weight)
    reference_model.weight = nn.Parameter(symmetrized_weight)
    assert is_symmetric(reference_model.weight)

    # setup vanilla model
    model = nn.Linear(dim_in, dim_out, bias=False)
    with torch.no_grad():
        model.weight.copy_(reference_model.weight)

    # check compatibility
    assert_model_ok(
        model,
        call_args=(x,),
        call_kwargs={},
        reference_model=reference_model,
        test_jit=False,
    )

    # now, parametrizations
    weight = model.weight
    param = parametrized(weight, symmetric)
    param.zero_grad(set_to_none=True)
    model.weight = param.original_parameter
    model.param = param

    # check compatibility
    assert_model_ok(
        model, call_args=(x,), reference_model=reference_model, test_jit=False
    )
