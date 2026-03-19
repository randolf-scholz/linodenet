from copy import deepcopy

import pytest
import torch
from torch import Tensor, nn
from torch.nn.functional import mse_loss
from torch.optim import SGD

from linodenet.nn.parametrize import (
    WrappedParametrization,
    get_parametrizations,
    is_parametrization,
    parametrized,
    register_parametrization,
    update_parametrizations,
)
from linodenet.testing import (
    all_close,
    assert_model_ok,
    check_jit_scriptable,
    check_jit_serializable,
    is_symmetric,
    is_upper_triangular,
)
from linodenet.testing.utils import get_norm


class Symmetric(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return (x + x.transpose(-1, -2)) / 2

    def right_inverse(self, y: Tensor) -> Tensor:
        return y


class UpperTriangular(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.triu()

    def right_inverse(self, y: Tensor) -> Tensor:
        return y


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


def test_jit_trainable_manual() -> None:
    torch.manual_seed(42)
    batch_size, dim_in, dim_out = 3, 5, 4

    x = torch.randn(batch_size, dim_in)
    y = torch.randn(batch_size, dim_out)
    model = nn.Linear(in_features=dim_in, out_features=dim_out, bias=False)
    original_weight = model.weight

    check_jit_scriptable(model)

    # create the parametrization
    param = parametrized(model.weight, UpperTriangular())
    assert param.original_parameter is model.weight
    assert param.cached_parameter is not model.weight

    # check that the parametrization is initialized
    assert not is_upper_triangular(model.weight)
    assert not is_upper_triangular(param.original_parameter)
    assert is_upper_triangular(param.cached_parameter)

    # remove weight and re-register it as a buffer
    delattr(model, "weight")
    model.register_buffer("weight", param.cached_parameter)
    model.register_module("param", param)
    model.register_parameter("original_weight", param.original_parameter)
    assert isinstance(model.param, WrappedParametrization)

    # check that the parametrization is registered
    assert model.weight is not original_weight
    assert model.weight is model.param.cached_parameter
    assert original_weight is model.param.original_parameter

    # check that model can be scripted
    check_jit_scriptable(model)

    # initialize the parametrization
    model.param.update_parametrization()

    # verify that the parametrization is initialized
    assert is_upper_triangular(model.weight)
    assert isinstance(model.param.original_parameter, Tensor)
    assert is_upper_triangular(model.param.original_parameter)
    assert isinstance(model.param.cached_parameter, Tensor)
    assert is_upper_triangular(model.param.cached_parameter)

    # test training plain model
    check_optimization(model, args=(x,), target=y)

    # test training scripted model
    scripted = check_jit_scriptable(model)
    check_optimization(scripted, args=(x,), target=y)

    # test training serialized model
    loaded = check_jit_serializable(scripted)
    check_optimization(loaded, args=(x,), target=y)


def test_jit_trainable() -> None:
    r"""Tests the optimization of a JIT-compiled model."""
    torch.manual_seed(42)
    batch_size, dim_in, dim_out = 3, 5, 4
    x = torch.randn(batch_size, dim_in)
    y = torch.randn(batch_size, dim_out)

    model = nn.Linear(in_features=dim_in, out_features=dim_out, bias=False)
    register_parametrization(model, "weight", UpperTriangular())

    scripted_model = check_jit_scriptable(model)
    assert is_upper_triangular(scripted_model.weight)
    check_optimization(scripted_model, args=(x,), target=y)

    deserialized_model = check_jit_serializable(model)
    assert is_upper_triangular(deserialized_model.weight)
    check_optimization(deserialized_model, args=(x,), target=y)


def test_jit_preserves_parameters() -> None:
    dim_in, dim_out = 2, 3
    model = nn.Linear(in_features=dim_in, out_features=dim_out, bias=False)

    original_parameters = deepcopy(tuple(model.parameters()))

    deserialized_model = check_jit_serializable(model)
    deserialized_param = deepcopy(tuple(deserialized_model.parameters()))

    # apply only a dummy parametrization for this test.
    register_parametrization(model, "weight", nn.Identity())
    parametrized_parameters = deepcopy(tuple(model.parameters()))

    deserialized_parametrized_model = check_jit_serializable(deserialized_model)
    deserialized_parametrized_param = deepcopy(
        tuple(deserialized_parametrized_model.parameters())
    )

    for x, y in zip(original_parameters, deserialized_param, strict=True):
        assert x.shape == y.shape
        assert torch.equal(x, y)

    for x, y in zip(original_parameters, parametrized_parameters, strict=True):
        assert x.shape == y.shape
        assert torch.equal(x, y)

    for x, y in zip(original_parameters, deserialized_parametrized_param, strict=True):
        assert x.shape == y.shape
        assert torch.equal(x, y)


@pytest.mark.xfail(
    reason="After deserialization update_parametrization must be called."
)
def test_jit() -> None:
    r"""Test that subclasses of Protocol-class work with JIT."""
    torch.manual_seed(42)

    batch_size, dim_in, dim_out = 7, 5, 4
    inputs = torch.randn(batch_size, dim_in)
    model = nn.Linear(in_features=dim_in, out_features=dim_out, bias=False)

    # check_object(model, input_args=(inputs,))

    register_parametrization(model, "weight", UpperTriangular())

    assert_model_ok(model, call_args=(inputs,))


def test_jit_attribute() -> None:
    dim_in, dim_out = 3, 5
    model = nn.Linear(in_features=dim_in, out_features=dim_out, bias=False)
    assert not is_upper_triangular(model.weight)

    register_parametrization(model, "weight", UpperTriangular())
    ps = get_parametrizations(model)
    assert is_upper_triangular(model.weight)
    assert is_parametrization(ps["weight"])

    # check that model can be scripted
    scripted = check_jit_scriptable(model)
    scripted_ps = get_parametrizations(model)
    assert is_upper_triangular(scripted.weight)
    assert is_parametrization(scripted_ps["weight"])

    # check that model can be scripted
    deserialized_model = check_jit_serializable(model)
    assert is_upper_triangular(deserialized_model.weight)
    # assert is_parametrization(deserialized_model["weight"])


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
        test_jit=True,
    )

    # now, parametrizations
    weight = model.weight
    param = parametrized(weight, symmetric)
    param.zero_grad(set_to_none=True)
    model.weight = param.original_parameter
    model.param = param

    # check compatibility
    assert_model_ok(
        model, call_args=(x,), reference_model=reference_model, test_jit=True
    )
