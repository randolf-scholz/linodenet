#!/usr/bin/env python
"""Test parametrization of modules."""

import logging
from copy import deepcopy

import torch
from pytest import mark
from torch import Tensor, nn
from torch.linalg import matrix_norm
from torch.nn.functional import mse_loss
from torch.optim import SGD

from linodenet.parametrize import (
    Identity,
    Parametrization,
    SpectralNormalization,
    UpperTriangular,
    cached,
    parametrize,
    register_optimizer_hook,
    register_parametrization,
    update_parametrizations,
)
from linodenet.projections import is_symmetric, is_upper_triangular, symmetric
from linodenet.testing import (
    check_jit,
    check_jit_scripting,
    check_jit_serialization,
    check_model,
    check_object,
)

logging.basicConfig(level=logging.INFO)


# class SlowUpperTriangular(ParametrizationBase):
#     """Parametrize a matrix to be upper triangular."""
#
#     def forward(self, x: Tensor) -> Tensor:
#         # waste some time:
#         z = torch.randn(100, 100)
#         for k in range(100_000):
#             z = z @ torch.randn(100, 100)
#
#         return upper_triangular(x)


def check_optimization(model: nn.Module, inputs: Tensor, targets: Tensor) -> None:
    optimizer = SGD(model.parameters(), lr=0.1)

    with torch.no_grad():
        original_outputs = model(inputs)
        original_loss = mse_loss(original_outputs, targets)
        original_params = [w.clone().detach() for w in model.parameters()]

    update_parametrizations(
        model
    )  # <-- crucial, otherwise no update in first iteration!

    for _ in range(5):
        model.zero_grad(set_to_none=True)
        outputs = model(inputs)
        loss = mse_loss(outputs, targets)
        loss.backward()
        assert all(w.grad is not None for w in model.parameters() if w.requires_grad)
        assert loss.isfinite()
        optimizer.step()
        update_parametrizations(model)
        assert is_upper_triangular(model.weight)

    # check that the loss has decreased
    assert loss < original_loss
    # check that the outputs are different
    assert not torch.allclose(outputs, original_outputs)
    # check that the parameters are different
    for x, y in zip(model.parameters(), original_params):
        assert not torch.allclose(x, y)


def test_jit_preserves_parameters() -> None:
    N, M = 2, 3
    model = nn.Linear(in_features=N, out_features=M, bias=False)

    original_parameters = deepcopy(tuple(model.parameters()))

    deserialized_model = check_jit(model)
    deserialized_param = deepcopy(tuple(deserialized_model.parameters()))

    # apply only a dummy parametrization for this test.
    register_parametrization(model, "weight", Identity)
    parametrized_parameters = deepcopy(tuple(model.parameters()))

    deserialized_parametrized_model = check_jit(deserialized_model)
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


@mark.xfail(reason="After deserialization update_parametrization must be called.")
def test_jit() -> None:
    """Test that subclasses of Protocol-class work with JIT."""
    torch.manual_seed(42)

    B, N, M = 7, 5, 4
    inputs = torch.randn(B, N)
    model = nn.Linear(in_features=N, out_features=M, bias=False)

    # check_object(model, input_args=(inputs,))

    register_parametrization(model, "weight", UpperTriangular)

    check_object(model, input_args=(inputs,))


def test_register_parametrization() -> None:
    N, M = 3, 5
    model = nn.Linear(in_features=N, out_features=M, bias=False)
    register_parametrization(model, "weight", UpperTriangular)
    assert is_upper_triangular(model.weight)
    assert isinstance(model.weight_parametrization, Parametrization)


def test_jit_attribute() -> None:
    N, M = 3, 5
    model = nn.Linear(in_features=N, out_features=M, bias=False)
    deepcopy(model)
    register_parametrization(model, "weight", UpperTriangular)
    model.weight_parametrization.detach_cache()
    deepcopy(model)

    # check that model can be scripted
    loaded = check_jit(model)
    assert is_upper_triangular(loaded.weight)
    assert isinstance(loaded.weight_parametrization, Parametrization)

    # check that model can be scripted
    scripted = check_jit_scripting(model)
    assert is_upper_triangular(scripted.weight)
    assert isinstance(scripted.weight_parametrization, Parametrization)

    # check that model can be serialized
    loaded = check_jit_serialization(scripted)
    assert is_upper_triangular(loaded.weight)
    assert isinstance(loaded.weight_parametrization, Parametrization)


def test_optimization_manual() -> None:
    torch.manual_seed(42)

    B, N, M = 3, 5, 4
    inputs = torch.randn(B, N)
    targets = torch.randn(B, M)
    model = nn.Linear(in_features=N, out_features=M, bias=False)
    original_weight = model.weight

    check_jit(model)

    # create the parametrization
    param = UpperTriangular(model.weight)
    assert param.original_parameter is model.weight
    assert param.cached_parameter is not model.weight

    # check that the parametrization is not initialized
    assert not is_upper_triangular(model.weight)
    assert not is_upper_triangular(param.original_parameter)
    assert not is_upper_triangular(param.cached_parameter)
    assert torch.allclose(param.original_parameter, param.cached_parameter)

    # remove weight and re-register it as a buffer
    delattr(model, "weight")
    model.register_buffer("weight", param.cached_parameter)
    model.register_module("param", param)
    model.register_parameter("original_weight", param.original_parameter)

    # check that the parametrization is registered
    assert model.weight is not original_weight
    assert model.weight is model.param.cached_parameter
    assert original_weight is model.param.original_parameter

    # check that the parametrization is still not initialized
    assert not is_upper_triangular(model.weight), model.weight

    # check that model can be scripted
    check_jit(model)

    # initialize the parametrization
    model.param.update_parametrization()

    # verify that the parametrization is initialized
    assert is_upper_triangular(model.weight)
    assert is_upper_triangular(model.param.original_parameter)
    assert is_upper_triangular(model.param.cached_parameter)

    # region test training -------------------------------------------------------------
    check_optimization(model, inputs, targets)
    # end region test training ---------------------------------------------------------

    # region test training -------------------------------------------------------------
    scripted = check_jit_scripting(model)
    check_optimization(scripted, inputs, targets)
    # end region test training ---------------------------------------------------------

    # region test training -------------------------------------------------------------
    loaded = check_jit_serialization(scripted)
    check_optimization(loaded, inputs, targets)
    # end region test training ---------------------------------------------------------


def test_optimization_missing() -> None:
    """Checks that if parametrization is not updated, loss does not change."""
    torch.manual_seed(42)
    B, N, M = 3, 5, 4
    inputs = torch.randn(B, N)
    targets = torch.randn(B, M)
    model = nn.Linear(in_features=N, out_features=M, bias=False)
    register_parametrization(model, "weight", UpperTriangular)

    with torch.no_grad():
        optimizer = SGD(model.parameters(), lr=0.1)
        original_loss = mse_loss(model(inputs), targets)

    for _ in range(5):
        model.zero_grad(set_to_none=True)
        loss = mse_loss(model(inputs), targets)
        loss.backward()
        assert loss.isfinite()
        optimizer.step()
        assert is_upper_triangular(model.weight)

    assert loss == original_loss


def test_optimization() -> None:
    torch.manual_seed(42)
    B, N, M = 3, 5, 4
    inputs = torch.randn(B, N)
    targets = torch.randn(B, M)
    model = nn.Linear(in_features=N, out_features=M, bias=False)
    register_parametrization(model, "weight", UpperTriangular)

    # check_optimization(model, inputs, targets)

    # scripted = check_jit_scripting(model)
    # check_optimization(scripted, inputs, targets)

    loaded = check_jit(model)
    check_optimization(loaded, inputs, targets)


def test_update_parametrization() -> None:
    torch.manual_seed(42)
    B, N, M = 3, 5, 4
    inputs = torch.randn(B, N)
    targets = torch.randn(B, M)
    model = nn.Linear(in_features=N, out_features=M, bias=False)
    register_parametrization(model, "weight", UpperTriangular)

    with torch.no_grad():
        optimizer = SGD(model.parameters(), lr=0.1)
        original_loss = mse_loss(model(inputs), targets)

    for _ in range(5):
        model.zero_grad(set_to_none=True)
        loss = mse_loss(model(inputs), targets)
        loss.backward()
        assert loss.isfinite()
        optimizer.step()
        update_parametrizations(model)
        assert is_upper_triangular(model.weight)

    assert loss < original_loss


def test_optimizer_hook() -> None:
    torch.manual_seed(42)
    B, N, M = 3, 5, 4
    inputs = torch.randn(B, N)
    targets = torch.randn(B, M)
    model = nn.Linear(in_features=N, out_features=M, bias=False)
    register_parametrization(model, "weight", UpperTriangular)

    with torch.no_grad():
        optimizer = SGD(model.parameters(), lr=0.1)
        register_optimizer_hook(optimizer, model)
        original_loss = mse_loss(model(inputs), targets)

    for _ in range(5):
        model.zero_grad(set_to_none=True)
        loss = mse_loss(model(inputs), targets)
        loss.backward()
        assert loss.isfinite()
        optimizer.step()
        assert is_upper_triangular(model.weight)

    assert loss < original_loss


def test_optimization_cached() -> None:
    """Tests the `cached` context manager."""
    torch.manual_seed(42)
    B, N, M = 3, 5, 4
    inputs = torch.randn(B, N)
    targets = torch.randn(B, M)
    model = nn.Linear(in_features=N, out_features=M, bias=False)
    register_parametrization(model, "weight", UpperTriangular)

    with torch.no_grad():
        optimizer = SGD(model.parameters(), lr=0.1)
        register_optimizer_hook(optimizer, model)
        original_loss = mse_loss(model(inputs), targets)

    for _ in range(5):
        with cached(model):
            model.zero_grad(set_to_none=True)
            loss = mse_loss(model(inputs), targets)
            loss.backward()
            assert loss.isfinite()
            optimizer.step()
            assert is_upper_triangular(model.weight)

    assert loss < original_loss


def test_surgery() -> None:
    # create model, parametrization and inputs
    inputs = torch.randn(2, 3)
    model = nn.Linear(3, 3)
    spec = SpectralNormalization(model.weight)
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
    torch.randn(2, 3)
    model = nn.Linear(3, 3)

    # plant specific weights
    weight = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    with torch.no_grad():
        model.weight.copy_(weight)
        assert matrix_norm(model.weight, ord=2) > 1

    spec = SpectralNormalization(model.weight)
    spec.update_parametrization()
    assert matrix_norm(spec.cached_parameter, ord=2) <= 1.0
    spec.original_parameter.norm().backward()
    spec.zero_grad(set_to_none=True)


def test_param() -> None:
    B, M, N = 4, 3, 3
    x = torch.randn(B, M)

    # setup reference model
    reference_model = nn.Linear(M, N, bias=False)
    symmetrized_weight = symmetric(reference_model.weight)
    reference_model.weight = nn.Parameter(symmetrized_weight)
    assert is_symmetric(reference_model.weight)

    # setup vanilla model
    model = nn.Linear(M, N, bias=False)
    with torch.no_grad():
        model.weight.copy_(reference_model.weight)

    # check compatibility
    check_model(model, input_args=(x,), reference_model=reference_model, test_jit=True)

    # now, parametrize
    weight = model.weight
    param = parametrize(weight, symmetric)
    param.zero_grad(set_to_none=True)
    model.weight = param.original_parameter
    model.param = param

    # check compatibility
    check_model(model, input_args=(x,), reference_model=reference_model, test_jit=True)


if __name__ == "__main__":
    pass
