#!/usr/bin/env python
"""Test parametrization of modules."""

import logging

import torch
from pytest import mark
from torch import nn
from torch.linalg import matrix_norm
from torch.nn.functional import mse_loss
from torch.optim import SGD

from linodenet.parametrize import SpectralNormalization, UpperTriangular, parametrize
from linodenet.projections import is_symmetric, is_upper_triangular, symmetric
from linodenet.testing import (
    check_jit,
    check_jit_scripting,
    check_jit_serialization,
    check_model,
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


@mark.skip(reason="Not implemented")
def test_jit_protocol() -> None:
    """Test that subclasses of Protocol-class work with JIT."""
    raise NotImplementedError


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

    # register the parametrization
    model.register_module("param", param)

    # remove weight and re-register it as a buffer
    del model.weight
    model.register_buffer("weight", model.param.cached_parameter)
    model.register_parameter("original_weight", model.param.original_parameter)

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
    with torch.no_grad():
        optimizer = SGD(model.parameters(), lr=0.1)
        original_loss = mse_loss(model(inputs), targets)

    for _ in range(5):
        model.zero_grad(set_to_none=True)
        loss = mse_loss(model(inputs), targets)
        loss.backward()
        assert loss.isfinite()
        print(loss)
        optimizer.step()
        model.param.update_parametrization()
        assert is_upper_triangular(model.weight)

    assert loss < original_loss
    # end region test training ---------------------------------------------------------

    # now, try with scripted model
    # region test training -------------------------------------------------------------
    with torch.no_grad():
        scripted = check_jit_scripting(model)
        del model
        optimizer = SGD(scripted.parameters(), lr=0.1)
        original_loss = mse_loss(scripted(inputs), targets)

    for _ in range(5):
        scripted.zero_grad(set_to_none=True)
        loss = mse_loss(scripted(inputs), targets)
        loss.backward()
        assert loss.isfinite()
        print(loss)
        optimizer.step()
        scripted.param.update_parametrization()
        assert is_upper_triangular(scripted.weight)

    assert loss < original_loss
    # end region test training ---------------------------------------------------------

    # now, try with serialized model
    # region test training -------------------------------------------------------------
    with torch.no_grad():
        loaded = check_jit_serialization(scripted)
        del scripted
        optimizer = SGD(loaded.parameters(), lr=0.1)
        original_loss = mse_loss(loaded(inputs), targets)

    for _ in range(5):
        loaded.zero_grad(set_to_none=True)
        loss = mse_loss(loaded(inputs), targets)
        loss.backward()
        assert loss.isfinite()
        print(loss)
        optimizer.step()
        loaded.param.update_parametrization()
        assert is_upper_triangular(loaded.weight)

    assert loss < original_loss
    # end region test training ---------------------------------------------------------


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


def test_parametrization() -> None:
    model = nn.Linear(4, 4)

    spec = SpectralNormalization(nn.Parameter(model.weight.clone().detach()))

    model.spec = spec
    spec.update_cache()

    inputs = torch.randn(2, 4)

    check_model(model, input_args=(inputs,), test_jit=True)


def test_dummy():
    model = nn.Linear(4, 4)

    spec = SpectralNormalization(nn.Parameter(model.weight.clone().detach()))
    del model.weight
    model.spec = spec
    spec.update_cache()

    model.register_buffer("weight", model.spec.weight)
    model.register_parameter(
        "parametrized_weight", model.spec.parametrized_tensors["weight"]
    )
    inputs = torch.randn(2, 4)

    check_model(model, input_args=(inputs,), test_jit=True)


@mark.skip(reason="Not implemented")
def test_parametrizations() -> None: ...


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
