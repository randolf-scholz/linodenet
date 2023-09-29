#!/usr/bin/env python
"""Test parametrization of modules."""

import logging

import torch
from pytest import mark
from torch import Tensor, nn
from torch.linalg import matrix_norm

from linodenet.parametrize import SimpleParametrization, SpectralNormalization
from linodenet.projections import is_symmetric, symmetric
from linodenet.testing import check_model

logging.basicConfig(level=logging.INFO)


def test_overwrite_module_attribute():
    class parametrize(nn.Module):
        weight: Tensor

        def __init__(self, weight: Tensor) -> None:
            super().__init__()
            self.weight = weight

    model = nn.Linear(4, 3)
    inputs = torch.randn(2, 4)

    model(inputs)


@mark.skip(reason="Not implemented")
def test_jit_protocol() -> None:
    """Test that subclasses of Protocol-class work with JIT."""
    raise NotImplementedError


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
    model.register_buffer("weight", model.spec.weight.clone().detach())
    assert not model.weight.requires_grad
    # copy the parametrized weight to the buffer.
    model.weight.copy_(model.spec.parametrized_tensors["weight"])
    assert model.weight.requires_grad

    # register the parametrization's weight as a parameter (optional)
    model.parametrized_weight = model.spec.parametrized_tensors["weight"]

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
    spec.reset_cache()
    assert matrix_norm(spec.weight, ord=2) <= 1.0
    spec.weight.norm().backward()
    spec.zero_grad(set_to_none=True)


def test_parametrization() -> None:
    model = nn.Linear(4, 4)

    spec = SpectralNormalization(nn.Parameter(model.weight.clone().detach()))

    model.spec = spec
    spec._update_cached_tensors()

    inputs = torch.randn(2, 4)

    check_model(model, input_args=inputs, test_jit=True)


def test_dummy():
    model = nn.Linear(4, 4)

    spec = SpectralNormalization(nn.Parameter(model.weight.clone().detach()))

    model.spec = spec
    spec._update_cached_tensors()
    model.register_buffer("weight", model.spec.weight)
    model.register_parameter(
        "parametrized_weight", model.spec.parametrized_tensors["weight"]
    )
    inputs = torch.randn(2, 4)

    check_model(model, input_args=inputs, test_jit=True)


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
    param = SimpleParametrization(weight, symmetric)
    param.zero_grad(set_to_none=True)
    model.weight = param.parametrized_tensor
    model.param = param

    # check compatibility
    check_model(model, input_args=(x,), reference_model=reference_model, test_jit=True)


if __name__ == "__main__":
    pass
