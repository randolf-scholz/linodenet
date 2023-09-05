#!/usr/bin/env python


import logging

import torch
from torch import Tensor, nn

from linodenet.parametrize import Parametrize, SpectralNormalization
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

    reference_outputs = model(inputs)


def test_jit_protocol() -> None:
    """...."""


def test_parametrization() -> None:
    model = nn.Linear(4, 4)

    spec = SpectralNormalization(nn.Parameter(model.weight.clone().detach()))

    model.spec = spec
    spec.recompute_cache()

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
    param = Parametrize(weight, symmetric)
    param.zero_grad(set_to_none=True)
    model.weight = param.parametrized_tensor
    model.param = param

    # check compatibility
    check_model(model, input_args=(x,), reference_model=reference_model, test_jit=True)


if __name__ == "__main__":
    pass
