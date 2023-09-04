#!/usr/bin/env python


import logging

import pytest
import torch
from torch import Tensor, jit, nn
from torch.nn.utils import parametrize as torch_parametrize

from linodenet.parametrize import Parametrization, SpectralNormalization
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


def test_builtin_parametrize_fails_jit() -> None:
    """Test builtin parametrize."""

    class Symmetric(nn.Module):
        def forward(self, x: Tensor) -> Tensor:
            return x.triu() + x.triu(1).transpose(-1, -2)

    model = nn.Linear(5, 5)

    torch_parametrize.register_parametrization(model, "weight", Symmetric())

    scripted = jit.script(model)
    inputs = torch.randn(7, 5)

    with pytest.raises(jit.Error):
        scripted(inputs)


def test_jit_protocol() -> None:
    """...."""


def test_parametrization() -> None:
    model = nn.Linear(4, 4)

    spec = SpectralNormalization(nn.Parameter(model.weight.clone().detach()))

    model.spec = spec
    spec.recompute_cache()

    inputs = torch.randn(2, 4)

    check_model(model, input_args=inputs, test_jit=True)


if __name__ == "__main__":
    pass
