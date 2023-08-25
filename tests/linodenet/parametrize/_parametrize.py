#!/usr/bin/env python


import pytest
import torch
from torch import Tensor, jit, nn
from torch.nn.utils import parametrize as torch_parametrize

from linodenet.testing import test_model


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


if __name__ == "__main__":
    pass
