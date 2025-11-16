r"""Check that spectral norm works as a parametrization."""

import torch
from torch import nn

from linodenet.parametrize import SpectralNormalization, register_parametrization
from linodenet.testing import is_contraction


def test_trainable() -> None:
    m = n = 4
    model = nn.Linear(m, n)
    with torch.no_grad():
        model.weight.copy_(4 * torch.eye(m) + torch.randn(m, n))
    assert not is_contraction(model.weight)
    register_parametrization(model, "weight", SpectralNormalization)
    assert is_contraction(model.weight)
