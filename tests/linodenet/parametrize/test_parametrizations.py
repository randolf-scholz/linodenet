r"""Test parametrization of modules."""

import pytest
import torch
from torch import nn

from linodenet.parametrize import PARAMETRIZATIONS, Parametrization


@pytest.mark.parametrize("name", PARAMETRIZATIONS)
def test_parametrization(name: str) -> None:
    r"""Test parametrization."""
    cls = PARAMETRIZATIONS[name]
    tensor = nn.Parameter(torch.randn(3, 3))
    parametrization = cls(tensor)
    assert isinstance(parametrization, Parametrization)
