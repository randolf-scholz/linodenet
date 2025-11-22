r"""Test parametrization of modules."""

import pytest
import torch
from torch import nn

from linodenet.parametrize import PARAMETRIZATIONS, is_parametrization


@pytest.mark.parametrize("name", PARAMETRIZATIONS)
def test_parametrization(name: str) -> None:
    r"""Test parametrization."""
    cls = PARAMETRIZATIONS[name]
    tensor = nn.Parameter(torch.randn(3, 3))

    try:
        parametrization = cls(tensor)
    except NotImplementedError:
        pytest.xfail(f"{name} parametrization not implemented")

    assert is_parametrization(parametrization)
    assert hasattr(parametrization, "DOMAIN")
    assert hasattr(parametrization, "CODOMAIN")
