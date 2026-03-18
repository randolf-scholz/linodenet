r"""Test parametrization of modules."""

import pytest
import torch
from torch import nn

from linodenet.parametrizations import (
    STATIC_PARAMETRIZATIONS,
    is_parametrization,
    parametrized,
)


@pytest.mark.parametrize("name", STATIC_PARAMETRIZATIONS)
def test_parametrization(name: str) -> None:
    r"""Test parametrization."""
    obj = STATIC_PARAMETRIZATIONS[name]
    tensor = nn.Parameter(torch.randn(3, 3))

    try:
        parametrization = parametrized(tensor, obj)
    except NotImplementedError:
        pytest.xfail(f"{name} parametrization not implemented")

    assert is_parametrization(parametrization)
    assert hasattr(parametrization, "DOMAIN")
    assert hasattr(parametrization, "CODOMAIN")
