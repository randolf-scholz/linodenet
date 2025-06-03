r"""Test parametrization of modules."""

import pytest
import torch
from torch import nn

from linodenet import parametrize
from linodenet.parametrize import PARAMETRIZATIONS, Parametrization


@pytest.mark.parametrize("name", PARAMETRIZATIONS)
def test_parametrization(name: str) -> None:
    r"""Test parametrization."""
    cls = PARAMETRIZATIONS[name]
    parametrization: Parametrization
    tensor = nn.Parameter(torch.randn(3, 3))

    match cls:
        case parametrize.Masked as Masked:
            # sample random mask
            mask = torch.randint(0, 2, (3, 3), dtype=torch.bool)
            parametrization = Masked(tensor, mask=mask)
        case _:
            parametrization = cls(tensor)

    assert isinstance(parametrization, Parametrization)
