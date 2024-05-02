r"""Test JIT-compatibility of `linodenet.initializations."""

import pytest
import torch

from linodenet.initializations import INITIALIZATIONS


@pytest.mark.xfail(reason="Uses scipy functions.")
@pytest.mark.parametrize("initialization_name", INITIALIZATIONS)
def test_jit_compatibility_functional(initialization_name: str) -> None:
    r"""Test JIT-compatibility of functional projections."""
    shape = (4, 4)

    initialization = INITIALIZATIONS[initialization_name]
    scripted_initialization = torch.jit.script(initialization)

    try:
        result_prior = initialization(shape)
    except NotImplementedError:
        pytest.skip(f"{initialization_name} is not implemented.")

    result_post = scripted_initialization(shape)
    assert torch.allclose(result_prior, result_post)
