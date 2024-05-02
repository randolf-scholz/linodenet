r"""Test JIT-compatibility of `linodenet.projections.testing`."""

import pytest
import torch

from linodenet.regularizations import (
    FUNCTIONAL_REGULARIZATIONS,
    MODULAR_REGULARIZATIONS,
)


@pytest.mark.parametrize("regularization_name", FUNCTIONAL_REGULARIZATIONS)
def test_jit_compatibility_functional(regularization_name):
    r"""Test JIT-compatibility of functional projections."""
    x = torch.randn(4, 4)
    projection = FUNCTIONAL_REGULARIZATIONS[regularization_name]
    scripted_projection = torch.jit.script(projection)

    try:
        result_prior = projection(x)
    except NotImplementedError:
        pytest.skip(f"{regularization_name} is not implemented.")

    result_post = scripted_projection(x)
    assert torch.allclose(result_prior, result_post)


@pytest.mark.parametrize("regularization_name", MODULAR_REGULARIZATIONS)
def test_jit_compatibility_modular(regularization_name):
    r"""Test JIT-compatibility of modular projections."""
    x = torch.randn(4, 4)
    projection_type = MODULAR_REGULARIZATIONS[regularization_name]
    projection = projection_type()
    scripted_projection = torch.jit.script(projection)

    try:
        result_prior = projection(x)
    except NotImplementedError:
        pytest.skip(f"{regularization_name} is not implemented.")

    result_post = scripted_projection(x)
    assert torch.allclose(result_prior, result_post)


@pytest.mark.skip(reason="Slow.")
@pytest.mark.parametrize("regularization_name", FUNCTIONAL_REGULARIZATIONS)
def test_compile_compatibility_functional(regularization_name):
    r"""Test JIT-compatibility of functional projections."""
    x = torch.randn(4, 4)
    projection = FUNCTIONAL_REGULARIZATIONS[regularization_name]
    scripted_projection = torch.compile(projection)

    try:
        result_prior = projection(x)
    except NotImplementedError:
        pytest.skip(f"{regularization_name} is not implemented.")

    result_post = scripted_projection(x)
    assert torch.allclose(result_prior, result_post)


@pytest.mark.skip(reason="Slow.")
@pytest.mark.parametrize("regularization_name", MODULAR_REGULARIZATIONS)
def test_compile_compatibility_modular(regularization_name):
    r"""Test JIT-compatibility of modular projections."""
    x = torch.randn(4, 4)
    projection_type = MODULAR_REGULARIZATIONS[regularization_name]
    projection = projection_type()
    scripted_projection = torch.compile(projection)

    try:
        result_prior = projection(x)
    except NotImplementedError:
        pytest.skip(f"{regularization_name} is not implemented.")

    result_post = scripted_projection(x)
    assert torch.allclose(result_prior, result_post)
