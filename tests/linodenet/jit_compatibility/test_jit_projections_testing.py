"""Test JIT-compatibility of `linodenet.projections.testing`."""

import pytest
import torch

from linodenet.projections import (
    FUNCTIONAL_PROJECTIONS,
    MATRIX_TESTS,
    MODULAR_PROJECTIONS,
)


@pytest.mark.parametrize("projection_name", FUNCTIONAL_PROJECTIONS)
def test_jit_compatibility_functional(projection_name: str):
    r"""Test JIT-compatibility of functional projections."""
    x = torch.randn(4, 4)
    projection = FUNCTIONAL_PROJECTIONS[projection_name]
    scripted_projection = torch.jit.script(projection)

    try:
        result_prior = projection(x)
    except NotImplementedError:
        pytest.skip(f"{projection_name} is not implemented.")

    result_post = scripted_projection(x)
    assert torch.allclose(result_prior, result_post)


@pytest.mark.parametrize("projection_name", MODULAR_PROJECTIONS)
def test_jit_compatibility_modular(projection_name: str):
    r"""Test JIT-compatibility of modular projections."""
    x = torch.randn(4, 4)
    projection_type = MODULAR_PROJECTIONS[projection_name]
    projection = projection_type()
    scripted_projection = torch.jit.script(projection)

    try:
        result_prior = projection(x)
    except NotImplementedError:
        pytest.skip(f"{projection_name} is not implemented.")

    result_post = scripted_projection(x)
    assert torch.allclose(result_prior, result_post)


@pytest.mark.parametrize("test_name", MATRIX_TESTS)
def test_jit_compatibility(test_name: str):
    r"""Test JIT-compatibility of matrix tests."""
    x = torch.randn(4, 4)
    matrix_test = MATRIX_TESTS[test_name]
    scripted_test = torch.jit.script(matrix_test)

    try:
        result_prior = matrix_test(x)
    except NotImplementedError:
        pytest.skip(f"{test_name} is not implemented.")

    result_post = scripted_test(x)
    assert result_prior == result_post


@pytest.mark.skip(reason="Slow.")
@pytest.mark.parametrize("projection_name", FUNCTIONAL_PROJECTIONS)
def test_compile_compatibility_functional(projection_name: str):
    r"""Test JIT-compatibility of functional projections."""
    x = torch.randn(4, 4)
    projection = FUNCTIONAL_PROJECTIONS[projection_name]
    scripted_projection = torch.compile(projection)

    try:
        result_prior = projection(x)
    except NotImplementedError:
        pytest.skip(f"{projection_name} is not implemented.")

    result_post = scripted_projection(x)
    assert torch.allclose(result_prior, result_post)


@pytest.mark.skip(reason="Slow.")
@pytest.mark.parametrize("projection_name", MODULAR_PROJECTIONS)
def test_compile_compatibility_modular(projection_name: str):
    r"""Test JIT-compatibility of modular projections."""
    x = torch.randn(4, 4)
    projection_type = MODULAR_PROJECTIONS[projection_name]
    projection = projection_type()
    scripted_projection = torch.compile(projection)

    try:
        result_prior = projection(x)
    except NotImplementedError:
        pytest.skip(f"{projection_name} is not implemented.")

    result_post = scripted_projection(x)
    assert torch.allclose(result_prior, result_post)


@pytest.mark.skip(reason="Slow.")
@pytest.mark.parametrize("test_name", MATRIX_TESTS)
def test_compile_compatibility(test_name: str):
    r"""Test JIT-compatibility of matrix tests."""
    x = torch.randn(4, 4)
    matrix_test = MATRIX_TESTS[test_name]
    scripted_test = torch.compile(matrix_test)

    try:
        result_prior = matrix_test(x)
    except NotImplementedError:
        pytest.skip(f"{test_name} is not implemented.")

    result_post = scripted_test(x)
    assert result_prior == result_post
