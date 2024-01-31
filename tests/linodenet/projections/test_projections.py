"""Tests for linodenet.projections."""

import pytest
import torch

from linodenet.projections import (
    FUNCTIONAL_PROJECTIONS,
    MODULAR_PROJECTIONS,
)
from linodenet.testing import MATRIX_TESTS
from test_utils import camel2snake, snake2camel


@pytest.mark.parametrize("projection_name", FUNCTIONAL_PROJECTIONS)
def test_names_functional(projection_name: str):
    r"""Test that all projections have the correct name."""
    assert projection_name == FUNCTIONAL_PROJECTIONS[projection_name].__name__


@pytest.mark.parametrize("projection_name", MODULAR_PROJECTIONS)
def test_names_modular(projection_name: str):
    r"""Test that all modular projections have the correct name."""
    assert projection_name == MODULAR_PROJECTIONS[projection_name].__name__


@pytest.mark.parametrize("test_name", MATRIX_TESTS)
def test_names_matrix_tests(test_name: str):
    r"""Test that all matrix tests have the correct name."""
    assert test_name == MATRIX_TESTS[test_name].__name__


@pytest.mark.parametrize("projection_name", FUNCTIONAL_PROJECTIONS)
def test_inclusion_functional_has_test(projection_name: str):
    r"""Test that all projections have tests."""
    if projection_name != "identity":
        assert f"is_{projection_name}" in MATRIX_TESTS


@pytest.mark.parametrize("projection_name", MODULAR_PROJECTIONS)
def test_inclusion_modular_has_functional(projection_name: str):
    assert camel2snake(projection_name) in FUNCTIONAL_PROJECTIONS


@pytest.mark.parametrize("projection_name", FUNCTIONAL_PROJECTIONS)
def test_inclusion_functional_has_modular(projection_name: str):
    assert snake2camel(projection_name) in MODULAR_PROJECTIONS


@pytest.mark.parametrize("projection_name", FUNCTIONAL_PROJECTIONS)
def test_projections_work(projection_name: str):
    r"""Test that all projections work."""
    if projection_name == "identity":
        return

    projection = FUNCTIONAL_PROJECTIONS[projection_name]
    matrix_test = MATRIX_TESTS[f"is_{projection_name}"]
    x = torch.randn(4, 4)

    try:
        y = projection(x)
    except NotImplementedError:
        pytest.skip(f"{projection_name} is not implemented.")

    try:
        result = matrix_test(y)
    except NotImplementedError:
        pytest.skip(f"test for {projection_name} is not implemented.")

    assert result is True
