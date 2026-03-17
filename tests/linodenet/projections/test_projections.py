r"""Tests for linodenet.projections."""

import pytest
import torch

from linodenet.projections import (
    FUNCTIONAL_PROJECTIONS,
    MODULAR_PROJECTIONS,
    RankOne,
    Tridiagonal,
    banded,
    low_rank,
    rank_one,
    tridiagonal,
)
from linodenet.testing import MATRIX_TESTS
from tests.testing import camel2snake, snake2camel


@pytest.mark.parametrize("projection_name", FUNCTIONAL_PROJECTIONS)
def test_names_functional(projection_name: str) -> None:
    r"""Test that all projections have the correct name."""
    projection = FUNCTIONAL_PROJECTIONS[projection_name]
    actual_name = getattr(projection, "__name__", None)
    assert projection_name == actual_name


@pytest.mark.parametrize("projection_name", MODULAR_PROJECTIONS)
def test_names_modular(projection_name: str) -> None:
    r"""Test that all modular projections have the correct name."""
    projection = MODULAR_PROJECTIONS[projection_name]
    actual_name = getattr(projection, "__name__", None)
    assert projection_name == actual_name


@pytest.mark.parametrize("test_name", MATRIX_TESTS)
def test_names_matrix_tests(test_name: str) -> None:
    r"""Test that all matrix tests have the correct name."""
    matrix_test = MATRIX_TESTS[test_name]
    actual_name = getattr(matrix_test, "__name__", None)
    assert test_name == actual_name


@pytest.mark.parametrize("projection_name", FUNCTIONAL_PROJECTIONS)
def test_inclusion_functional_has_test(projection_name: str) -> None:
    r"""Test that all projections have tests."""
    if projection_name != "identity":
        assert f"is_{projection_name}" in MATRIX_TESTS


@pytest.mark.parametrize("projection_name", MODULAR_PROJECTIONS)
def test_inclusion_modular_has_functional(projection_name: str) -> None:
    assert camel2snake(projection_name) in FUNCTIONAL_PROJECTIONS


@pytest.mark.parametrize("projection_name", FUNCTIONAL_PROJECTIONS)
def test_inclusion_functional_has_modular(projection_name: str) -> None:
    assert snake2camel(projection_name) in MODULAR_PROJECTIONS


@pytest.mark.parametrize("projection_name", FUNCTIONAL_PROJECTIONS)
def test_projections_work(projection_name: str) -> None:
    r"""Test that all projections work."""
    if projection_name == "identity":
        return

    projection = FUNCTIONAL_PROJECTIONS[projection_name]
    matrix_test = MATRIX_TESTS[f"is_{projection_name}"]
    x = torch.randn(4, 4)

    try:
        y = projection(x)
    except NotImplementedError as exc:
        raise pytest.skip(f"{projection_name} is not implemented.") from exc

    try:
        result = matrix_test(y)
    except NotImplementedError as exc:
        raise pytest.skip(f"test for {projection_name} is not implemented.") from exc

    assert result.item() is True


def test_rank_one_matches_low_rank_rank_1() -> None:
    r"""Test that `rank_one` is the rank-1 specialization of `low_rank`."""
    x = torch.randn(5, 4)

    assert torch.allclose(rank_one(x), low_rank(x, rank=1))
    assert torch.allclose(RankOne()(x), low_rank(x, rank=1))


def test_tridiagonal_matches_banded_1() -> None:
    r"""Test that `tridiagonal` is the tridiagonal specialization of `banded`."""
    x = torch.randn(5, 5)

    assert torch.allclose(tridiagonal(x), banded(x, lower=-1, upper=1))
    assert torch.allclose(Tridiagonal()(x), banded(x, lower=-1, upper=1))
