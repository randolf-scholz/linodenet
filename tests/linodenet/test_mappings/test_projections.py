r"""Tests for linodenet.projections."""

import pytest
import torch

from linodenet.mappings import (
    MATRIX_PROJECTION_FNS,
    MATRIX_PROJECTIONS,
    MATRIX_PROJECTIONS_WITH_ARGS,
    PROJECTION_FNS,
    PROJECTIONS,
    RankOne,
    Tridiagonal,
    banded,
    low_rank,
    rank_one,
    tridiagonal,
)
from linodenet.registry import get_registry_entry
from tests.testing import camel2snake, snake2camel


def get_projection_test(name: str, /):
    r"""Return the registered test for a projection name."""
    entry = get_registry_entry(name)
    if callable(entry.test):
        return entry.test

    if name.endswith("_projection"):
        entry = get_registry_entry(name.removesuffix("_projection"))
        if callable(entry.test):
            return entry.test

    raise LookupError(f"No registry test found for {name!r}.")


@pytest.mark.parametrize("name", PROJECTIONS)
def test_functional_modular_both_present(name: str) -> None:
    assert snake2camel(name) in PROJECTIONS
    functional_name = camel2snake(name)
    if functional_name not in PROJECTION_FNS:
        functional_name = f"{functional_name}_projection"
    assert functional_name in PROJECTION_FNS


@pytest.mark.parametrize("name", MATRIX_PROJECTION_FNS)
def test_names_functional(name: str) -> None:
    r"""Test that all projections have the correct name."""
    projection = MATRIX_PROJECTION_FNS[name]
    actual_name = getattr(projection, "__name__", None)
    assert name == actual_name


@pytest.mark.parametrize("name", MATRIX_PROJECTIONS)
def test_names_modular(name: str) -> None:
    r"""Test that all modular projections have the correct name."""
    projection = MATRIX_PROJECTIONS[name]
    actual_name = getattr(projection, "__name__", None)
    assert actual_name in {name, f"{name}Projection"}


@pytest.mark.parametrize("name", MATRIX_PROJECTION_FNS | MATRIX_PROJECTIONS_WITH_ARGS)
def test_names_matrix_tests(name: str) -> None:
    r"""Test that all matrix tests have the correct name."""
    matrix_test = get_projection_test(name)
    actual_name = getattr(matrix_test, "__name__", None)
    expected_name = camel2snake(name.removesuffix("_projection"))
    assert actual_name == f"is_{expected_name}"


@pytest.mark.parametrize("name", MATRIX_PROJECTION_FNS | MATRIX_PROJECTIONS_WITH_ARGS)
def test_inclusion_functional_has_test(name: str) -> None:
    r"""Test that all projections have tests."""
    assert callable(get_projection_test(name))


@pytest.mark.parametrize("name", MATRIX_PROJECTION_FNS)
def test_projections_work(name: str) -> None:
    r"""Test that all projections work."""
    projection = MATRIX_PROJECTION_FNS[name]
    matrix_test = get_projection_test(name)
    x = torch.randn(4, 4)

    try:
        y = projection(x)
    except NotImplementedError as exc:
        raise pytest.skip(f"{name} is not implemented.") from exc

    try:
        result = matrix_test(y)
    except NotImplementedError as exc:
        raise pytest.skip(f"test for {name} is not implemented.") from exc

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
