r"""Tests for linodenet.surjections."""

import pytest
import torch

from linodenet.mappings import (
    OrthogonalCayley,
    OrthogonalHouseholder,
    OrthogonalMatExp,
    OrthogonalProjection,
    PositiveVector,
    StochasticVector,
    Surjection,
)
from linodenet.registry import get_registry_entry
from linodenet.testing import is_orthogonal
from tests.testing import SEEDS_10

SURJECTION_MODULES = {
    "NonNegativeVector": PositiveVector,
    "StochasticVector": StochasticVector,
}


def get_surjection_test(surjection_cls: type[Surjection], /):
    r"""Return the registered test for a surjection class."""
    entry = get_registry_entry(surjection_cls.__name__)
    if callable(entry.test):
        return entry.test

    raise LookupError(f"No registry test found for {surjection_cls.__name__!r}.")


@pytest.mark.parametrize("name", SURJECTION_MODULES)
def test_modular_surjections_work(name: str) -> None:
    surjection = SURJECTION_MODULES[name]()
    vector_test = get_surjection_test(type(surjection))

    x = torch.randn(8, 5)
    y = surjection(x)

    assert vector_test(y).all()


@pytest.mark.parametrize(
    "surjection_cls",
    [
        OrthogonalMatExp,
        OrthogonalCayley,
        OrthogonalHouseholder,
        OrthogonalProjection,
    ],
)
@pytest.mark.parametrize("seed", SEEDS_10, ids="seed={}".format)
def test_orthogonal_maps(surjection_cls: type[Surjection], seed: int) -> None:
    torch.manual_seed(seed)
    surjection = surjection_cls()

    x = torch.randn(8, 5, 5)
    y = surjection(x)

    assert is_orthogonal(y).all()
    z = surjection.right_inverse(y)
    assert torch.allclose(surjection(z), y, atol=1e-5, rtol=1e-5)
