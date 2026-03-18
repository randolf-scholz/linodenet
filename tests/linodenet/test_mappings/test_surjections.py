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
from linodenet.testing import MATRIX_TESTS, VECTOR_TESTS
from tests.testing import SEEDS_10

SURJECTION_MODULES = {
    "NonNegativeVector": PositiveVector,
    "StochasticVector": StochasticVector,
}

SURJECTION_TESTS = {
    "unit_vector": VECTOR_TESTS["is_unit_vector"],
    "positive_vector": VECTOR_TESTS["is_positive_vector"],
    "stochastic_vector": VECTOR_TESTS["is_stochastic_vector"],
    "UnitVector": VECTOR_TESTS["is_unit_vector"],
    "NonNegativeVector": VECTOR_TESTS["is_positive_vector"],
    "StochasticVector": VECTOR_TESTS["is_stochastic_vector"],
}


@pytest.mark.parametrize("name", SURJECTION_MODULES)
def test_modular_surjections_work(name: str) -> None:
    surjection = SURJECTION_MODULES[name]()
    vector_test = SURJECTION_TESTS[name]

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
    matrix_test = MATRIX_TESTS["is_orthogonal"]

    x = torch.randn(8, 5, 5)
    y = surjection(x)

    assert matrix_test(y).all()
    z = surjection.right_inverse(y)
    assert torch.allclose(surjection(z), y, atol=1e-5, rtol=1e-5)
