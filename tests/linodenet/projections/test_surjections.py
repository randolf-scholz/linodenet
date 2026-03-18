r"""Tests for linodenet.surjections."""

import pytest
import torch

from linodenet.mappings import (
    PositiveVector,
    StochasticVector,
)
from linodenet.testing import VECTOR_TESTS

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
