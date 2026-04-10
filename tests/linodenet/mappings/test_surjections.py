r"""Tests for linodenet.surjections."""

import pytest
import torch

from linodenet.domains.matrix_tests import is_orthogonal, is_row_orthogonal
from linodenet.mappings import (
    CholeskyFactor,
    NegativeDefinite,
    Orthogonal,
    OrthogonalCayley,
    OrthogonalHouseholder,
    PositiveDefinite,
    PositiveVector,
    SpecialOrthogonal,
    StochasticVector,
    Surjection,
)
from linodenet.registry import get_registry_entry
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
@pytest.mark.parametrize("seed", SEEDS_10, ids="seed={}".format)
def test_modular_surjections_work(name: str, seed: int) -> None:
    torch.manual_seed(seed)
    surjection = SURJECTION_MODULES[name]()
    vector_test = get_surjection_test(type(surjection))

    x_single = torch.randn(5)
    y_single = surjection(x_single)
    assert vector_test(y_single).all()

    x_batch = torch.randn(8, 5)
    y_batch = surjection(x_batch)
    assert vector_test(y_batch).all()


@pytest.mark.parametrize(
    "surjection_cls",
    [
        SpecialOrthogonal,
        OrthogonalCayley,
        OrthogonalHouseholder,
        Orthogonal,
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


@pytest.mark.parametrize("seed", SEEDS_10, ids="seed={}".format)
def test_orthogonal_householder_rows_mode(seed: int) -> None:
    torch.manual_seed(seed)
    surjection = OrthogonalHouseholder(mode="rows")

    x = torch.randn(8, 3, 5)
    y = surjection(x)

    assert is_row_orthogonal(y).all()
    z = surjection.right_inverse(y)
    assert torch.allclose(surjection(z), y, atol=1e-5, rtol=1e-5)


def test_cholesky_factor_surjection() -> None:
    surjection = CholeskyFactor()
    matrix_test = get_surjection_test(type(surjection))

    x = torch.randn(8, 5, 5).tril()
    y = surjection(x)

    assert matrix_test(y).all()

    z = surjection.right_inverse(y)
    assert torch.allclose(z, z.tril())
    assert torch.allclose(surjection(z), y, atol=1e-5, rtol=1e-5)
    assert torch.all(y.diagonal(dim1=-2, dim2=-1) > 0)


@pytest.mark.parametrize("surjection_cls", [PositiveDefinite, NegativeDefinite])
def test_cholesky_surjection(surjection_cls: type[Surjection]) -> None:
    surjection = surjection_cls()
    matrix_test = get_surjection_test(type(surjection))

    x = torch.randn(8, 5, 5).tril()
    y = surjection(x)

    assert matrix_test(y).all()

    z = surjection.right_inverse(y)
    assert torch.allclose(z, z.tril())
    assert torch.allclose(surjection(z), y, atol=1e-5, rtol=1e-5)

    lower = z.tril(diagonal=-1) + torch.diag_embed(
        torch.exp(z.diagonal(dim1=-2, dim2=-1))
    )
    assert torch.all(lower.diagonal(dim1=-2, dim2=-1) > 0)


@pytest.mark.parametrize("surjection_cls", [PositiveDefinite, NegativeDefinite])
def test_cholesky_surjection_right_inverse(
    surjection_cls: type[Surjection],
) -> None:
    surjection = surjection_cls()
    factor = torch.tensor([[2.0, 0.0, 0.0], [1.0, 3.0, 0.0], [-3.0, 2.0, 1.0]])
    y = surjection(factor)

    z = surjection.right_inverse(y)

    assert torch.allclose(z, z.tril())
    assert torch.allclose(surjection(z), y, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize(
    ("surjection_cls", "y"),
    [
        (
            PositiveDefinite,
            torch.tensor([[4.0, 2.0, -6.0], [2.0, 1.0, -3.0], [-6.0, -3.0, 9.0]]),
        ),
        (
            NegativeDefinite,
            -torch.tensor([[4.0, 2.0, -6.0], [2.0, 1.0, -3.0], [-6.0, -3.0, 9.0]]),
        ),
    ],
)
def test_cholesky_surjection_right_inverse_singular_psd_raises(
    surjection_cls: type[Surjection], y: torch.Tensor
) -> None:
    surjection = surjection_cls()

    with pytest.raises(RuntimeError):
        surjection.right_inverse(y)
