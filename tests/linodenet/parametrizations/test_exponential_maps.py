r"""Tests for Riemannian manifolds."""

import pytest
import torch

from linodenet.domains import MATRIX_TESTS, VECTOR_TESTS, MatrixDomains, VectorDomains
from linodenet.parametrizations.exponential_maps import (
    ManifoldBase,
    MatrixLieGroup,
    PositiveDefiniteManifold,
    RiemannManifold,
    SpecialOrthogonalManifold,
    SphereManifold,
)
from tests.testing import SEEDS_10

DTYPE = torch.float64

MATRIX_MANIFOLDS = [
    (PositiveDefiniteManifold, MatrixDomains.POSITIVE_DEFINITE, (5, 5), 0.05),
    (
        SpecialOrthogonalManifold,
        MatrixDomains.SPECIAL_ORTHOGONAL,
        (5, 5),
        0.2,
    ),
]

VECTOR_MANIFOLDS = [
    (SphereManifold, VectorDomains.UNIT_VECTOR, 5, 0.2),
]


@pytest.mark.parametrize(
    ("manifold_cls", "domain", "shape", "scale"),
    MATRIX_MANIFOLDS,
    ids=lambda item: getattr(item, "__name__", str(item)),
)
def test_matrix_manifolds_implement_protocol(
    manifold_cls: type[ManifoldBase],
    domain: MatrixDomains,  # noqa: ARG001
    shape: tuple[int, int],  # noqa: ARG001
    scale: float,  # noqa: ARG001
) -> None:
    assert isinstance(manifold_cls(), ManifoldBase)


@pytest.mark.parametrize(
    ("manifold_cls", "domain", "size", "scale"),
    VECTOR_MANIFOLDS,
    ids=lambda item: getattr(item, "__name__", str(item)),
)
def test_vector_manifolds_implement_protocol(
    manifold_cls: type[ManifoldBase],
    domain: VectorDomains,  # noqa: ARG001
    size: int,  # noqa: ARG001
    scale: float,  # noqa: ARG001
) -> None:
    assert isinstance(manifold_cls(), ManifoldBase)


@pytest.mark.parametrize(
    "manifold",
    [PositiveDefiniteManifold(), SpecialOrthogonalManifold(), SphereManifold()],
    ids=type,
)
def test_riemann_manifolds_implement_protocol(manifold: ManifoldBase) -> None:
    assert isinstance(manifold, RiemannManifold)


def test_special_orthogonal_implements_lie_group_protocol() -> None:
    assert isinstance(SpecialOrthogonalManifold(), MatrixLieGroup)
    assert not isinstance(PositiveDefiniteManifold(), ManifoldBase)


@pytest.mark.parametrize(
    ("manifold_cls", "domain", "shape", "scale"),
    MATRIX_MANIFOLDS,
    ids=lambda item: getattr(item, "__name__", str(item)),
)
@pytest.mark.parametrize("seed", SEEDS_10, ids="seed={}".format)
def test_project_lands_in_expected_manifold(
    manifold_cls: type[ManifoldBase],
    domain: MatrixDomains,
    shape: tuple[int, int],
    scale: float,
    seed: int,
) -> None:
    torch.manual_seed(seed)
    manifold = manifold_cls()
    x = scale * torch.randn(4, *shape, dtype=DTYPE)
    y = manifold.project_manifold(x)
    assert MATRIX_TESTS[domain](y).all()


@pytest.mark.parametrize(
    ("manifold_cls", "domain", "shape", "scale"),
    MATRIX_MANIFOLDS,
    ids=lambda item: getattr(item, "__name__", str(item)),
)
@pytest.mark.parametrize("seed", SEEDS_10, ids="seed={}".format)
def test_project_tangent_lands_in_expected_tangent_space(
    manifold_cls: type[ManifoldBase],
    domain: MatrixDomains,  # noqa: ARG001
    shape: tuple[int, int],
    scale: float,
    seed: int,
) -> None:
    torch.manual_seed(seed)
    manifold = manifold_cls()
    x = manifold.project_manifold(scale * torch.randn(4, *shape, dtype=DTYPE))
    v = manifold.project_tangent(x, torch.randn(4, *shape, dtype=DTYPE))

    if isinstance(manifold, PositiveDefiniteManifold):
        assert MATRIX_TESTS[MatrixDomains.SYMMETRIC](v).all()
        return

    assert isinstance(manifold, SpecialOrthogonalManifold)
    assert MATRIX_TESTS[MatrixDomains.SKEW_SYMMETRIC](x.mT @ v).all()


@pytest.mark.parametrize(
    ("manifold_cls", "domain", "shape", "scale"),
    MATRIX_MANIFOLDS,
    ids=lambda item: getattr(item, "__name__", str(item)),
)
@pytest.mark.parametrize("seed", SEEDS_10, ids="seed={}".format)
def test_exp_lands_in_expected_manifold(
    manifold_cls: type[ManifoldBase],
    domain: MatrixDomains,
    shape: tuple[int, int],
    scale: float,
    seed: int,
) -> None:
    torch.manual_seed(seed)
    manifold = manifold_cls()
    x = manifold.project_manifold(torch.randn(4, *shape, dtype=DTYPE))
    v = manifold.project_tangent(x, scale * torch.randn(4, *shape, dtype=DTYPE))
    y = manifold.exp(x, v)

    assert MATRIX_TESTS[domain](y).all()


@pytest.mark.parametrize(
    ("manifold_cls", "_domain", "shape", "scale"),
    MATRIX_MANIFOLDS,
    ids=lambda item: getattr(item, "__name__", str(item)),
)
@pytest.mark.parametrize("seed", SEEDS_10, ids="seed={}".format)
def test_log_exp_roundtrip(
    manifold_cls: type[ManifoldBase],
    _domain: MatrixDomains,
    shape: tuple[int, int],
    scale: float,
    seed: int,
) -> None:
    torch.manual_seed(seed)
    manifold = manifold_cls()
    x = manifold.project_manifold(torch.randn(4, *shape, dtype=DTYPE))
    v = manifold.project_tangent(x, scale * torch.randn(4, *shape, dtype=DTYPE))
    y = manifold.exp(x, v)
    z = manifold.log(x, y)

    torch.testing.assert_close(z, v, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(manifold.exp(x, z), y, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("seed", SEEDS_10, ids="seed={}".format)
def test_special_orthogonal_lie_algebra_roundtrip(seed: int) -> None:
    torch.manual_seed(seed)
    manifold = SpecialOrthogonalManifold()
    x = manifold.project_manifold(torch.randn(4, 5, 5, dtype=DTYPE))
    algebra = manifold.project_algebra(0.05 * torch.randn(4, 5, 5, dtype=DTYPE))
    tangent = manifold.from_algebra(x, algebra)

    recovered = manifold.to_algebra(x, tangent)
    y = manifold.exp_identity(algebra)
    z = manifold.log_identity(y)

    assert MATRIX_TESTS[MatrixDomains.SKEW_SYMMETRIC](recovered).all()
    assert MATRIX_TESTS[MatrixDomains.SPECIAL_ORTHOGONAL](y).all()
    torch.testing.assert_close(recovered, algebra, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(z, algebra, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize(
    ("manifold_cls", "domain", "size", "scale"),
    VECTOR_MANIFOLDS,
    ids=lambda item: getattr(item, "__name__", str(item)),
)
@pytest.mark.parametrize("seed", SEEDS_10, ids="seed={}".format)
def test_project_lands_in_expected_vector_manifold(
    manifold_cls: type[ManifoldBase],
    domain: VectorDomains,
    size: int,
    scale: float,
    seed: int,
) -> None:
    torch.manual_seed(seed)
    manifold = manifold_cls()
    x = scale * torch.randn(4, size, dtype=DTYPE)
    y = manifold.project_manifold(x)
    assert VECTOR_TESTS[domain](y).all()


@pytest.mark.parametrize(
    ("manifold_cls", "domain", "size", "scale"),
    VECTOR_MANIFOLDS,
    ids=lambda item: getattr(item, "__name__", str(item)),
)
@pytest.mark.parametrize("seed", SEEDS_10, ids="seed={}".format)
def test_project_tangent_lands_in_expected_vector_tangent_space(
    manifold_cls: type[ManifoldBase],
    domain: VectorDomains,  # noqa: ARG001
    size: int,
    scale: float,
    seed: int,
) -> None:
    torch.manual_seed(seed)
    manifold = manifold_cls()
    x = manifold.project_manifold(scale * torch.randn(4, size, dtype=DTYPE))
    v = manifold.project_tangent(x, torch.randn(4, size, dtype=DTYPE))
    zeros = torch.zeros(v.shape[:-1], dtype=v.dtype, device=v.device)
    torch.testing.assert_close((x * v).sum(dim=-1), zeros, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize(
    ("manifold_cls", "domain", "size", "scale"),
    VECTOR_MANIFOLDS,
    ids=lambda item: getattr(item, "__name__", str(item)),
)
@pytest.mark.parametrize("seed", SEEDS_10, ids="seed={}".format)
def test_exp_lands_in_expected_vector_manifold(
    manifold_cls: type[ManifoldBase],
    domain: VectorDomains,
    size: int,
    scale: float,
    seed: int,
) -> None:
    torch.manual_seed(seed)
    manifold = manifold_cls()
    x = manifold.project_manifold(torch.randn(4, size, dtype=DTYPE))
    v = manifold.project_tangent(x, scale * torch.randn(4, size, dtype=DTYPE))
    y = manifold.exp(x, v)
    assert VECTOR_TESTS[domain](y).all()


@pytest.mark.parametrize(
    ("manifold_cls", "_domain", "size", "scale"),
    VECTOR_MANIFOLDS,
    ids=lambda item: getattr(item, "__name__", str(item)),
)
@pytest.mark.parametrize("seed", SEEDS_10, ids="seed={}".format)
def test_log_exp_roundtrip_vector(
    manifold_cls: type[ManifoldBase],
    _domain: VectorDomains,
    size: int,
    scale: float,
    seed: int,
) -> None:
    torch.manual_seed(seed)
    manifold = manifold_cls()
    x = manifold.project_manifold(torch.randn(4, size, dtype=DTYPE))
    v = manifold.project_tangent(x, scale * torch.randn(4, size, dtype=DTYPE))
    y = manifold.exp(x, v)
    z = manifold.log(x, y)
    torch.testing.assert_close(z, v, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(manifold.exp(x, z), y, atol=1e-5, rtol=1e-5)
