r"""Tests for linodenet.bijections."""

import pytest
import torch

from linodenet.domains import MATRIX_TESTS, MatrixDomains, ScalarDomains
from linodenet.mappings import (
    BijectionBase,
    PositiveDiagonal,
    PositiveScalarMatrix,
    SmoothSoftsign,
    TanhMap,
)
from tests.testing import SEEDS_10


@pytest.mark.parametrize("bijection_cls", [TanhMap, SmoothSoftsign])
class TestScalarOpenUnitBallMap:
    @torch.no_grad()
    def test_roundtrip(self, bijection_cls: type[BijectionBase]) -> None:
        torch.manual_seed(0)
        bijection = bijection_cls()

        # Note: outside (-5,+5) torch.tanh collapses to ±1
        x = torch.linspace(-4, 4, 128)
        y = bijection(x)

        assert ScalarDomains.OPEN_UNIT_BALL.check(y).all()
        torch.testing.assert_close(bijection.inverse(y), x, atol=1e-5, rtol=1e-5)

    def test_inverse_roundtrip_on_codomain_samples(
        self, bijection_cls: type[BijectionBase]
    ) -> None:
        torch.manual_seed(0)
        bijection = bijection_cls()
        y = torch.linspace(-0.95, 0.95, steps=32).reshape(1, 32)

        torch.testing.assert_close(
            bijection(bijection.inverse(y)), y, atol=1e-6, rtol=1e-6
        )

    @pytest.mark.parametrize("seed", SEEDS_10, ids="seed={}".format)
    @torch.no_grad()
    def test_handles_batches(
        self, bijection_cls: type[BijectionBase], seed: int
    ) -> None:
        torch.manual_seed(seed)
        bijection = bijection_cls()

        x = torch.randn(2, 3, 4, 5)
        y = bijection(x)

        assert y.shape == x.shape
        assert ScalarDomains.OPEN_UNIT_BALL.check(y).all()


class TestPositiveDiagonal:
    @torch.no_grad()
    def test_roundtrip(self) -> None:
        torch.manual_seed(0)
        bijection = PositiveDiagonal()
        x = torch.linspace(-4, 4, steps=32).reshape(4, 8)

        y = bijection(x)

        assert y.shape == (4, 8, 8)
        assert MATRIX_TESTS[MatrixDomains.DIAGONAL](y).all()
        assert MATRIX_TESTS[MatrixDomains.POSITIVE_DIAGONAL](y).all()
        assert MATRIX_TESTS[MatrixDomains.POSITIVE_DEFINITE](y).all()
        torch.testing.assert_close(y, torch.diag_embed(torch.exp(x)))
        torch.testing.assert_close(bijection.inverse(y), x)

    @pytest.mark.parametrize("seed", SEEDS_10, ids="seed={}".format)
    @torch.no_grad()
    def test_handles_batches(self, seed: int) -> None:
        torch.manual_seed(seed)
        bijection = PositiveDiagonal()
        x = torch.randn(2, 3, 5)

        y = bijection(x)

        assert y.shape == (2, 3, 5, 5)
        torch.testing.assert_close(bijection.inverse(y), x)


class TestPositiveScalarMatrix:
    @torch.no_grad()
    def test_roundtrip(self) -> None:
        torch.manual_seed(0)
        bijection = PositiveScalarMatrix(size=5)
        x = torch.linspace(-4, 4, steps=12).reshape(3, 4)

        y = bijection(x)

        assert y.shape == (3, 4, 5, 5)
        assert MATRIX_TESTS[MatrixDomains.POSITIVE_SCALAR_MATRIX](y).all()
        torch.testing.assert_close(
            y,
            torch.exp(x).unsqueeze(-1).unsqueeze(-1)
            * torch.eye(5, dtype=x.dtype, device=x.device),
        )
        torch.testing.assert_close(bijection.inverse(y), x)

    @pytest.mark.parametrize("seed", SEEDS_10, ids="seed={}".format)
    @torch.no_grad()
    def test_handles_batches(self, seed: int) -> None:
        torch.manual_seed(seed)
        bijection = PositiveScalarMatrix(size=3)
        x = torch.randn(2, 3, 4)

        y = bijection(x)

        assert y.shape == (2, 3, 4, 3, 3)
        torch.testing.assert_close(bijection.inverse(y), x)
