r"""Tests for linodenet.bijections."""

import pytest
import torch

from linodenet.domains import ScalarDomains
from linodenet.mappings import BijectionBase, SmoothSoftsign, TanhMap
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
