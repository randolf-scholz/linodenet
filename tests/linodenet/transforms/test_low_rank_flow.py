import pytest
import torch

from linodenet.mappings.transforms import LowRankTransform, SymmetricLowRankTransform
from tests.testing import SEEDS_10

from .base import TestTransform


class TestLowRankFlow(TestTransform):
    VALUE_ATOL = 1e-3
    VALUE_RTOL = 1e-5
    LOGABSDET_ATOL = 1e-5
    LOGABSDET_RTOL = 1e-5
    BATCH_SIZE = 128

    @pytest.mark.parametrize("seed", SEEDS_10, ids="seed={}".format)
    @pytest.mark.parametrize("input_size", [4, 16, 64, 256], ids="input_size={}".format)
    @pytest.mark.parametrize("rank", [1, 2, 4], ids="rank={}".format)
    def test_invertibility(self, seed: int, input_size: int, rank: int) -> None:
        r"""Check forward/inverse round trips and logabsdet cancellation."""
        torch.manual_seed(seed)
        flow = LowRankTransform(input_size, rank=min(rank, input_size))
        with torch.no_grad():
            flow.S.copy_(torch.linspace(0.5, 1.5, flow.rank))

        x = torch.randn(self.BATCH_SIZE, input_size)
        y = torch.randn(self.BATCH_SIZE, input_size)
        self.assert_invertible(
            flow,
            x,
            y,
            atol=self.VALUE_ATOL,
            rtol=self.VALUE_RTOL,
            logdet_atol=self.LOGABSDET_ATOL,
            logdet_rtol=self.LOGABSDET_RTOL,
        )


class TestSymmetricLowRankFlow(TestTransform):
    VALUE_ATOL = 1e-3
    VALUE_RTOL = 1e-5
    LOGABSDET_ATOL = 1e-5
    LOGABSDET_RTOL = 1e-5
    BATCH_SIZE = 128

    @pytest.mark.parametrize("seed", SEEDS_10, ids="seed={}".format)
    @pytest.mark.parametrize("input_size", [4, 16, 64, 256], ids="input_size={}".format)
    @pytest.mark.parametrize("rank", [1, 2, 4], ids="rank={}".format)
    def test_invertibility(self, seed: int, input_size: int, rank: int) -> None:
        r"""Check forward/inverse round trips and logabsdet cancellation."""
        torch.manual_seed(seed)
        flow = SymmetricLowRankTransform(input_size, rank=min(rank, input_size))

        x = torch.randn(self.BATCH_SIZE, input_size)
        y = torch.randn(self.BATCH_SIZE, input_size)
        self.assert_invertible(
            flow,
            x,
            y,
            atol=self.VALUE_ATOL,
            rtol=self.VALUE_RTOL,
            logdet_atol=self.LOGABSDET_ATOL,
            logdet_rtol=self.LOGABSDET_RTOL,
        )
