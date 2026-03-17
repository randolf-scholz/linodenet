import pytest
import torch

from linodenet.bijections import LowRankFlow
from tests.testing import SEEDS_10, TestCase


class TestLowRankFlow(TestCase):
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
        flow = LowRankFlow(input_size, rank=min(rank, input_size))

        x = torch.randn(self.BATCH_SIZE, input_size)
        y, forward_logabsdet = flow.encode_and_logabsdet(x)
        xhat, inverse_logabsdet = flow.decode_and_logabsdet(y)

        assert y.shape == x.shape
        assert forward_logabsdet.shape == (self.BATCH_SIZE,)
        assert xhat.shape == x.shape
        assert inverse_logabsdet.shape == (self.BATCH_SIZE,)

        self.assert_close(xhat, x, atol=self.VALUE_ATOL, rtol=self.VALUE_RTOL)
        self.assert_close(
            forward_logabsdet + inverse_logabsdet,
            torch.zeros_like(forward_logabsdet),
            atol=self.LOGABSDET_ATOL,
            rtol=self.LOGABSDET_RTOL,
        )

        y = torch.randn(self.BATCH_SIZE, input_size)
        x, inverse_logabsdet = flow.decode_and_logabsdet(y)
        yhat, forward_logabsdet = flow.encode_and_logabsdet(x)

        assert x.shape == y.shape
        assert inverse_logabsdet.shape == (self.BATCH_SIZE,)
        assert yhat.shape == y.shape
        assert forward_logabsdet.shape == (self.BATCH_SIZE,)

        self.assert_close(yhat, y, atol=self.VALUE_ATOL, rtol=self.VALUE_RTOL)
        self.assert_close(
            inverse_logabsdet + forward_logabsdet,
            torch.zeros_like(inverse_logabsdet),
            atol=self.LOGABSDET_ATOL,
            rtol=self.LOGABSDET_RTOL,
        )
