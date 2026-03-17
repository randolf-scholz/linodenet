import pytest
import torch

from linodenet.flows.transforms import ContractiveFlow
from linodenet.nn import LinearContraction
from tests.testing import SEEDS_10, TestCase


class TestContractiveFlow(TestCase):
    VALUE_ATOL = 1e-3
    VALUE_RTOL = 1e-3
    BATCH_SIZE = 128

    @pytest.mark.parametrize("seed", SEEDS_10, ids="seed={}".format)
    @pytest.mark.parametrize("input_size", [4, 16, 64, 256], ids="input_size={}".format)
    def test_invertibility(self, seed: int, input_size: int) -> None:
        r"""Check forward/inverse round trips; does not test logabsdet (not implemented yet)."""
        torch.manual_seed(seed)
        layer = LinearContraction(input_size, input_size, bias=True)
        flow = ContractiveFlow(layer)

        x = torch.randn(self.BATCH_SIZE, input_size)
        y = flow.encode(x)
        xhat = flow.decode(y)

        assert y.shape == x.shape
        assert xhat.shape == x.shape
        self.assert_close(xhat, x, atol=self.VALUE_ATOL, rtol=self.VALUE_RTOL)

        y = torch.randn(self.BATCH_SIZE, input_size)
        x = flow.decode(y)
        yhat = flow.encode(x)

        assert x.shape == y.shape
        assert yhat.shape == y.shape
        self.assert_close(yhat, y, atol=self.VALUE_ATOL, rtol=self.VALUE_RTOL)
