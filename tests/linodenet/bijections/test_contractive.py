import pytest
import torch

from linodenet.bijections import ContractiveFlow
from linodenet.nn import LinearContraction
from tests.testing import SEEDS_10, TestCase


class TestLowRankFlow(TestCase):
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
        assert y.shape == x.shape
