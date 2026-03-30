import pytest
import torch
from torch import nn

from linodenet.mappings import LowRankContraction
from tests.testing import DEVICES, TestSuite


class TestLowRankContraction(TestSuite):
    VALUE_ATOL = 1e-5
    VALUE_RTOL = 1e-5

    def test_init_signature_compatible_with_linear(self) -> None:
        layer = LowRankContraction(4, 3, rank=2, bias=False, dtype=torch.float64)

        assert isinstance(layer, nn.Module)
        assert layer.in_features == 4
        assert layer.out_features == 3
        assert layer.rank == 2
        assert layer.weight.dtype == torch.float64
        assert layer.bias is None

    def test_invalid_rank_raises(self) -> None:
        with pytest.raises(ValueError, match="rank must"):
            LowRankContraction(4, 3, rank=0)

        with pytest.raises(ValueError, match="rank must"):
            LowRankContraction(4, 3, rank=5)

    @pytest.mark.parametrize("device", DEVICES)
    def test_weight_is_low_rank_contraction(self, device: str) -> None:
        torch.manual_seed(0)
        c = 0.73
        layer = LowRankContraction(5, 4, rank=2, c=c, bias=False).to(device=device)

        assert layer.input_size == layer.in_features == 5
        assert layer.output_size == layer.out_features == 4

        weight = layer.weight
        sigma = torch.linalg.matrix_norm(weight, ord=2)
        rank = torch.linalg.matrix_rank(weight)

        self.assert_upper_bounded(sigma, c, atol=self.VALUE_ATOL, rtol=self.VALUE_RTOL)
        assert rank <= layer.rank

    @pytest.mark.parametrize("device", DEVICES)
    def test_forward_matches_dense_weight(self, device: str) -> None:
        torch.manual_seed(0)
        layer = LowRankContraction(4, 3, rank=2, bias=True).to(device=device)
        x = torch.randn(5, 4, device=device, dtype=layer.weight.dtype)

        expected = torch.nn.functional.linear(x, layer.weight, layer.bias)
        actual = layer(x)

        self.assert_close(actual, expected, atol=self.VALUE_ATOL, rtol=self.VALUE_RTOL)
