import pytest
import torch
from torch import nn

from linodenet.mappings import RankOneContraction
from tests.testing import DEVICES, TestCase


class TestRankOneContraction(TestCase):
    VALUE_ATOL = 1e-5
    VALUE_RTOL = 1e-5

    def test_init_signature_compatible_with_linear(self) -> None:
        layer = RankOneContraction(4, 3, bias=False, dtype=torch.float64)

        assert isinstance(layer, nn.Module)
        assert layer.in_features == 4
        assert layer.out_features == 3
        assert layer.weight.dtype == torch.float64
        assert layer.bias is None

    @pytest.mark.parametrize("device", DEVICES)
    def test_weight_is_rank_one_contraction(self, device: str) -> None:
        torch.manual_seed(0)
        c = 0.73
        layer = RankOneContraction(4, 3, c=c, bias=False).to(device=device)

        assert layer.input_size == layer.in_features == 4
        assert layer.output_size == layer.out_features == 3

        weight = layer.weight
        sigma = torch.linalg.matrix_norm(weight, ord=2)
        rank = torch.linalg.matrix_rank(weight)

        self.assert_upper_bounded(sigma, c, atol=self.VALUE_ATOL, rtol=self.VALUE_RTOL)
        assert rank <= 1

    @pytest.mark.parametrize("device", DEVICES)
    def test_forward_matches_dense_weight(self, device: str) -> None:
        torch.manual_seed(0)
        layer = RankOneContraction(4, 3, bias=True).to(device=device)
        x = torch.randn(5, 4, device=device, dtype=layer.weight.dtype)

        expected = torch.nn.functional.linear(x, layer.weight, layer.bias)
        actual = layer(x)

        self.assert_close(actual, expected, atol=self.VALUE_ATOL, rtol=self.VALUE_RTOL)
