import pytest
import torch
from torch import Tensor, nn

from linodenet.mappings import LinearContraction
from linodenet.nn.parametrize import (
    get_parametrizations,
    is_parametrized,
    update_parametrizations,
)
from tests.testing import DEVICES, TestCase


class TestLinearContraction(TestCase):
    VALUE_ATOL = 1e-5
    VALUE_RTOL = 1e-5

    def test_init_signature_compatible_with_linear(self) -> None:
        layer = LinearContraction(4, 3, bias=False, dtype=torch.float64)

        assert isinstance(layer, nn.Linear)
        assert layer.in_features == 4
        assert layer.out_features == 3
        assert layer.weight.dtype == torch.float64
        assert layer.bias is None

    @pytest.mark.parametrize("device", DEVICES, ids=str)
    def test_weight_parametrization(self, device: str) -> None:
        torch.manual_seed(0)
        c = 0.73
        layer = LinearContraction(4, 3, c=c, bias=False).to(device=device)

        assert isinstance(layer, nn.Linear)
        assert layer.input_size == layer.in_features == 4
        assert layer.output_size == layer.out_features == 3
        assert is_parametrized(layer, "weight")

        parametrization = get_parametrizations(layer)["weight"]
        assert layer.weight is parametrization.cached_parameter

        sigma = torch.linalg.matrix_norm(layer.weight, ord=2)
        self.assert_upper_bounded(sigma, c, atol=self.VALUE_ATOL, rtol=self.VALUE_RTOL)

        cached_weight = layer.weight.detach().clone()
        assert isinstance(parametrization.original_parameter, Tensor)
        with torch.no_grad():
            parametrization.original_parameter.mul_(10)

        self.assert_close(
            layer.weight, cached_weight, atol=self.VALUE_ATOL, rtol=self.VALUE_RTOL
        )

        update_parametrizations(layer)

        self.assert_not_close(
            layer.weight, cached_weight, atol=self.VALUE_ATOL, rtol=self.VALUE_RTOL
        )
        sigma = torch.linalg.matrix_norm(layer.weight, ord=2)
        self.assert_upper_bounded(sigma, c, atol=self.VALUE_ATOL, rtol=self.VALUE_RTOL)
