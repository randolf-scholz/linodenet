from contextlib import nullcontext

import pytest
import torch
from torch import Tensor, nn

from linodenet.mappings import LinearContraction
from linodenet.nn.parametrize import (
    get_parametrizations,
    is_parametrized,
    update_parametrizations,
)
from tests.testing import DEVICES, TestSuite


class TestLinearContraction(TestSuite):
    VALUE_ATOL = 1e-5
    VALUE_RTOL = 1e-5

    def test_init_signature_compatible_with_linear(self) -> None:
        layer = LinearContraction(4, 3, bias=False, dtype=torch.float64)

        assert isinstance(layer, nn.Linear)
        assert layer.in_features == 4
        assert layer.out_features == 3
        assert layer.weight.dtype == torch.float64
        assert layer.bias is None

    def test_update_parametrizations_heals_weight_alias_after_to(self) -> None:
        layer = LinearContraction(4, 3, bias=False)
        parametrizations = get_parametrizations(layer)
        assert parametrizations is not None
        parametrization = parametrizations["weight"]
        assert layer.weight is parametrization.cached_parameter

        layer = layer.to(dtype=torch.float64)
        parametrizations = get_parametrizations(layer)
        assert parametrizations is not None
        parametrization = parametrizations["weight"]

        assert layer.weight is not parametrization.cached_parameter
        assert layer.weight.dtype == torch.float64

        cached_weight = layer.weight.detach().clone()
        assert isinstance(parametrization.original_parameter, Tensor)
        with torch.no_grad():
            parametrization.original_parameter.add_(
                torch.ones_like(parametrization.original_parameter)
            )

        update_parametrizations(layer)

        assert layer.weight is parametrization.cached_parameter
        self.assert_not_close(
            layer.weight, cached_weight, atol=self.VALUE_ATOL, rtol=self.VALUE_RTOL
        )

    @pytest.mark.parametrize("device", DEVICES)
    def test_weight_parametrization(self, device: str) -> None:
        torch.manual_seed(0)
        c = 0.73
        layer = LinearContraction(4, 3, c=c, bias=False)
        assert isinstance(layer, nn.Linear)
        assert layer.input_size == layer.in_features == 4
        assert layer.output_size == layer.out_features == 3
        assert is_parametrized(layer, "weight")

        parametrizations = get_parametrizations(layer)
        assert parametrizations is not None
        parametrization = parametrizations["weight"]
        assert layer.weight is parametrization.cached_parameter

        # NOTE: using .to() screws up the parametrization
        layer = layer.to(device=device)
        parametrizations = get_parametrizations(layer)
        assert parametrizations is not None
        parametrization = parametrizations["weight"]
        with pytest.raises(AssertionError) if device == "cuda" else nullcontext():
            assert layer.weight is parametrization.cached_parameter

        # Note: We can heal it by doing update_parametrizations
        update_parametrizations(layer)
        parametrizations = get_parametrizations(layer)
        assert parametrizations is not None
        parametrization = parametrizations["weight"]
        assert layer.weight is parametrization.cached_parameter

        sigma = torch.linalg.matrix_norm(layer.weight, ord=2)
        self.assert_upper_bounded(sigma, c, atol=self.VALUE_ATOL, rtol=self.VALUE_RTOL)

        cached_weight = layer.weight.detach().clone()
        assert isinstance(parametrization.original_parameter, Tensor)
        with torch.no_grad():
            parametrization.original_parameter.add_(
                torch.ones_like(parametrization.original_parameter)
            )

        self.assert_close(
            layer.weight, cached_weight, atol=self.VALUE_ATOL, rtol=self.VALUE_RTOL
        )

        update_parametrizations(layer)

        self.assert_not_close(
            layer.weight, cached_weight, atol=self.VALUE_ATOL, rtol=self.VALUE_RTOL
        )
        sigma = torch.linalg.matrix_norm(layer.weight, ord=2)
        self.assert_upper_bounded(sigma, c, atol=self.VALUE_ATOL, rtol=self.VALUE_RTOL)
