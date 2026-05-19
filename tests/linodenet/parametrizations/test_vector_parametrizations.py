r"""Tests for vector parametrizations."""

import pytest
import torch
from torch import Tensor, nn
from torch._dynamo import OptimizedModule
from torch.fx import GraphModule
from torch.nn.functional import mse_loss
from torch.optim import SGD

from linodenet.domains import VectorDomains, VectorTest
from linodenet.parametrizations import (
    VECTOR_PARAMETRIZATIONS,
    ParametrizationBase,
    get_parametrizations,
    register_optimizer_hook,
    register_parametrization,
    update_parametrizations,
)
from linodenet.registry import get_registry_entry
from tests.testing import DEVICES, TestSuite, pytest_xfail

VECTOR_SIZE = 4


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("name", VECTOR_PARAMETRIZATIONS)
class TestParametrization(TestSuite):
    BATCH_SIZE = 8
    VALUE_ATOL = 1e-6
    VALUE_RTOL = 1e-6
    NUM_ITERATIONS = 3

    def make_test_case(self, /, *, device: str) -> tuple[nn.Sequential, Tensor, Tensor]:
        model = nn.Sequential(
            nn.Linear(VECTOR_SIZE, VECTOR_SIZE, bias=False),
            nn.ReLU(),
            nn.Linear(VECTOR_SIZE, VECTOR_SIZE, bias=True),
            nn.ReLU(),
            nn.Linear(VECTOR_SIZE, VECTOR_SIZE, bias=False),
        ).to(device=device)
        x = torch.randn(self.BATCH_SIZE, VECTOR_SIZE, device=device)
        y = torch.randn(self.BATCH_SIZE, VECTOR_SIZE, device=device)
        return model, x, y

    def get_parametrized_layer(self, model: nn.Module) -> nn.Linear:
        if isinstance(model, OptimizedModule):
            model = model._orig_mod  # noqa: SLF001
            assert isinstance(model, nn.Sequential)
            layer = model[2]
            assert isinstance(layer, nn.Linear)
        if isinstance(model, GraphModule):
            children = dict(model.named_children())
            layer = children["2"]
            assert isinstance(layer, nn.Linear)
        else:
            assert isinstance(model, nn.Sequential)
            layer = model[2]
            assert isinstance(layer, nn.Linear)
        return layer

    def get_bias_parametrization(self, layer: nn.Linear, /) -> ParametrizationBase:
        parametrizations = get_parametrizations(layer)
        assert parametrizations is not None
        parametrization = parametrizations["bias"]
        assert isinstance(parametrization, ParametrizationBase)
        return parametrization

    def get_parametrization(self, name: str, /) -> nn.Module:
        cls = VECTOR_PARAMETRIZATIONS[name]
        assert issubclass(cls, nn.Module)
        return cls()

    def get_vector_test(self, name: str, /) -> VectorTest:
        entry = get_registry_entry(name)
        assert callable(entry.test)
        return entry.test

    def get_domain(self, name: str, /) -> VectorDomains:
        entry = get_registry_entry(name)
        domain = entry.domain
        assert domain is not None
        assert type(domain) is VectorDomains
        return domain

    def check_parametrization(self, name: str, model: nn.Module) -> None:
        vector_test = self.get_vector_test(name)
        bias = self.get_parametrized_layer(model).bias
        assert isinstance(bias, Tensor)
        assert vector_test(bias)

    def assert_stale(self, parametrization: nn.Module, expected: bool) -> None:
        is_stale = getattr(parametrization, "is_stale", None)
        assert is_stale is not None
        assert isinstance(is_stale, Tensor)
        assert is_stale.dtype == torch.bool
        assert bool(is_stale) is expected

    @pytest_xfail(raises=NotImplementedError, strict=False)
    def test_register_parametrization(self, name: str, device: str) -> None:
        self.get_domain(name)
        model, _, _ = self.make_test_case(device=device)
        layer = self.get_parametrized_layer(model)
        register_parametrization(layer, "bias", self.get_parametrization(name))

        parametrization = self.get_bias_parametrization(layer)
        assert layer.bias is parametrization.cached_parameter
        self.assert_stale(parametrization, False)
        self.check_parametrization(name, model)

    @pytest_xfail(raises=NotImplementedError, strict=False)
    def test_forward_uses_cached_parameter(self, name: str, device: str) -> None:
        torch.manual_seed(0)
        self.get_domain(name)
        model, x, _ = self.make_test_case(device=device)
        layer = self.get_parametrized_layer(model)
        register_parametrization(layer, "bias", self.get_parametrization(name))
        parametrization = self.get_bias_parametrization(layer)
        self.assert_stale(parametrization, False)

        y0 = model(x)
        cached_bias = layer.bias.clone()

        with torch.no_grad():
            parametrization.original_parameter.add_(
                torch.randn_like(parametrization.original_parameter)
            )

        y1 = model(x)
        self.assert_close(y1, y0, atol=self.VALUE_ATOL, rtol=self.VALUE_RTOL)
        self.assert_close(
            layer.bias, cached_bias, atol=self.VALUE_ATOL, rtol=self.VALUE_RTOL
        )

        update_parametrizations(model)
        self.assert_stale(parametrization, False)
        y2 = model(x)
        assert not torch.allclose(y2, y0)
        self.check_parametrization(name, model)

    @pytest_xfail(raises=NotImplementedError, strict=False)
    def test_trainable(self, name: str, device: str) -> None:
        torch.manual_seed(0)
        self.get_domain(name)
        model, x, y = self.make_test_case(device=device)
        layer = self.get_parametrized_layer(model)
        register_parametrization(layer, "bias", self.get_parametrization(name))
        optimizer = SGD(model.parameters(), lr=0.1)
        register_optimizer_hook(optimizer, model)
        parametrization = self.get_bias_parametrization(layer)
        self.assert_stale(parametrization, False)
        original_parameter = parametrization.original_parameter.detach().clone()
        original_output = model(x).detach().clone()

        for _ in range(self.NUM_ITERATIONS):
            model.zero_grad(set_to_none=True)
            loss = mse_loss(model(x), y)
            loss.backward()
            self.assert_stale(parametrization, True)
            optimizer.step()
            self.assert_stale(parametrization, False)
            self.check_parametrization(name, model)

        assert not torch.allclose(
            parametrization.original_parameter, original_parameter
        )
        assert not torch.allclose(model(x), original_output)

    @pytest_xfail(raises=NotImplementedError, strict=False)
    def test_compile_forward_uses_cached_parameter(
        self, name: str, device: str
    ) -> None:
        torch.manual_seed(0)
        self.get_domain(name)
        model, x, _ = self.make_test_case(device=device)
        layer = self.get_parametrized_layer(model)
        register_parametrization(layer, "bias", self.get_parametrization(name))

        compiled_model = torch.compile(model)
        assert isinstance(compiled_model, OptimizedModule)
        self.check_parametrization(name, compiled_model)

        y0 = compiled_model(x)
        compiled_layer = self.get_parametrized_layer(compiled_model)
        parametrization = self.get_bias_parametrization(compiled_layer)
        self.assert_stale(parametrization, False)

        with torch.no_grad():
            parametrization.original_parameter.add_(
                torch.randn_like(parametrization.original_parameter)
            )

        y1 = compiled_model(x)
        self.assert_close(y1, y0, atol=self.VALUE_ATOL, rtol=self.VALUE_RTOL)
        self.check_parametrization(name, compiled_model)

    @pytest_xfail(raises=NotImplementedError, strict=False)
    def test_compile_trainable(self, name: str, device: str) -> None:
        torch.manual_seed(0)
        self.get_domain(name)
        model, x, y = self.make_test_case(device=device)
        layer = self.get_parametrized_layer(model)
        register_parametrization(layer, "bias", self.get_parametrization(name))

        compiled_model = torch.compile(model)
        assert isinstance(compiled_model, OptimizedModule)

        optimizer = SGD(compiled_model.parameters(), lr=0.1)
        compiled_layer = self.get_parametrized_layer(compiled_model)
        parametrization = self.get_bias_parametrization(compiled_layer)
        self.assert_stale(parametrization, False)
        original_parameter = parametrization.original_parameter.detach().clone()

        for _ in range(self.NUM_ITERATIONS):
            compiled_model.zero_grad(set_to_none=True)
            loss = mse_loss(compiled_model(x), y)
            loss.backward()
            self.assert_stale(parametrization, True)
            optimizer.step()
            update_parametrizations(compiled_model)
            self.assert_stale(parametrization, False)
            self.check_parametrization(name, compiled_model)

        assert not torch.allclose(
            parametrization.original_parameter, original_parameter
        )

    @pytest.mark.xfail
    def test_exported_trainable(self, name: str, device: str) -> None:
        torch.manual_seed(0)
        self.get_domain(name)
        model, x, y = self.make_test_case(device=device)
        layer = self.get_parametrized_layer(model)
        register_parametrization(layer, "bias", self.get_parametrization(name))
        parametrization = self.get_bias_parametrization(layer)
        self.assert_stale(parametrization, False)

        exported_model = torch.export.export(model, args=(x,)).module()
        optimizer = SGD(exported_model.parameters(), lr=0.1)
        exported_layer = self.get_parametrized_layer(exported_model)
        parametrization = self.get_bias_parametrization(exported_layer)
        self.assert_stale(parametrization, False)
        original_parameter = parametrization.original_parameter.detach().clone()

        for _ in range(self.NUM_ITERATIONS):
            exported_model.zero_grad(set_to_none=True)
            loss = mse_loss(exported_model(x), y)
            loss.backward()
            self.assert_stale(parametrization, True)
            optimizer.step()
            update_parametrizations(exported_model)
            self.assert_stale(parametrization, False)
            self.check_parametrization(name, exported_model)

        assert not torch.allclose(
            parametrization.original_parameter, original_parameter
        )
