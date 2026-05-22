import tempfile

import pytest
import torch
from torch import Tensor, jit, nn
from torch._dynamo import OptimizedModule
from torch.nn.functional import mse_loss
from torch.optim import SGD

from linodenet.nn.parametrize import register_parametrization, update_parametrizations


class UpperTriangular(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.triu()

    def right_inverse(self, y: Tensor) -> Tensor:
        return y


def is_upper_triangular(x: Tensor) -> bool:
    return torch.equal(x, x.triu())


def make_parametrized_model() -> nn.Linear:
    model = nn.Linear(in_features=5, out_features=4, bias=False)
    register_parametrization(model, "weight", UpperTriangular())
    return model


def assert_update_refreshes_parametrized_weight(model: nn.Module) -> None:
    # model.train()
    inputs = torch.randn(3, 5)
    target = torch.randn(3, 4)
    optimizer = SGD(model.parameters(), lr=0.1)

    update_parametrizations(model)
    assert isinstance(model.weight, Tensor)
    weight_before = model.weight.detach().clone()
    outputs_before = model(inputs).detach().clone()
    params_before = [parameter.detach().clone() for parameter in model.parameters()]

    model.zero_grad(set_to_none=True)
    loss = mse_loss(model(inputs), target)
    loss.backward()
    optimizer.step()

    weight_after_step = model.weight.detach().clone()
    assert (
        torch.equal(weight_after_step, weight_before) or weight_after_step.isnan().all()
    )

    update_parametrizations(model)

    assert is_upper_triangular(model.weight)
    assert not torch.equal(model.weight, weight_before)
    assert not torch.allclose(model(inputs), outputs_before)
    assert any(
        not torch.equal(parameter.detach(), before)
        for parameter, before in zip(model.parameters(), params_before, strict=True)
    )


def test_register_parametrization_uses_upper_triangular_cache() -> None:
    model = nn.Linear(in_features=5, out_features=4, bias=False)

    assert not is_upper_triangular(model.weight)
    register_parametrization(model, "weight", UpperTriangular())
    assert is_upper_triangular(model.weight)


class TestTorchScript:
    @staticmethod
    def deserialize_module(module: nn.Module | jit.ScriptModule) -> jit.ScriptModule:
        scripted = jit.script(module)

        with tempfile.TemporaryFile() as file:
            jit.save(scripted, file)
            file.seek(0)
            return jit.load(file)

    def test_training_freshly_scripted_model(self) -> None:
        model = make_parametrized_model()
        scripted = jit.script(model)

        assert is_upper_triangular(scripted.weight)
        assert_update_refreshes_parametrized_weight(scripted)

    def test_training_deserialized_scripted_model(self) -> None:
        model = make_parametrized_model()
        scripted = self.deserialize_module(model)

        assert is_upper_triangular(scripted.weight)
        assert_update_refreshes_parametrized_weight(scripted)


class TestTorchCompile:
    def test_training_freshly_compiled_model(self) -> None:
        model = make_parametrized_model()
        compiled = torch.compile(model)

        assert isinstance(compiled, OptimizedModule)
        assert is_upper_triangular(compiled.weight)
        assert_update_refreshes_parametrized_weight(compiled)


class TestTorchExport:
    @staticmethod
    def export_module(module: nn.Module) -> nn.Module:
        inputs = torch.randn(3, 5)
        return torch.export.export(module, args=(inputs,)).module()

    @staticmethod
    def deserialize_module(module: nn.Module) -> nn.Module:
        exported_program = torch.export.export(module, args=(torch.randn(3, 5),))

        with tempfile.TemporaryFile() as file:
            torch.export.save(exported_program, file)
            file.seek(0)
            return torch.export.load(file).module()

    @pytest.mark.xfail(
        raises=NotImplementedError,
        reason="torch.export does not support exporting entrypoints (https://github.com/pytorch/pytorch/issues/167631)",
        strict=True,
    )
    def test_training_freshly_exported_model(self) -> None:
        model = make_parametrized_model()
        exported = self.export_module(model)

        assert isinstance(exported.weight, Tensor)
        assert is_upper_triangular(exported.weight)
        assert_update_refreshes_parametrized_weight(exported)

    @pytest.mark.xfail(
        raises=NotImplementedError,
        reason="torch.export does not support exporting entrypoints (https://github.com/pytorch/pytorch/issues/167631)",
        strict=True,
    )
    def test_training_deserialized_exported_model(self) -> None:
        model = make_parametrized_model()
        exported = self.deserialize_module(model)

        assert isinstance(exported.weight, Tensor)
        assert is_upper_triangular(exported.weight)
        assert_update_refreshes_parametrized_weight(exported)
