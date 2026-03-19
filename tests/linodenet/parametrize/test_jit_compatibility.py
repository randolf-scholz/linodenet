import tempfile

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


def deserialize_module(module: nn.Module | jit.ScriptModule) -> jit.ScriptModule:
    scripted = jit.script(module)

    with tempfile.TemporaryFile() as file:
        jit.save(scripted, file)
        file.seek(0)
        return jit.load(file)


def assert_update_refreshes_parametrized_weight(model: nn.Module) -> None:
    model.train()
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


def test_scripted_model_refreshes_parametrized_weight_during_training() -> None:
    model = nn.Linear(in_features=5, out_features=4, bias=False)
    register_parametrization(model, "weight", UpperTriangular())
    compiled = jit.script(model)
    assert is_upper_triangular(compiled.weight)
    assert_update_refreshes_parametrized_weight(compiled)


def test_deserialized_model_refreshes_parametrized_weight_during_training() -> None:
    model = nn.Linear(in_features=5, out_features=4, bias=False)
    register_parametrization(model, "weight", UpperTriangular())
    compiled = deserialize_module(model)
    assert is_upper_triangular(compiled.weight)
    assert_update_refreshes_parametrized_weight(compiled)


def test_compile_compatibility() -> None:
    model = nn.Linear(in_features=5, out_features=4, bias=False)
    register_parametrization(model, "weight", UpperTriangular())

    compiled = torch.compile(model)

    assert isinstance(compiled, OptimizedModule)
    assert is_upper_triangular(compiled.weight)
    assert_update_refreshes_parametrized_weight(compiled)
