import torch
from torch import Tensor, nn
from torch.nn.functional import mse_loss
from torch.optim import SGD

from linodenet.nn import ModuleMapping
from linodenet.nn.parametrize import (
    ParametrizationList,
    cached,
    is_parametrized,
    register_parametrization,
    update_parametrizations,
)


class UpperTriangular(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.triu()

    def right_inverse(self, y: Tensor) -> Tensor:
        return y


class ScaleByTwo(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return 2 * x

    def right_inverse(self, y: Tensor) -> Tensor:
        return y / 2


class ShiftByOne(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return x + 1

    def right_inverse(self, y: Tensor) -> Tensor:
        return y - 1


def is_upper_triangular(x: Tensor) -> bool:
    return torch.equal(x, x.triu())


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


def test_register_parametrization_and_is_parametrized() -> None:
    model = nn.Linear(in_features=5, out_features=4, bias=False)

    assert not is_parametrized(model)
    assert not is_parametrized(model, "weight")
    assert not is_upper_triangular(model.weight)

    register_parametrization(model, "weight", UpperTriangular())

    assert is_parametrized(model)
    assert is_parametrized(model, "weight")
    assert not is_parametrized(model, "bias")
    assert is_upper_triangular(model.weight)


def test_cached_refreshes_parametrizations_on_exit() -> None:
    model = nn.Linear(in_features=5, out_features=4, bias=False)
    register_parametrization(model, "weight", UpperTriangular())

    optimizer = SGD(model.parameters(), lr=0.1)
    inputs = torch.randn(3, 5)
    target = torch.randn(3, 4)

    update_parametrizations(model)
    weight_before = model.weight.detach().clone()

    with cached(model):
        model.zero_grad(set_to_none=True)
        loss = mse_loss(model(inputs), target)
        loss.backward()
        optimizer.step()

        weight_during_context = model.weight.detach().clone()
        assert weight_during_context.isnan().all()

    assert is_upper_triangular(model.weight)
    assert not torch.equal(model.weight, weight_before)
    assert not torch.equal(model.weight, weight_during_context)


def test_parametrized_model_training_requires_explicit_refresh() -> None:
    model = nn.Linear(in_features=5, out_features=4, bias=False)
    register_parametrization(model, "weight", UpperTriangular())

    assert is_upper_triangular(model.weight)
    assert_update_refreshes_parametrized_weight(model)


def test_update_parametrizations_heals_connections_after_to() -> None:
    model = nn.Linear(in_features=5, out_features=4, bias=False)
    register_parametrization(model, "weight", UpperTriangular())

    assert isinstance(model.parametrizations, ModuleMapping)
    parametrization = model.parametrizations["weight"]
    assert isinstance(parametrization, nn.Module)
    assert isinstance(model.weight, Tensor)
    assert model.weight is parametrization.cached_parameter

    model = model.to(dtype=torch.float64)
    assert isinstance(model.parametrizations, ModuleMapping)
    parametrization = model.parametrizations["weight"]
    assert isinstance(parametrization, nn.Module)
    assert isinstance(model.weight, Tensor)
    assert model.weight is not parametrization.cached_parameter

    update_parametrizations(model)

    assert model.weight is parametrization.cached_parameter
    assert model.weight.dtype == torch.float64
    assert is_upper_triangular(model.weight)


def test_register_parametrization_new_installs_parametrization_list() -> None:
    model = nn.Linear(in_features=4, out_features=4, bias=False)

    register_parametrization(model, "weight", ScaleByTwo())

    parametrization = model.parametrizations["weight"]
    assert isinstance(parametrization, ParametrizationList)
    assert len(parametrization) == 1
    assert torch.allclose(model.weight, 2 * parametrization.original_parameter)


def test_register_parametrization_new_appends_on_second_registration() -> None:
    model = nn.Linear(in_features=4, out_features=4, bias=False)

    register_parametrization(model, "weight", ScaleByTwo())
    register_parametrization(model, "weight", ShiftByOne())

    parametrization = model.parametrizations["weight"]
    assert isinstance(parametrization, ParametrizationList)
    assert len(parametrization) == 2
    assert is_parametrized(model, "weight")
    assert torch.allclose(model.weight, 2 * parametrization.original_parameter + 1)
