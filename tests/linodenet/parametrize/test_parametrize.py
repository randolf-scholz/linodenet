import torch
from torch import Tensor, nn
from torch.nn.functional import mse_loss
from torch.optim import SGD

from linodenet.nn.parametrize import (
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
