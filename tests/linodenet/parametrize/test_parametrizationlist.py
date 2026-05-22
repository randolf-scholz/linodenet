import torch
from torch import Tensor, nn
from torch.nn.functional import mse_loss
from torch.optim import SGD

from linodenet.domains.matrix_tests import is_orthogonal
from linodenet.mappings.bijections import CayleyMap
from linodenet.mappings.projections import SkewSymmetric
from linodenet.nn.parametrize import (
    ParametrizationList,
    get_parametrizations,
    update_parametrizations,
)
from linodenet.nn.rezero import ReZero


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


def register_parametrization_list(
    module: nn.Module,
    tensor_name: str,
    *parametrizations: nn.Module,
) -> ParametrizationList:
    tensor = getattr(module, tensor_name)
    assert isinstance(tensor, nn.Parameter)

    wrapper = ParametrizationList([], original=tensor)
    for parametrization in parametrizations:
        wrapper.append(parametrization)

    match getattr(module, "parametrizations", None):
        case None:
            module.register_module(
                "parametrizations",
                nn.ModuleDict({tensor_name: wrapper}),
            )
        case nn.ModuleDict() as parametrization_dict:
            parametrization_dict[tensor_name] = wrapper
        case value:
            raise TypeError(f"Expected a nn.ModuleDict, but got {type(value)}!")

    match getattr(module, "parametrized_tensors", None):
        case None:
            module.register_module(
                "parametrized_tensors",
                nn.ParameterDict({tensor_name: wrapper.original_parameter}),
            )
        case nn.ParameterDict() as parametrized_tensors:
            parametrized_tensors[tensor_name] = wrapper.original_parameter
        case value:
            raise TypeError(f"Expected a nn.ParameterDict, but got {type(value)}!")

    delattr(module, tensor_name)
    module.register_buffer(tensor_name, wrapper.get_cached_tensor())
    update_parametrizations(module)
    return wrapper


def run_training_step(model: nn.Module, /) -> tuple[Tensor, Tensor]:
    inputs = torch.randn(8, 4)
    target = torch.randn(8, 4)
    optimizer = SGD(model.parameters(), lr=0.1)

    output_before = model(inputs).detach().clone()
    loss_before = mse_loss(output_before, target)

    model.zero_grad(set_to_none=True)
    loss = mse_loss(model(inputs), target)
    loss.backward()
    optimizer.step()
    update_parametrizations(model)

    output_after = model(inputs).detach().clone()
    loss_after = mse_loss(output_after, target)

    assert torch.isfinite(loss)
    assert torch.isfinite(loss_after)
    assert not torch.allclose(output_after, output_before)

    return loss_before, loss_after


def test_parametrization_list_skew_symmetric_then_cayley_supports_training_step() -> (
    None
):
    torch.manual_seed(0)
    model = nn.Linear(4, 4, bias=False)

    parametrization = register_parametrization_list(
        model,
        "weight",
        SkewSymmetric(),
        CayleyMap(),
    )

    parametrizations = get_parametrizations(model)
    assert parametrizations is not None
    assert parametrizations["weight"] is parametrization
    assert len(parametrization) == 2
    assert is_orthogonal(model.weight, size=4).all()

    weight_before = model.weight.detach().clone()
    run_training_step(model)

    assert is_orthogonal(model.weight, size=4).all()
    assert not torch.allclose(model.weight, weight_before)


def test_parametrization_list_skew_symmetric_cayley_and_rezero_supports_training_step() -> (
    None
):
    torch.manual_seed(0)
    model = nn.Linear(4, 4, bias=False)

    parametrization = register_parametrization_list(
        model,
        "weight",
        SkewSymmetric(),
        CayleyMap(),
        ReZero(),
    )

    parametrizations = get_parametrizations(model)
    assert parametrizations is not None
    assert parametrizations["weight"] is parametrization
    assert len(parametrization) == 3
    assert torch.allclose(model.weight, torch.zeros_like(model.weight))

    rezero = parametrization[-1].transform
    assert isinstance(rezero, ReZero)
    scalar_before = rezero.scalar.detach().clone()

    run_training_step(model)

    assert not torch.allclose(rezero.scalar, scalar_before)
    assert not torch.allclose(model.weight, torch.zeros_like(model.weight))


def test_parametrization_list_right_inverse_inverts_forward_in_reverse_order() -> None:
    tensor = nn.Parameter(torch.zeros(2, 2))
    parametrization = ParametrizationList([], original=tensor)
    parametrization.append(ScaleByTwo())
    parametrization.append(ShiftByOne())

    original = torch.tensor([[1.0, -2.0], [0.5, 3.0]])
    transformed = parametrization(original)
    recovered = parametrization.right_inverse(transformed)

    assert recovered is not None
    assert torch.allclose(transformed, 2 * original + 1)
    assert torch.allclose(recovered, original)
