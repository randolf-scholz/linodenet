r"""Tests for gradient-based state updates."""

import torch
from torch import Tensor, nn

from linodenet.state_update.gradient_based import GradientStepUpdater


class ScaleDecoder(nn.Module):
    r"""Simple scalar decoder with an analytic gradient."""

    def __init__(self, weight: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(weight))

    def forward(self, z: Tensor, /) -> Tensor:
        return self.weight * z


def test_gradient_step_updater_forward_and_backward() -> None:
    r"""Forward output and parameter gradients should match the closed form."""
    decoder = ScaleDecoder(weight=1.7)
    updater = GradientStepUpdater(
        decoder=decoder,
        loss="l2",
        regularizer="l2",
        regularization_strength=0.0,
        step_size=0.2,
    )
    z_prev = torch.tensor([1.0, 2.0, -0.5])
    y = torch.tensor([0.5, -1.0, 1.5])

    actual = updater(z_prev, y)

    n = z_prev.numel()
    weight = decoder.weight.detach()
    step_size = updater.step_size.detach()
    expected_grad = (2 * weight / n) * (weight * z_prev - y)
    expected = z_prev - step_size * expected_grad

    torch.testing.assert_close(actual, expected)

    objective = actual.sum()
    actual_grad_weight, actual_grad_step_size = torch.autograd.grad(
        objective,
        (decoder.weight, updater.step_size),
    )
    expected_grad_weight = -step_size * (2 / n) * (2 * weight * z_prev - y).sum()
    expected_grad_step_size = -expected_grad.sum()

    torch.testing.assert_close(actual_grad_weight, expected_grad_weight)
    torch.testing.assert_close(actual_grad_step_size, expected_grad_step_size)


def test_gradient_step_updater_compile_fullgraph() -> None:
    r"""The updater should compile under `torch.compile(fullgraph=True)`."""
    updater = GradientStepUpdater(
        decoder=ScaleDecoder(weight=1.3),
        loss="l2",
        regularizer="l2",
        regularization_strength=0.0,
        step_size=0.15,
    )
    compiled = torch.compile(updater, fullgraph=True)

    z_prev = torch.tensor([0.25, -0.75, 1.5])
    y = torch.tensor([1.0, -0.5, 0.75])

    expected = updater(z_prev, y)
    actual = compiled(z_prev, y)

    torch.testing.assert_close(actual, expected)


def test_gradient_step_updater_gradients_wrt_z_prev() -> None:
    r"""The update should remain differentiable with respect to the previous state."""
    decoder = ScaleDecoder(weight=1.5)
    updater = GradientStepUpdater(
        decoder=decoder,
        loss="l2",
        regularizer="l2",
        regularization_strength=0.0,
        step_size=0.3,
    )
    z_prev = torch.tensor([1.0, -2.0, 0.5], requires_grad=True)
    y = torch.tensor([0.0, 1.0, -1.5])

    objective = updater(z_prev, y).sum()
    (actual_grad,) = torch.autograd.grad(objective, z_prev)

    n = z_prev.numel()
    weight = decoder.weight.detach()
    step_size = updater.step_size.detach()
    expected_grad = torch.full_like(
        z_prev.detach(),
        fill_value=1 - step_size * (2 * weight.square() / n),
    )

    torch.testing.assert_close(actual_grad, expected_grad)


def test_gradient_step_updater_is_consistent_for_exact_decoder_outputs() -> None:
    r"""For $y = decoder(z₋)$ and L2 losses, the update should be idempotent."""
    decoder = nn.Linear(4, 4, bias=True)
    with torch.no_grad():
        decoder.weight.copy_(
            torch.tensor(
                [
                    [1.0, 0.5, 0.0, -0.5],
                    [0.0, 1.5, -0.5, 0.0],
                    [0.25, 0.0, 0.75, 0.5],
                    [-0.5, 0.25, 0.0, 1.25],
                ]
            )
        )
        decoder.bias.copy_(torch.tensor([0.25, -0.5, 1.0, -1.5]))

    updater = GradientStepUpdater(
        decoder=decoder,
        loss="l2",
        regularizer="l2",
        regularization_strength=2.0,
        step_size=0.7,
    )
    z_prev = torch.tensor([0.75, -1.25, 0.5, 2.0])
    y = decoder(z_prev)

    actual = updater(z_prev, y)

    torch.testing.assert_close(actual, z_prev)
