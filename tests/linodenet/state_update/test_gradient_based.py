r"""Tests for gradient-based state updates."""

import tempfile

import pytest
import torch
from torch import Tensor, nn

from linodenet.state_update.gradient_based import (
    GradientStepUpdater,
    LpLoss,
)


class ScaleDecoder(nn.Module):
    r"""Simple scalar decoder with an analytic gradient."""

    def __init__(self, weight: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(weight))

    def forward(self, z: Tensor, /) -> Tensor:
        return self.weight * z


class TestLpLoss:
    def test_mean_returns_per_batch_element_loss(self) -> None:
        r"""Mean aggregation should preserve the batch shape."""
        loss = LpLoss(p=2.0, dim=-1, aggregation="mean")
        x = torch.tensor([[1.0, -2.0, 0.5], [0.0, 3.0, -1.0]])
        y = torch.tensor([[0.0, 1.0, -1.5], [2.0, -1.0, 0.5]])

        actual = loss(x, y)
        expected = (x - y).abs().pow(2).mean(dim=-1)

        torch.testing.assert_close(actual, expected)

    def test_sum_matches_vector_norm_power(self) -> None:
        r"""Sum aggregation should match the powered vector norm."""
        loss = LpLoss(p=2.0, dim=-1, aggregation="sum")
        x = torch.tensor([[1.0, -2.0, 0.5], [0.0, 3.0, -1.0]])
        y = torch.tensor([[0.0, 1.0, -1.5], [2.0, -1.0, 0.5]])

        actual = loss(x, y)
        expected = torch.linalg.vector_norm(x - y, ord=2.0, dim=-1).pow(2)

        torch.testing.assert_close(actual, expected)


class TestGradientStepUpdater:
    def test_forward_and_backward(self) -> None:
        r"""Forward output and parameter gradients should match the closed form."""
        decoder = ScaleDecoder(weight=1.7)
        updater = GradientStepUpdater(
            decoder=decoder,
            loss="l2",
            step_size=0.2,
        )
        z_prev = torch.tensor([1.0, 2.0, -0.5])
        y = torch.tensor([0.5, -1.0, 1.5])

        actual = updater(y, z_prev)

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

    def test_grad_fn_returns_per_batch_gradient(self) -> None:
        r"""The gradient helper should return one gradient per batch element."""
        decoder = ScaleDecoder(weight=1.7)
        updater = GradientStepUpdater(
            decoder=decoder,
            loss="l2",
            step_size=0.2,
        )
        z = torch.tensor([[1.0, 2.0, -0.5], [0.5, -1.5, 2.0]])
        y = torch.tensor([[0.5, -1.0, 1.5], [-0.25, 0.75, 1.0]])

        actual = updater.grad_fn(y, z)

        d = z.shape[-1]
        weight = decoder.weight.detach()
        expected = (2 * weight / d) * (weight * z - y)

        assert actual.shape == z.shape
        torch.testing.assert_close(actual, expected)

    def test_compile_fullgraph(self) -> None:
        r"""The updater should compile under `torch.compile(fullgraph=True)`."""
        updater = GradientStepUpdater(
            decoder=ScaleDecoder(weight=1.3),
            loss="l2",
            step_size=0.15,
        )
        compiled = torch.compile(updater, fullgraph=True)

        z_prev = torch.tensor([0.25, -0.75, 1.5])
        y = torch.tensor([1.0, -0.5, 0.75])

        expected = updater(y, z_prev)
        actual = compiled(y, z_prev)

        torch.testing.assert_close(actual, expected)

    @pytest.mark.xfail(reason="torch.export limitation / bug")
    def test_export_save_load_roundtrip(self) -> None:
        r"""The updater should round-trip through `torch.export.save`."""
        updater = GradientStepUpdater(
            decoder=ScaleDecoder(weight=1.3),
            loss="l2",
            step_size=0.15,
        )
        compiled = torch.compile(updater, fullgraph=True)

        z_prev = torch.tensor([0.25, -0.75, 1.5])
        y = torch.tensor([1.0, -0.5, 0.75])

        try:
            exported = torch.export.export(updater, (y, z_prev))
        except (AssertionError, RuntimeError) as err:
            pytest.xfail(
                "torch.export.export currently does not support this updater "
                f"with torch.func.vjp: {err}"
            )

        with tempfile.NamedTemporaryFile(suffix=".pt2") as f:
            torch.export.save(exported, f.name)
            loaded = torch.export.load(f.name)

        expected = updater(y, z_prev)
        actual_compiled = compiled(y, z_prev)
        actual_loaded = loaded.module()(y, z_prev)

        torch.testing.assert_close(actual_compiled, expected)
        torch.testing.assert_close(actual_loaded, expected)

    def test_gradients_wrt_z_prev(self) -> None:
        r"""The update should remain differentiable with respect to the previous state."""
        decoder = ScaleDecoder(weight=1.5)
        updater = GradientStepUpdater(
            decoder=decoder,
            loss="l2",
            step_size=0.3,
        )
        z_prev = torch.tensor([1.0, -2.0, 0.5], requires_grad=True)
        y = torch.tensor([0.0, 1.0, -1.5])

        objective = updater(y, z_prev).sum()
        (actual_grad,) = torch.autograd.grad(objective, z_prev)

        n = z_prev.numel()
        weight = decoder.weight.detach()
        step_size = updater.step_size.detach()
        expected_grad = torch.full_like(
            z_prev.detach(),
            fill_value=float(1 - step_size * (2 * weight.square() / n)),
        )

        torch.testing.assert_close(actual_grad, expected_grad)

    def test_is_consistent_for_exact_decoder_outputs(self) -> None:
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
            step_size=0.7,
        )
        z_prev = torch.tensor([0.75, -1.25, 0.5, 2.0])
        y = decoder(z_prev)

        actual = updater(y, z_prev)

        torch.testing.assert_close(actual, z_prev)
