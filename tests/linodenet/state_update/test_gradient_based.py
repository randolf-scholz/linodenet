r"""Tests for gradient-based state updates."""

import tempfile

import pytest
import torch
from torch import Tensor, nn
from torch.distributions import MultivariateNormal

from linodenet.distributions.gaussian import argmin_proximal_kl, argmin_reverse_kl
from linodenet.state_update.gradient_based import (
    GaussianKLProximalUpdater,
    GaussianReverseKLUpdater,
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


class ShiftTransform(nn.Module):
    r"""Simple scalar translation transform with an analytic log-Jacobian."""

    def __init__(self, shift: float) -> None:
        super().__init__()
        self.shift = nn.Parameter(torch.tensor(shift))

    def forward(self, z: Tensor, /) -> Tensor:
        return z + self.shift

    def inverse(self, y: Tensor, /) -> Tensor:
        return y - self.shift

    def encode_and_logabsdet(self, y: Tensor, /) -> tuple[Tensor, Tensor]:
        return self.inverse(y), y.new_zeros(y.shape[:-1])

    def decode_and_logabsdet(self, z: Tensor, /) -> tuple[Tensor, Tensor]:
        return self.forward(z), z.new_zeros(z.shape[:-1])


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


def _scale_tril(log_chol: Tensor, /) -> Tensor:
    r"""Convert log-Cholesky parameters to a lower-triangular scale matrix."""
    return log_chol.tril(diagonal=-1) + torch.diag_embed(
        log_chol.diagonal(dim1=-2, dim2=-1).exp()
    )


def _reference_gaussian_step(
    mean: Tensor,
    log_chol: Tensor,
    y_obs: Tensor,
    /,
    *,
    shift: Tensor,
    regularization_strength: Tensor,
) -> tuple[Tensor, Tensor]:
    r"""Compute the Gaussian KL-proximal update using PyTorch distributions."""

    def loss_fn(theta: tuple[Tensor, Tensor], /) -> Tensor:
        current = MultivariateNormal(theta[0], scale_tril=_scale_tril(theta[1]))
        return -current.log_prob(y_obs - shift)

    return argmin_proximal_kl(
        loss_fn,
        (mean, log_chol),
        gamma=regularization_strength,
        parametrization="log-cholesky",
    )


def _reference_gaussian_reverse_kl_step(
    mean: Tensor,
    log_chol: Tensor,
    y_obs: Tensor,
    /,
    *,
    shift: Tensor,
    regularization_strength: Tensor,
) -> tuple[Tensor, Tensor]:
    r"""Compute the exact Gaussian reverse-KL update in latent coordinates."""
    return argmin_reverse_kl(
        y_obs - shift,
        (mean, log_chol),
        gamma=regularization_strength,
        parametrization="log-cholesky",
    )


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


class TestGaussianGradientStepUpdater:
    def test_regularization_strength_must_be_positive(self) -> None:
        r"""The KL regularization strength should be validated at construction."""
        with pytest.raises(ValueError, match="regularization_strength"):
            GaussianKLProximalUpdater(
                decoder=ShiftTransform(shift=0.0),
                parametrization="log-cholesky",
                regularization_strength=0.0,
            )

    def test_regularization_learnable_can_be_disabled(self) -> None:
        r"""The log-regularization parameter should support frozen initialization."""
        updater = GaussianKLProximalUpdater(
            decoder=ShiftTransform(shift=0.0),
            parametrization="log-cholesky",
            regularization_strength=1.7,
            regularization_learnable=False,
        )

        assert not updater.log_regularization_strength.requires_grad
        torch.testing.assert_close(updater.regularization_strength, torch.tensor(1.7))

    def test_forward_and_backward(self) -> None:
        r"""Forward output and parameter gradients should match the closed form."""
        decoder = ShiftTransform(shift=0.2)
        updater = GaussianKLProximalUpdater(
            decoder=decoder,
            parametrization="log-cholesky",
            regularization_strength=1.7,
        )
        mean = torch.tensor([0.4, -0.3, 0.8])
        log_chol = torch.tensor(
            [
                [0.1, 0.0, 0.0],
                [0.2, -0.2, 0.0],
                [-0.1, 0.3, 0.4],
            ]
        )
        y_obs = torch.tensor([1.1, -0.7, 0.5])

        actual_mean, actual_log_chol = updater(y_obs, (mean, log_chol))

        mean_ref = mean.detach().clone().requires_grad_()
        log_chol_ref = log_chol.detach().clone().requires_grad_()
        expected_mean, expected_log_chol = _reference_gaussian_step(
            mean_ref,
            log_chol_ref,
            y_obs,
            shift=decoder.shift,
            regularization_strength=updater.regularization_strength,
        )

        torch.testing.assert_close(actual_mean, expected_mean)
        torch.testing.assert_close(actual_log_chol, expected_log_chol)

        objective = actual_mean.sum() + actual_log_chol.sum()
        actual_grad_shift, actual_grad_log_gamma = torch.autograd.grad(
            objective,
            (decoder.shift, updater.log_regularization_strength),
        )
        expected_objective = expected_mean.sum() + expected_log_chol.sum()
        expected_grad_shift, expected_grad_log_gamma = torch.autograd.grad(
            expected_objective,
            (decoder.shift, updater.log_regularization_strength),
        )

        torch.testing.assert_close(actual_grad_shift, expected_grad_shift)
        torch.testing.assert_close(actual_grad_log_gamma, expected_grad_log_gamma)

    @pytest.mark.xfail(reason="torch.compile limitation / bug in solve_proximal_kl")
    def test_compile_fullgraph(self) -> None:
        r"""The updater should compile under `torch.compile(fullgraph=True)`."""
        updater = GaussianKLProximalUpdater(
            decoder=ShiftTransform(shift=-0.1),
            parametrization="log-cholesky",
            regularization_strength=1.2,
        )
        compiled = torch.compile(updater, fullgraph=True)

        mean = torch.tensor([-0.3, 0.6, 0.1])
        log_chol = torch.tensor(
            [
                [0.25, 0.0, 0.0],
                [-0.2, 0.1, 0.0],
                [0.15, -0.05, -0.3],
            ]
        )
        y_obs = torch.tensor([0.7, -0.4, 0.5])

        expected = updater(y_obs, (mean, log_chol))
        actual = compiled(y_obs, (mean, log_chol))

        torch.testing.assert_close(actual[0], expected[0])
        torch.testing.assert_close(actual[1], expected[1])

    def test_batched_forward(self) -> None:
        r"""Batched parameters should use a per-sample objective."""
        decoder = ShiftTransform(shift=0.1)
        updater = GaussianKLProximalUpdater(
            decoder=decoder,
            parametrization="log-cholesky",
            regularization_strength=1.0,
        )
        mean = torch.tensor([[0.2, -0.4, 0.6], [-0.4, 0.3, 0.1]])
        log_chol = torch.tensor(
            [
                [
                    [0.0, 0.0, 0.0],
                    [0.1, 0.2, 0.0],
                    [-0.2, 0.3, -0.1],
                ],
                [
                    [0.3, 0.0, 0.0],
                    [-0.1, -0.2, 0.0],
                    [0.2, 0.1, 0.4],
                ],
            ]
        )
        y_obs = torch.tensor([[0.7, -1.2, 0.4], [-1.2, 0.5, 0.8]])

        actual_mean, actual_log_chol = updater(y_obs, (mean, log_chol))

        mean_ref = mean.detach().clone().requires_grad_()
        log_chol_ref = log_chol.detach().clone().requires_grad_()
        expected_mean, expected_log_chol = _reference_gaussian_step(
            mean_ref,
            log_chol_ref,
            y_obs,
            shift=decoder.shift,
            regularization_strength=updater.regularization_strength,
        )

        torch.testing.assert_close(actual_mean, expected_mean)
        torch.testing.assert_close(
            actual_log_chol, expected_log_chol, atol=1e-4, rtol=1e-4
        )

    def test_grad_fn_returns_per_batch_gradient(self) -> None:
        r"""The Gaussian gradient helper should preserve the batch shape."""
        decoder = ShiftTransform(shift=0.1)
        updater = GaussianKLProximalUpdater(
            decoder=decoder,
            parametrization="log-cholesky",
            regularization_strength=1.0,
        )
        mean = torch.tensor([[0.2, -0.4, 0.6], [-0.4, 0.3, 0.1]])
        log_chol = torch.tensor(
            [
                [
                    [0.0, 0.0, 0.0],
                    [0.1, 0.2, 0.0],
                    [-0.2, 0.3, -0.1],
                ],
                [
                    [0.3, 0.0, 0.0],
                    [-0.1, -0.2, 0.0],
                    [0.2, 0.1, 0.4],
                ],
            ]
        )
        y_obs = torch.tensor([[0.7, -1.2, 0.4], [-1.2, 0.5, 0.8]])

        actual_grad_mean, actual_grad_log_chol = updater.grad_fn(
            y_obs, (mean, log_chol)
        )

        mean_ref = mean.detach().clone().requires_grad_()
        log_chol_ref = log_chol.detach().clone().requires_grad_()
        current = MultivariateNormal(mean_ref, scale_tril=_scale_tril(log_chol_ref))
        objective = -current.log_prob(y_obs - decoder.shift)
        expected_grad_mean, expected_grad_log_chol = torch.autograd.grad(
            objective.sum(),
            (mean_ref, log_chol_ref),
        )

        assert actual_grad_mean.shape == mean.shape
        assert actual_grad_log_chol.shape == log_chol.shape
        torch.testing.assert_close(actual_grad_mean, expected_grad_mean)
        torch.testing.assert_close(actual_grad_log_chol, expected_grad_log_chol)

    def test_gradients_wrt_theta(self) -> None:
        r"""The update should remain differentiable with respect to the prior."""
        decoder = ShiftTransform(shift=0.25)
        updater = GaussianKLProximalUpdater(
            decoder=decoder,
            parametrization="log-cholesky",
            regularization_strength=1.4,
        )
        mean = torch.tensor([0.1, -0.2, 0.5], requires_grad=True)
        log_chol = torch.tensor(
            [
                [-0.2, 0.0, 0.0],
                [0.1, 0.3, 0.0],
                [-0.3, 0.2, 0.1],
            ],
            requires_grad=True,
        )
        y_obs = torch.tensor([0.9, -0.6, 0.2])

        mean_post, log_chol_post = updater(y_obs, (mean, log_chol))
        objective = mean_post.sum() + log_chol_post.sum()
        actual_grad_mean, actual_grad_log_chol = torch.autograd.grad(
            objective,
            (mean, log_chol),
        )

        expected_mean, expected_log_chol = _reference_gaussian_step(
            mean,
            log_chol,
            y_obs,
            shift=decoder.shift,
            regularization_strength=updater.regularization_strength,
        )
        expected_objective = expected_mean.sum() + expected_log_chol.sum()
        expected_grad_mean, expected_grad_log_chol = torch.autograd.grad(
            expected_objective,
            (mean, log_chol),
        )

        torch.testing.assert_close(actual_grad_mean, expected_grad_mean)
        torch.testing.assert_close(actual_grad_log_chol, expected_grad_log_chol)

    def test_is_consistent_for_exact_decoder_mean(self) -> None:
        r"""For $y_obs = decoder(μ₋)$ and unit variance, the mean should not update."""
        decoder = ShiftTransform(shift=-0.4)
        updater = GaussianKLProximalUpdater(
            decoder=decoder,
            parametrization="log-cholesky",
            regularization_strength=2.0,
        )
        mean = torch.tensor([0.75, -0.25, 0.5])
        log_chol = torch.zeros(3, 3)
        y_obs = decoder(mean)

        actual_mean, actual_log_chol = updater(y_obs, (mean, log_chol))
        expected_mean, expected_log_chol = _reference_gaussian_step(
            mean,
            log_chol,
            y_obs,
            shift=decoder.shift,
            regularization_strength=updater.regularization_strength,
        )

        torch.testing.assert_close(actual_mean, mean)
        torch.testing.assert_close(actual_mean, expected_mean)
        torch.testing.assert_close(actual_log_chol, expected_log_chol)


class TestGaussianReverseKLUpdater:
    def test_regularization_strength_must_be_positive(self) -> None:
        r"""The reverse-KL regularization strength should be positive."""
        with pytest.raises(ValueError, match="regularization_strength"):
            GaussianReverseKLUpdater(
                decoder=ShiftTransform(shift=0.0),
                parametrization="log-cholesky",
                regularization_strength=0.0,
            )

    def test_regularization_learnable_can_be_disabled(self) -> None:
        r"""The log-regularization parameter should support frozen initialization."""
        updater = GaussianReverseKLUpdater(
            decoder=ShiftTransform(shift=0.0),
            parametrization="log-cholesky",
            regularization_strength=1.7,
            regularization_learnable=False,
        )

        assert not updater.log_regularization_strength.requires_grad
        torch.testing.assert_close(updater.regularization_strength, torch.tensor(1.7))

    def test_forward_and_backward(self) -> None:
        r"""Forward output and parameter gradients should match the exact update."""
        decoder = ShiftTransform(shift=0.2)
        updater = GaussianReverseKLUpdater(
            decoder=decoder,
            parametrization="log-cholesky",
            regularization_strength=1.7,
        )
        mean = torch.tensor([0.4, -0.3, 0.8])
        log_chol = torch.tensor(
            [
                [0.1, 0.0, 0.0],
                [0.2, -0.2, 0.0],
                [-0.1, 0.3, 0.4],
            ]
        )
        y_obs = torch.tensor([1.1, -0.7, 0.5])

        actual_mean, actual_log_chol = updater(y_obs, (mean, log_chol))
        expected_mean, expected_log_chol = _reference_gaussian_reverse_kl_step(
            mean,
            log_chol,
            y_obs,
            shift=decoder.shift,
            regularization_strength=updater.regularization_strength,
        )

        torch.testing.assert_close(actual_mean, expected_mean)
        torch.testing.assert_close(actual_log_chol, expected_log_chol)

        objective = actual_mean.sum() + actual_log_chol.sum()
        actual_grad_shift, actual_grad_log_gamma = torch.autograd.grad(
            objective,
            (decoder.shift, updater.log_regularization_strength),
        )
        expected_objective = expected_mean.sum() + expected_log_chol.sum()
        expected_grad_shift, expected_grad_log_gamma = torch.autograd.grad(
            expected_objective,
            (decoder.shift, updater.log_regularization_strength),
        )

        torch.testing.assert_close(actual_grad_shift, expected_grad_shift)
        torch.testing.assert_close(actual_grad_log_gamma, expected_grad_log_gamma)

    def test_compile_fullgraph(self) -> None:
        r"""The updater should compile under `torch.compile(fullgraph=True)`."""
        updater = GaussianReverseKLUpdater(
            decoder=ShiftTransform(shift=-0.1),
            parametrization="log-cholesky",
            regularization_strength=1.2,
        )
        compiled = torch.compile(updater, fullgraph=True)

        mean = torch.tensor([-0.3, 0.6, 0.1])
        log_chol = torch.tensor(
            [
                [0.25, 0.0, 0.0],
                [-0.2, 0.1, 0.0],
                [0.15, -0.05, -0.3],
            ]
        )
        y_obs = torch.tensor([0.7, -0.4, 0.5])

        expected = updater(y_obs, (mean, log_chol))
        actual = compiled(y_obs, (mean, log_chol))

        torch.testing.assert_close(actual[0], expected[0])
        torch.testing.assert_close(actual[1], expected[1])

    def test_batched_forward(self) -> None:
        r"""Batched parameters should use a per-sample exact reverse-KL update."""
        decoder = ShiftTransform(shift=0.1)
        updater = GaussianReverseKLUpdater(
            decoder=decoder,
            parametrization="log-cholesky",
            regularization_strength=1.0,
        )
        mean = torch.tensor([[0.2, -0.4, 0.6], [-0.4, 0.3, 0.1]])
        log_chol = torch.tensor(
            [
                [
                    [0.0, 0.0, 0.0],
                    [0.1, 0.2, 0.0],
                    [-0.2, 0.3, -0.1],
                ],
                [
                    [0.3, 0.0, 0.0],
                    [-0.1, -0.2, 0.0],
                    [0.2, 0.1, 0.4],
                ],
            ]
        )
        y_obs = torch.tensor([[0.7, -1.2, 0.4], [-1.2, 0.5, 0.8]])

        actual_mean, actual_log_chol = updater(y_obs, (mean, log_chol))
        expected_mean, expected_log_chol = _reference_gaussian_reverse_kl_step(
            mean,
            log_chol,
            y_obs,
            shift=decoder.shift,
            regularization_strength=updater.regularization_strength,
        )

        torch.testing.assert_close(actual_mean, expected_mean)
        torch.testing.assert_close(actual_log_chol, expected_log_chol)

    def test_gradients_wrt_theta(self) -> None:
        r"""The update should remain differentiable with respect to the prior."""
        decoder = ShiftTransform(shift=0.25)
        updater = GaussianReverseKLUpdater(
            decoder=decoder,
            parametrization="log-cholesky",
            regularization_strength=1.4,
        )
        mean = torch.tensor([0.1, -0.2, 0.5], requires_grad=True)
        log_chol = torch.tensor(
            [
                [-0.2, 0.0, 0.0],
                [0.1, 0.3, 0.0],
                [-0.3, 0.2, 0.1],
            ],
            requires_grad=True,
        )
        y_obs = torch.tensor([0.9, -0.6, 0.2])

        mean_post, log_chol_post = updater(y_obs, (mean, log_chol))
        objective = mean_post.sum() + log_chol_post.sum()
        actual_grad_mean, actual_grad_log_chol = torch.autograd.grad(
            objective,
            (mean, log_chol),
        )

        expected_mean, expected_log_chol = _reference_gaussian_reverse_kl_step(
            mean,
            log_chol,
            y_obs,
            shift=decoder.shift,
            regularization_strength=updater.regularization_strength,
        )
        expected_objective = expected_mean.sum() + expected_log_chol.sum()
        expected_grad_mean, expected_grad_log_chol = torch.autograd.grad(
            expected_objective,
            (mean, log_chol),
        )

        torch.testing.assert_close(actual_grad_mean, expected_grad_mean)
        torch.testing.assert_close(actual_grad_log_chol, expected_grad_log_chol)

    def test_is_consistent_for_exact_decoder_mean(self) -> None:
        r"""For $y_obs = decoder(μ₋)$, the posterior mean should remain unchanged."""
        decoder = ShiftTransform(shift=-0.4)
        updater = GaussianReverseKLUpdater(
            decoder=decoder,
            parametrization="log-cholesky",
            regularization_strength=2.0,
        )
        mean = torch.tensor([0.75, -0.25, 0.5])
        log_chol = torch.zeros(3, 3)
        y_obs = decoder(mean)

        actual_mean, actual_log_chol = updater(y_obs, (mean, log_chol))
        expected_mean, expected_log_chol = _reference_gaussian_reverse_kl_step(
            mean,
            log_chol,
            y_obs,
            shift=decoder.shift,
            regularization_strength=updater.regularization_strength,
        )

        torch.testing.assert_close(actual_mean, mean)
        torch.testing.assert_close(actual_mean, expected_mean)
        torch.testing.assert_close(actual_log_chol, expected_log_chol)
