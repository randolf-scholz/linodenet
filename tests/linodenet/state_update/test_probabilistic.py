import pytest
import torch
from torch import Tensor, nn

from linodenet.distributions.gaussian import argmin_forward_kl, argmin_reverse_kl
from linodenet.state_update import GaussianForwardUpdater, GaussianReverseUpdater


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


def _scale_tril(log_chol: Tensor, /) -> Tensor:
    r"""Convert log-Cholesky parameters to a lower-triangular scale matrix."""
    return log_chol.tril(diagonal=-1) + torch.diag_embed(
        log_chol.diagonal(dim1=-2, dim2=-1).exp()
    )


def _reference_gaussian_forward_kl_step(
    mean: Tensor,
    log_chol: Tensor,
    y_obs: Tensor,
    /,
    *,
    shift: Tensor,
    retention: Tensor | tuple[Tensor, Tensor],
) -> tuple[Tensor, Tensor]:
    r"""Compute the exact Gaussian forward-KL update in latent coordinates."""
    return argmin_forward_kl(
        y_obs - shift,
        (mean, log_chol),
        retention=retention,
        parametrization="log-cholesky",
    )


def _reference_gaussian_reverse_kl_step(
    mean: Tensor,
    log_chol: Tensor,
    y_obs: Tensor,
    /,
    *,
    shift: Tensor,
    retention: Tensor | tuple[Tensor, Tensor],
) -> tuple[Tensor, Tensor]:
    r"""Compute the exact Gaussian reverse-KL update in latent coordinates."""
    return argmin_reverse_kl(
        y_obs - shift,
        (mean, log_chol),
        retention=retention,
        parametrization="log-cholesky",
    )


class TestGaussianForwardKLUpdater:
    def test_retention_must_lie_in_unit_interval(self) -> None:
        r"""Shared retention should be constrained to the unit interval."""
        with pytest.raises(ValueError, match=r"ρ must be in \[0, 1\]"):
            GaussianForwardUpdater(
                decoder=ShiftTransform(shift=0.0),
                parametrization="log-cholesky",
                retention=1.1,
            )

    def test_retention_learnable_can_be_disabled(self) -> None:
        r"""The shared retention parameter should support frozen initialization."""
        retention = torch.tensor(1.7 / 2.7)
        updater = GaussianForwardUpdater(
            decoder=ShiftTransform(shift=0.0),
            parametrization="log-cholesky",
            retention=retention,
            retention_learnable=False,
        )

        assert all(
            not param.requires_grad for param in updater.retention_mu.parameters()
        )
        assert all(
            not param.requires_grad for param in updater.retention_sigma.parameters()
        )
        torch.testing.assert_close(updater.retention_mu(None), retention)
        torch.testing.assert_close(updater.retention_sigma(None), retention)

    def test_forward_and_backward(self) -> None:
        r"""Forward output and parameter gradients should match the exact update."""
        decoder = ShiftTransform(shift=0.2)
        retention = torch.tensor(1.7 / 2.7)
        updater = GaussianForwardUpdater(
            decoder=decoder,
            parametrization="log-cholesky",
            retention=retention,
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
        retention_param = next(updater.retention_mu.parameters())

        actual_mean, actual_log_chol = updater(y_obs, (mean, log_chol))
        expected_mean, expected_log_chol = _reference_gaussian_forward_kl_step(
            mean,
            log_chol,
            y_obs,
            shift=decoder.shift,
            retention=torch.sigmoid(retention_param),
        )

        torch.testing.assert_close(actual_mean, expected_mean)
        torch.testing.assert_close(actual_log_chol, expected_log_chol)

        objective = actual_mean.sum() + actual_log_chol.sum()
        actual_grad_shift, actual_grad_retention = torch.autograd.grad(
            objective,
            (decoder.shift, retention_param),
        )
        expected_objective = expected_mean.sum() + expected_log_chol.sum()
        expected_grad_shift, expected_grad_retention = torch.autograd.grad(
            expected_objective,
            (decoder.shift, retention_param),
        )

        torch.testing.assert_close(actual_grad_shift, expected_grad_shift)
        torch.testing.assert_close(actual_grad_retention, expected_grad_retention)

    def test_compile_fullgraph(self) -> None:
        r"""The updater should compile under `torch.compile(fullgraph=True)`."""
        updater = GaussianForwardUpdater(
            decoder=ShiftTransform(shift=-0.1),
            parametrization="log-cholesky",
            retention=1.2 / 2.2,
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
        r"""Batched parameters should use a per-sample exact forward-KL update."""
        decoder = ShiftTransform(shift=0.1)
        retention = torch.tensor(0.5)
        updater = GaussianForwardUpdater(
            decoder=decoder,
            parametrization="log-cholesky",
            retention=retention,
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
        expected_mean, expected_log_chol = _reference_gaussian_forward_kl_step(
            mean,
            log_chol,
            y_obs,
            shift=decoder.shift,
            retention=retention,
        )

        torch.testing.assert_close(actual_mean, expected_mean)
        torch.testing.assert_close(actual_log_chol, expected_log_chol)

    def test_gradients_wrt_theta(self) -> None:
        r"""The update should remain differentiable with respect to the prior."""
        decoder = ShiftTransform(shift=0.25)
        retention = torch.tensor(1.4 / 2.4)
        updater = GaussianForwardUpdater(
            decoder=decoder,
            parametrization="log-cholesky",
            retention=retention,
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

        expected_mean, expected_log_chol = _reference_gaussian_forward_kl_step(
            mean,
            log_chol,
            y_obs,
            shift=decoder.shift,
            retention=retention,
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
        retention = torch.tensor(2.0 / 3.0)
        updater = GaussianForwardUpdater(
            decoder=decoder,
            parametrization="log-cholesky",
            retention=retention,
        )
        mean = torch.tensor([0.75, -0.25, 0.5])
        log_chol = torch.zeros(3, 3)
        y_obs = decoder(mean)

        actual_mean, actual_log_chol = updater(y_obs, (mean, log_chol))
        expected_mean, expected_log_chol = _reference_gaussian_forward_kl_step(
            mean,
            log_chol,
            y_obs,
            shift=decoder.shift,
            retention=retention,
        )

        torch.testing.assert_close(actual_mean, mean)
        torch.testing.assert_close(actual_mean, expected_mean)
        torch.testing.assert_close(actual_log_chol, expected_log_chol)


class TestGaussianReverseKLUpdater:
    def test_retention_must_lie_in_unit_interval(self) -> None:
        r"""Shared retention should be constrained to the unit interval."""
        with pytest.raises(ValueError, match=r"ρ must be in \[0, 1\]"):
            GaussianReverseUpdater(
                decoder=ShiftTransform(shift=0.0),
                parametrization="log-cholesky",
                retention=1.1,
            )

    def test_split_retention_sigma_must_be_positive(self) -> None:
        r"""Split retention should enforce the reverse-KL covariance constraint."""
        with pytest.raises(ValueError, match=r"ρ_sigma must be in \(0, 1\]"):
            GaussianReverseUpdater(
                decoder=ShiftTransform(shift=0.0),
                parametrization="log-cholesky",
                retention=(0.5, 0.0),
            )

    def test_retention_learnable_can_be_disabled(self) -> None:
        r"""Split retention parameters should support frozen initialization."""
        retention = (torch.tensor(1.7 / 2.7), torch.tensor((1.7 - 1.0) / 1.7))
        updater = GaussianReverseUpdater(
            decoder=ShiftTransform(shift=0.0),
            parametrization="log-cholesky",
            retention=retention,
            retention_learnable=False,
        )

        assert all(
            not param.requires_grad for param in updater.retention_mu.parameters()
        )
        assert all(
            not param.requires_grad for param in updater.retention_sigma.parameters()
        )
        torch.testing.assert_close(updater.retention_mu(None), retention[0])
        torch.testing.assert_close(updater.retention_sigma(None), retention[1])

    def test_forward_and_backward(self) -> None:
        r"""Forward output and parameter gradients should match the exact update."""
        decoder = ShiftTransform(shift=0.2)
        retention = (torch.tensor(1.7 / 2.7), torch.tensor((1.7 - 1.0) / 1.7))
        updater = GaussianReverseUpdater(
            decoder=decoder,
            parametrization="log-cholesky",
            retention=retention,
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
        retention_mu_param = next(updater.retention_mu.parameters())
        retention_sigma_param = next(updater.retention_sigma.parameters())

        actual_mean, actual_log_chol = updater(y_obs, (mean, log_chol))
        expected_mean, expected_log_chol = _reference_gaussian_reverse_kl_step(
            mean,
            log_chol,
            y_obs,
            shift=decoder.shift,
            retention=(
                torch.sigmoid(retention_mu_param),
                torch.sigmoid(retention_sigma_param),
            ),
        )

        torch.testing.assert_close(actual_mean, expected_mean)
        torch.testing.assert_close(actual_log_chol, expected_log_chol)

        objective = actual_mean.sum() + actual_log_chol.sum()
        actual_grad_shift, actual_grad_retention_mu, actual_grad_retention_sigma = (
            torch.autograd.grad(
                objective,
                (decoder.shift, retention_mu_param, retention_sigma_param),
            )
        )
        expected_objective = expected_mean.sum() + expected_log_chol.sum()
        (
            expected_grad_shift,
            expected_grad_retention_mu,
            expected_grad_retention_sigma,
        ) = torch.autograd.grad(
            expected_objective,
            (decoder.shift, retention_mu_param, retention_sigma_param),
        )

        torch.testing.assert_close(actual_grad_shift, expected_grad_shift)
        torch.testing.assert_close(actual_grad_retention_mu, expected_grad_retention_mu)
        torch.testing.assert_close(
            actual_grad_retention_sigma, expected_grad_retention_sigma
        )

    def test_compile_fullgraph(self) -> None:
        r"""The updater should compile under `torch.compile(fullgraph=True)`."""
        updater = GaussianReverseUpdater(
            decoder=ShiftTransform(shift=-0.1),
            parametrization="log-cholesky",
            retention=(1.2 / 2.2, (1.2 - 1.0) / 1.2),
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
        retention = (torch.tensor(0.5), torch.tensor(0.75))
        updater = GaussianReverseUpdater(
            decoder=decoder,
            parametrization="log-cholesky",
            retention=retention,
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
            retention=retention,
        )

        torch.testing.assert_close(actual_mean, expected_mean)
        torch.testing.assert_close(actual_log_chol, expected_log_chol)

    def test_gradients_wrt_theta(self) -> None:
        r"""The update should remain differentiable with respect to the prior."""
        decoder = ShiftTransform(shift=0.25)
        retention = (torch.tensor(1.4 / 2.4), torch.tensor((1.4 - 1.0) / 1.4))
        updater = GaussianReverseUpdater(
            decoder=decoder,
            parametrization="log-cholesky",
            retention=retention,
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
            retention=retention,
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
        retention = (torch.tensor(2.0 / 3.0), torch.tensor(0.75))
        updater = GaussianReverseUpdater(
            decoder=decoder,
            parametrization="log-cholesky",
            retention=retention,
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
            retention=retention,
        )

        torch.testing.assert_close(actual_mean, mean)
        torch.testing.assert_close(actual_mean, expected_mean)
        torch.testing.assert_close(actual_log_chol, expected_log_chol)
