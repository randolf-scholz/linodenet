r"""Test Gaussian distribution utilities."""

import pytest
import torch
from torch import Tensor
from torch.distributions import MultivariateNormal
from torch.distributions.kl import kl_divergence

from linodenet.distributions.gaussian import (
    CovarianceType,
    argmin_proximal_kl,
    fisher,
    inverse_fisher,
    kl,
    log_prob,
)
from tests.testing import SEEDS_3


@pytest.fixture(params=SEEDS_3, ids="seed={}".format)
def seed(request: pytest.FixtureRequest) -> int:
    r"""Return a reproducible seed for randomized Gaussian distribution tests."""
    return int(request.param)


def _symmetric(matrix: Tensor, /) -> Tensor:
    r"""Return the symmetric part of a square matrix."""
    return 0.5 * (matrix + matrix.mT)


def _directional_second_derivative(fn, /) -> Tensor:
    r"""Return the second derivative of `fn(t)` at `t = 0`."""
    t = torch.zeros((), requires_grad=True)
    value = fn(t)
    gradient = torch.autograd.grad(value, t, create_graph=True)[0]
    return torch.autograd.grad(gradient, t)[0]


def _parameter_inner_product(
    left: tuple[Tensor, Tensor],
    right: tuple[Tensor, Tensor],
    /,
) -> Tensor:
    r"""Return the Euclidean/Frobenius inner product on Gaussian parameters."""
    left_mean, left_matrix = left
    right_mean, right_matrix = right
    return (left_mean * right_mean).sum(dim=-1) + (left_matrix * right_matrix).sum(
        dim=(-2, -1)
    )


def test_kl_matches_torch_distribution() -> None:
    r"""Test the closed-form KL divergence against PyTorch."""
    batch_shape = (2, 3)
    dim = 4

    mean_p = torch.randn(*batch_shape, dim)
    mean_q = torch.randn(*batch_shape, dim)

    factor_p = torch.randn(*batch_shape, dim, dim)
    factor_q = torch.randn(*batch_shape, dim, dim)
    eye = torch.eye(dim)
    cov_p = factor_p @ factor_p.mT + eye
    cov_q = factor_q @ factor_q.mT + eye
    chol_p = torch.linalg.cholesky(cov_p)
    chol_q = torch.linalg.cholesky(cov_q)
    log_chol_p = chol_p.tril(diagonal=-1) + torch.diag_embed(
        chol_p.diagonal(dim1=-2, dim2=-1).log()
    )
    log_chol_q = chol_q.tril(diagonal=-1) + torch.diag_embed(
        chol_q.diagonal(dim1=-2, dim2=-1).log()
    )

    actual = kl((mean_p, cov_p), (mean_q, cov_q))
    actual_cholesky = kl(
        (mean_p, chol_p),
        (mean_q, chol_q),
        parametrization="cholesky",
    )
    actual_log_cholesky = kl(
        (mean_p, log_chol_p),
        (mean_q, log_chol_q),
        parametrization="log-cholesky",
    )
    expected = kl_divergence(
        MultivariateNormal(mean_p, covariance_matrix=cov_p),
        MultivariateNormal(mean_q, covariance_matrix=cov_q),
    )

    assert torch.allclose(actual, expected)
    assert torch.allclose(actual_cholesky, expected)
    assert torch.allclose(actual_log_cholesky, expected)


def test_kl_precision_matches_torch_distribution() -> None:
    r"""Test the precision-parameterized closed-form KL divergence against PyTorch."""
    batch_shape = (2, 3)
    dim = 4

    mean_p = torch.randn(*batch_shape, dim)
    mean_q = torch.randn(*batch_shape, dim)

    factor_p = torch.randn(*batch_shape, dim, dim)
    factor_q = torch.randn(*batch_shape, dim, dim)
    eye = torch.eye(dim)
    cov_p = factor_p @ factor_p.mT + eye
    cov_q = factor_q @ factor_q.mT + eye
    precision_p = torch.linalg.inv(cov_p)
    precision_q = torch.linalg.inv(cov_q)

    actual = kl(
        (mean_p, precision_p),
        (mean_q, precision_q),
        parametrization="precision",
    )
    expected = kl_divergence(
        MultivariateNormal(mean_p, covariance_matrix=cov_p),
        MultivariateNormal(mean_q, covariance_matrix=cov_q),
    )

    assert torch.allclose(actual, expected)


@pytest.mark.parametrize(
    ("sample_shape", "batch_shape"),
    [
        ((), ()),
        ((5,), ()),
        ((), (2, 3)),
        ((5,), (2, 3)),
        ((4, 2), (3,)),
    ],
)
@pytest.mark.parametrize("parametrization", CovarianceType)
def test_log_prob_matches_torch_distribution(
    parametrization: CovarianceType,
    sample_shape: tuple[int, ...],
    batch_shape: tuple[int, ...],
) -> None:
    r"""Test the Gaussian log-density against PyTorch in all parametrizations."""
    dim = 4

    mean = torch.randn(*batch_shape, dim)
    factor = torch.randn(*batch_shape, dim, dim)
    covariance = factor @ factor.mT + torch.eye(dim)
    precision = torch.cholesky_inverse(torch.linalg.cholesky(covariance))
    chol = torch.linalg.cholesky(covariance)
    log_chol = chol.tril(diagonal=-1) + torch.diag_embed(
        chol.diagonal(dim1=-2, dim2=-1).log()
    )
    value = torch.randn(*sample_shape, *batch_shape, dim)

    theta = {
        "covariance": (mean, covariance),
        "precision": (mean, precision),
        "cholesky": (mean, chol),
        "log-cholesky": (mean, log_chol),
    }[parametrization]
    actual = log_prob(value, theta, parametrization=parametrization)
    expected = MultivariateNormal(mean, covariance_matrix=covariance).log_prob(value)

    assert actual.shape == (*sample_shape, *batch_shape)
    assert torch.allclose(actual, expected)


def test_log_prob_rejects_unknown_parametrization() -> None:
    r"""Test that the public log-density dispatch rejects unknown parametrizations."""
    dim = 4
    mean = torch.randn(dim)
    factor = torch.randn(dim, dim)
    covariance = factor @ factor.mT + torch.eye(dim)

    with pytest.raises(ValueError, match="'unknown' is not a valid CovarianceType"):
        log_prob(mean, (mean, covariance), parametrization="unknown")


class TestFisher:
    r"""Tests for the Fisher operator."""

    @pytest.mark.parametrize("parametrization", CovarianceType)
    def test_matches_kl_curvature(
        self, seed: int, parametrization: CovarianceType
    ) -> None:
        r"""Test the Fisher metric against the local KL curvature."""
        torch.manual_seed(seed)
        batch_shape = (2, 3)
        dim = 4

        mean = torch.randn(*batch_shape, dim)
        factor = torch.randn(*batch_shape, dim, dim)
        covariance = factor @ factor.mT + torch.eye(dim)
        delta_mean = torch.randn(*batch_shape, dim)
        chol = torch.linalg.cholesky(covariance)
        match parametrization:
            case CovarianceType.COVARIANCE:
                theta = (mean, covariance)
                tangent = (
                    delta_mean,
                    _symmetric(torch.randn(*batch_shape, dim, dim)),
                )
            case CovarianceType.PRECISION:
                theta = (mean, torch.linalg.inv(covariance))
                tangent = (
                    delta_mean,
                    _symmetric(torch.randn(*batch_shape, dim, dim)),
                )
            case CovarianceType.CHOLESKY:
                theta = (mean, chol)
                tangent = (
                    delta_mean,
                    torch.tril(torch.randn(*batch_shape, dim, dim)),
                )
            case CovarianceType.LOG_CHOLESKY:
                theta = (
                    mean,
                    chol.tril(diagonal=-1)
                    + torch.diag_embed(chol.diagonal(dim1=-2, dim2=-1).log()),
                )
                tangent = (
                    delta_mean,
                    torch.tril(torch.randn(*batch_shape, dim, dim)),
                )

        expected = _parameter_inner_product(
            tangent,
            fisher(theta, tangent, parametrization=parametrization),
        ).sum()
        actual = _directional_second_derivative(
            lambda t: kl(
                (mean + t * tangent[0], theta[1] + t * tangent[1]),
                theta,
                parametrization=parametrization,
            ).sum()
        )

        assert torch.allclose(actual, expected)

    def test_rejects_unknown_parametrization(self) -> None:
        r"""Test that the public Fisher dispatch rejects unknown parametrizations."""
        dim = 4
        mean = torch.randn(dim)
        factor = torch.randn(dim, dim)
        covariance = factor @ factor.mT + torch.eye(dim)

        with pytest.raises(ValueError, match="'unknown' is not a valid CovarianceType"):
            fisher((mean, covariance), (mean, covariance), parametrization="unknown")

    @pytest.mark.parametrize("parametrization", CovarianceType)
    def test_inverse_fisher(self, seed: int, parametrization: CovarianceType) -> None:
        r"""Test that the inverse Fisher operator inverts the Fisher operator."""
        torch.manual_seed(seed)
        batch_shape = (2, 3)
        dim = 4

        mean = torch.randn(*batch_shape, dim)
        factor = torch.randn(*batch_shape, dim, dim)
        covariance = factor @ factor.mT + torch.eye(dim)

        match parametrization:
            case CovarianceType.COVARIANCE:
                theta = (mean, covariance)
                tangent = (
                    torch.randn(*batch_shape, dim),
                    _symmetric(torch.randn(*batch_shape, dim, dim)),
                )
            case CovarianceType.PRECISION:
                theta = (mean, torch.linalg.inv(covariance))
                tangent = (
                    torch.randn(*batch_shape, dim),
                    _symmetric(torch.randn(*batch_shape, dim, dim)),
                )
            case CovarianceType.CHOLESKY:
                theta = (mean, torch.linalg.cholesky(covariance))
                tangent = (
                    torch.randn(*batch_shape, dim),
                    torch.tril(torch.randn(*batch_shape, dim, dim)),
                )
            case CovarianceType.LOG_CHOLESKY:
                chol = torch.linalg.cholesky(covariance)
                theta = (
                    mean,
                    chol.tril(diagonal=-1)
                    + torch.diag_embed(chol.diagonal(dim1=-2, dim2=-1).log()),
                )
                tangent = (
                    torch.randn(*batch_shape, dim),
                    torch.tril(torch.randn(*batch_shape, dim, dim)),
                )
            case _:
                raise AssertionError(f"Unexpected parametrization {parametrization!r}.")

        transported = fisher(theta, tangent, parametrization=parametrization)
        recovered = inverse_fisher(theta, transported, parametrization=parametrization)

        assert (recovered[0] - tangent[0]).abs().amax() < 1e-5
        assert (recovered[1] - tangent[1]).abs().amax() < 1e-5

    def test_inverse_fisher_rejects_unknown_parametrization(self) -> None:
        r"""Test that the public inverse Fisher dispatch rejects unknown parametrizations."""
        dim = 4
        mean = torch.randn(dim)
        factor = torch.randn(dim, dim)
        covariance = factor @ factor.mT + torch.eye(dim)

        with pytest.raises(ValueError, match="'unknown' is not a valid CovarianceType"):
            inverse_fisher(
                (mean, covariance),
                (mean, covariance),
                parametrization="unknown",
            )


class TestArgminProximal:
    r"""Tests for the KL-proximal Gaussian update."""

    @pytest.mark.parametrize("parametrization", CovarianceType)
    def test_solves_closed_form_problem(
        self, seed: int, parametrization: CovarianceType
    ) -> None:
        r"""Test the KL-proximal Gaussian update."""
        torch.manual_seed(seed)
        batch_shape = (2, 3)
        dim = 4
        gamma = torch.tensor(1.7)

        mean_prior = torch.randn(*batch_shape, dim)
        factor = torch.randn(*batch_shape, dim, dim)
        covariance_prior = factor @ factor.mT + torch.eye(dim)
        precision_prior = torch.cholesky_inverse(
            torch.linalg.cholesky(covariance_prior)
        )
        g = torch.randn(*batch_shape, dim)
        expected_mean = mean_prior - torch.einsum(
            "...ij,...j->...i", covariance_prior, g / gamma
        )
        precision_chol_prior = torch.linalg.cholesky(precision_prior)
        covariance_chol_prior = torch.linalg.cholesky(covariance_prior)
        log_chol_prior = covariance_chol_prior.tril(diagonal=-1) + torch.diag_embed(
            covariance_chol_prior.diagonal(dim1=-2, dim2=-1).log()
        )
        match parametrization:
            case CovarianceType.COVARIANCE:
                precision_shift_factor = torch.randn(*batch_shape, dim, dim)
                precision_shift = precision_shift_factor @ precision_shift_factor.mT
                gradient_matrix = 0.5 * gamma * precision_shift
                theta_prior = (mean_prior, covariance_prior)

                def objective_fn(theta: tuple[Tensor, Tensor], /) -> Tensor:
                    mean, covariance = theta
                    return (g * mean).sum() + (gradient_matrix * covariance).sum()

                expected_precision = 0.5 * (
                    precision_prior
                    + precision_shift
                    + (precision_prior + precision_shift).mT
                )
                expected_matrix = torch.cholesky_inverse(
                    torch.linalg.cholesky(expected_precision)
                )
                matrix_tol = 1e-4
                mean_tol = 1e-6
                projected_grad = _symmetric

            case CovarianceType.PRECISION:
                gradient_precision_factor = torch.randn(*batch_shape, dim, dim)
                gradient_matrix = (
                    gradient_precision_factor @ gradient_precision_factor.mT
                )
                theta_prior = (mean_prior, precision_prior)

                def objective_fn(theta: tuple[Tensor, Tensor], /) -> Tensor:
                    mean, precision = theta
                    return (g * mean).sum() + (gradient_matrix * precision).sum()

                whitened_gradient = (
                    precision_chol_prior.mT @ gradient_matrix @ precision_chol_prior
                )
                whitened_gradient = _symmetric(whitened_gradient)
                eigenvalues, eigenvectors = torch.linalg.eigh(whitened_gradient)
                spectral_scale = 2 / (1 + torch.sqrt(1 + 8 * eigenvalues / gamma))
                whitened_precision = (
                    eigenvectors @ torch.diag_embed(spectral_scale) @ eigenvectors.mT
                )
                expected_matrix = (
                    precision_chol_prior @ whitened_precision @ precision_chol_prior.mT
                )
                matrix_tol = 1e-4
                mean_tol = 1e-5
                projected_grad = _symmetric

            case CovarianceType.CHOLESKY:
                gradient_matrix = torch.tril(torch.randn(*batch_shape, dim, dim))
                theta_prior = (mean_prior, covariance_chol_prior)

                def objective_fn(theta: tuple[Tensor, Tensor], /) -> Tensor:
                    mean, chol = theta
                    return (g * mean).sum() + (gradient_matrix * chol).sum()

                whitened_gradient = covariance_chol_prior.mT @ gradient_matrix
                diagonal_gradient = whitened_gradient.diagonal(dim1=-2, dim2=-1)
                diagonal_update = (
                    0.5
                    * (
                        -diagonal_gradient
                        + torch.sqrt(diagonal_gradient.square() + 4 * gamma.square())
                    )
                    / gamma
                )
                whitened_cholesky = torch.tril(
                    -whitened_gradient / gamma, diagonal=-1
                ) + torch.diag_embed(diagonal_update)
                expected_matrix = torch.tril(covariance_chol_prior @ whitened_cholesky)
                matrix_tol = 1e-5
                mean_tol = 1e-5
                projected_grad = torch.tril

            case CovarianceType.LOG_CHOLESKY:
                gradient_matrix = torch.tril(torch.randn(*batch_shape, dim, dim))
                diagonal_gradient = gradient_matrix.diagonal(dim1=-2, dim2=-1)
                diagonal_gradient = 0.5 * gamma * torch.tanh(diagonal_gradient)
                gradient_matrix = gradient_matrix.tril(diagonal=-1) + torch.diag_embed(
                    diagonal_gradient
                )
                theta_prior = (mean_prior, log_chol_prior)

                def objective_fn(theta: tuple[Tensor, Tensor], /) -> Tensor:
                    mean, log_chol = theta
                    return (g * mean).sum() + (gradient_matrix * log_chol).sum()

                gradient_off = torch.tril(gradient_matrix, diagonal=-1)
                linear_term = covariance_chol_prior.mT @ gradient_off
                diagonal_linear = linear_term.diagonal(dim1=-2, dim2=-1)
                diagonal_update = (
                    0.5
                    * (
                        -diagonal_linear
                        + torch.sqrt(
                            diagonal_linear.square()
                            + 4 * gamma * (gamma - diagonal_gradient)
                        )
                    )
                    / gamma
                )
                whitened_cholesky = torch.tril(
                    -linear_term / gamma, diagonal=-1
                ) + torch.diag_embed(diagonal_update)
                expected_cholesky = torch.tril(
                    covariance_chol_prior @ whitened_cholesky
                )
                expected_matrix = expected_cholesky.tril(
                    diagonal=-1
                ) + torch.diag_embed(expected_cholesky.diagonal(dim1=-2, dim2=-1).log())
                matrix_tol = 1e-5
                mean_tol = 1e-5
                projected_grad = torch.tril

        mean_post, matrix_post = argmin_proximal_kl(
            objective_fn,
            theta_prior,
            gamma=gamma,
            parametrization=parametrization,
        )

        assert (mean_post - expected_mean).abs().amax() < 1e-5
        assert (matrix_post - expected_matrix).abs().amax() < 1e-5

        mean_var = mean_post.detach().clone().requires_grad_(True)
        matrix_var = matrix_post.detach().clone().requires_grad_(True)
        objective = (
            (g * (mean_var - mean_prior)).sum()
            + (gradient_matrix * (matrix_var - theta_prior[1])).sum()
            + gamma
            * kl(
                (mean_var, matrix_var),
                theta_prior,
                parametrization=parametrization,
            ).sum()
        )
        mean_grad, matrix_grad = torch.autograd.grad(
            objective,
            (mean_var, matrix_var),
        )

        assert mean_grad.abs().amax() < mean_tol
        assert projected_grad(matrix_grad).abs().amax() < matrix_tol

    @pytest.mark.parametrize("parametrization", CovarianceType)
    def test_raises_when_objective_has_no_finite_minimizer(
        self, seed: int, parametrization: CovarianceType
    ) -> None:
        r"""Test that ill-posed linearized objectives are rejected."""
        torch.manual_seed(seed)
        dim = 4
        mean_prior = torch.randn(dim)
        factor = torch.randn(dim, dim)
        covariance_prior = factor @ factor.mT + torch.eye(dim)
        precision_prior = torch.linalg.inv(covariance_prior)
        chol_prior = torch.linalg.cholesky(covariance_prior)
        log_chol_prior = chol_prior.tril(diagonal=-1) + torch.diag_embed(
            chol_prior.diagonal(dim1=-2, dim2=-1).log()
        )
        mean_gradient = torch.randn(dim)
        gamma = torch.tensor(1.0)

        match parametrization:
            case CovarianceType.COVARIANCE:
                theta_prior = (mean_prior, covariance_prior)

                def objective_fn(theta: tuple[Tensor, Tensor], /) -> Tensor:
                    return (mean_gradient * theta[0]).sum() + (
                        -precision_prior * theta[1]
                    ).sum()

            case CovarianceType.PRECISION:
                theta_prior = (mean_prior, precision_prior)

                def objective_fn(theta: tuple[Tensor, Tensor], /) -> Tensor:
                    return (mean_gradient * theta[0]).sum() + (
                        -torch.eye(dim) * theta[1]
                    ).sum()

            case CovarianceType.CHOLESKY:
                pytest.skip("Not implemented")

            case CovarianceType.LOG_CHOLESKY:
                diagonal_gradient = gamma + 0.1 + torch.rand(dim)
                gradient_log_cholesky = torch.diag_embed(diagonal_gradient)
                theta_prior = (mean_prior, log_chol_prior)

                def objective_fn(theta: tuple[Tensor, Tensor], /) -> Tensor:
                    return (mean_gradient * theta[0]).sum() + (
                        gradient_log_cholesky * theta[1]
                    ).sum()

        with pytest.raises(ValueError, match="finite minimizer"):
            argmin_proximal_kl(
                objective_fn,
                theta_prior,
                gamma=gamma,
                parametrization=parametrization,
            )

    def test_rejects_unknown_parametrization(self) -> None:
        r"""Test that the proximal update rejects unknown parametrizations."""
        dim = 4
        mean = torch.randn(dim)
        factor = torch.randn(dim, dim)
        covariance = factor @ factor.mT + torch.eye(dim)

        with pytest.raises(ValueError, match="'unknown' is not a valid CovarianceType"):
            argmin_proximal_kl(
                lambda theta: theta[0].sum() + theta[1].sum(),
                (mean, covariance),
                parametrization="unknown",
            )
