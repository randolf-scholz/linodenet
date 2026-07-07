r"""Test Gaussian distribution utilities."""

import pytest
import torch
from torch.distributions import MultivariateNormal
from torch.distributions.kl import kl_divergence

from linodenet.distributions.gaussian import (
    argmin_proximal_kl,
    fisher,
    inverse_fisher,
    kl,
    log_prob,
)


def _symmetric(matrix: torch.Tensor, /) -> torch.Tensor:
    r"""Return the symmetric part of a square matrix."""
    return 0.5 * (matrix + matrix.mT)


def _directional_second_derivative(fn, /) -> torch.Tensor:
    r"""Return the second derivative of `fn(t)` at `t = 0`."""
    t = torch.zeros((), requires_grad=True)
    value = fn(t)
    gradient = torch.autograd.grad(value, t, create_graph=True)[0]
    return torch.autograd.grad(gradient, t)[0]


def _parameter_inner_product(
    left: tuple[torch.Tensor, torch.Tensor],
    right: tuple[torch.Tensor, torch.Tensor],
    /,
) -> torch.Tensor:
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
    "parametrization",
    ["covariance", "precision", "cholesky", "log-cholesky"],
)
def test_log_prob_matches_torch_distribution(parametrization: str) -> None:
    r"""Test the Gaussian log-density against PyTorch in all parametrizations."""
    batch_shape = (2, 3)
    sample_shape = (5,)
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

    assert torch.allclose(actual, expected)


def test_fisher_matches_covariance_kl_curvature() -> None:
    r"""Test the covariance Fisher metric against the local KL curvature."""
    batch_shape = (2, 3)
    dim = 4

    mean = torch.randn(*batch_shape, dim)
    factor = torch.randn(*batch_shape, dim, dim)
    covariance = factor @ factor.mT + torch.eye(dim)

    delta_mean = torch.randn(*batch_shape, dim)
    delta_covariance = _symmetric(torch.randn(*batch_shape, dim, dim))

    expected = _parameter_inner_product(
        (delta_mean, delta_covariance),
        fisher((mean, covariance), (delta_mean, delta_covariance)),
    ).sum()

    actual = _directional_second_derivative(
        lambda t: kl(
            (mean + t * delta_mean, covariance + t * delta_covariance),
            (mean, covariance),
        ).sum()
    )

    assert torch.allclose(actual, expected)


def test_fisher_precision_matches_kl_curvature() -> None:
    r"""Test the precision Fisher metric against the local KL curvature."""
    batch_shape = (2, 3)
    dim = 4

    mean = torch.randn(*batch_shape, dim)
    factor = torch.randn(*batch_shape, dim, dim)
    covariance = factor @ factor.mT + torch.eye(dim)
    precision = torch.linalg.inv(covariance)

    delta_mean = torch.randn(*batch_shape, dim)
    delta_precision = _symmetric(torch.randn(*batch_shape, dim, dim))

    expected = _parameter_inner_product(
        (delta_mean, delta_precision),
        fisher(
            (mean, precision),
            (delta_mean, delta_precision),
            parametrization="precision",
        ),
    ).sum()

    actual = _directional_second_derivative(
        lambda t: kl(
            (mean + t * delta_mean, precision + t * delta_precision),
            (mean, precision),
            parametrization="precision",
        ).sum()
    )

    assert torch.allclose(actual, expected)


def test_fisher_cholesky_matches_kl_curvature() -> None:
    r"""Test the Cholesky Fisher metric against the local KL curvature."""
    batch_shape = (2, 3)
    dim = 4

    mean = torch.randn(*batch_shape, dim)
    factor = torch.randn(*batch_shape, dim, dim)
    covariance = factor @ factor.mT + torch.eye(dim)
    chol = torch.linalg.cholesky(covariance)

    delta_mean = torch.randn(*batch_shape, dim)
    delta_chol = torch.tril(torch.randn(*batch_shape, dim, dim))

    expected = _parameter_inner_product(
        (delta_mean, delta_chol),
        fisher((mean, chol), (delta_mean, delta_chol), parametrization="cholesky"),
    ).sum()

    actual = _directional_second_derivative(
        lambda t: kl(
            (mean + t * delta_mean, chol + t * delta_chol),
            (mean, chol),
            parametrization="cholesky",
        ).sum()
    )

    assert torch.allclose(actual, expected)


def test_fisher_log_cholesky_matches_kl_curvature() -> None:
    r"""Test the log-Cholesky Fisher metric against the local KL curvature."""
    batch_shape = (2, 3)
    dim = 4

    mean = torch.randn(*batch_shape, dim)
    factor = torch.randn(*batch_shape, dim, dim)
    covariance = factor @ factor.mT + torch.eye(dim)
    chol = torch.linalg.cholesky(covariance)
    log_chol = chol.tril(diagonal=-1) + torch.diag_embed(
        chol.diagonal(dim1=-2, dim2=-1).log()
    )

    delta_mean = torch.randn(*batch_shape, dim)
    delta_log_chol = torch.tril(torch.randn(*batch_shape, dim, dim))

    expected = _parameter_inner_product(
        (delta_mean, delta_log_chol),
        fisher(
            (mean, log_chol),
            (delta_mean, delta_log_chol),
            parametrization="log-cholesky",
        ),
    ).sum()

    actual = _directional_second_derivative(
        lambda t: kl(
            (mean + t * delta_mean, log_chol + t * delta_log_chol),
            (mean, log_chol),
            parametrization="log-cholesky",
        ).sum()
    )

    assert torch.allclose(actual, expected)


@pytest.mark.parametrize(
    "parametrization",
    ["covariance", "precision", "cholesky", "log-cholesky"],
)
def test_inverse_fisher_inverts_fisher(parametrization: str) -> None:
    r"""Test that the inverse Fisher operator inverts the Fisher operator."""
    batch_shape = (2, 3)
    dim = 4

    mean = torch.randn(*batch_shape, dim)
    factor = torch.randn(*batch_shape, dim, dim)
    covariance = factor @ factor.mT + torch.eye(dim)

    match parametrization:
        case "covariance":
            theta = (mean, covariance)
            tangent = (
                torch.randn(*batch_shape, dim),
                _symmetric(torch.randn(*batch_shape, dim, dim)),
            )
        case "precision":
            theta = (mean, torch.linalg.inv(covariance))
            tangent = (
                torch.randn(*batch_shape, dim),
                _symmetric(torch.randn(*batch_shape, dim, dim)),
            )
        case "cholesky":
            theta = (mean, torch.linalg.cholesky(covariance))
            tangent = (
                torch.randn(*batch_shape, dim),
                torch.tril(torch.randn(*batch_shape, dim, dim)),
            )
        case "log-cholesky":
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


def test_argmin_proximal_kl_covariance_solves_closed_form_problem() -> None:
    r"""Test the covariance-form KL-proximal Gaussian update."""
    batch_shape = (2, 3)
    dim = 4
    gamma = torch.tensor(1.7)

    mean_prior = torch.randn(*batch_shape, dim)
    factor = torch.randn(*batch_shape, dim, dim)
    covariance_prior = factor @ factor.mT + torch.eye(dim)
    precision_prior = torch.cholesky_inverse(torch.linalg.cholesky(covariance_prior))

    g = torch.randn(*batch_shape, dim)
    precision_shift_factor = torch.randn(*batch_shape, dim, dim)
    precision_shift = precision_shift_factor @ precision_shift_factor.mT
    gradient_covariance = 0.5 * gamma * precision_shift

    def objective_fn(theta: tuple[torch.Tensor, torch.Tensor], /) -> torch.Tensor:
        mean, covariance = theta
        return (g * mean).sum() + (gradient_covariance * covariance).sum()

    mean_post, covariance_post = argmin_proximal_kl(
        objective_fn,
        (mean_prior, covariance_prior),
        gamma=gamma,
    )

    expected_mean = mean_prior - torch.einsum(
        "...ij,...j->...i", covariance_prior, g / gamma
    )
    expected_precision = 0.5 * (
        precision_prior + precision_shift + (precision_prior + precision_shift).mT
    )
    expected_covariance = torch.cholesky_inverse(
        torch.linalg.cholesky(expected_precision)
    )

    assert (mean_post - expected_mean).abs().amax() < 1e-5
    assert (covariance_post - expected_covariance).abs().amax() < 1e-5

    mean_var = mean_post.detach().clone().requires_grad_(True)
    covariance_var = covariance_post.detach().clone().requires_grad_(True)
    objective = (
        (g * (mean_var - mean_prior)).sum()
        + (gradient_covariance * (covariance_var - covariance_prior)).sum()
        + gamma * kl((mean_var, covariance_var), (mean_prior, covariance_prior)).sum()
    )
    mean_grad, covariance_grad = torch.autograd.grad(
        objective,
        (mean_var, covariance_var),
    )

    assert mean_grad.abs().amax() < 1e-6
    assert _symmetric(covariance_grad).abs().amax() < 1e-4


def test_argmin_proximal_kl_precision_solves_closed_form_problem() -> None:
    r"""Test the precision-form KL-proximal Gaussian update."""
    batch_shape = (2, 3)
    dim = 4
    gamma = torch.tensor(1.7)

    mean_prior = torch.randn(*batch_shape, dim)
    factor = torch.randn(*batch_shape, dim, dim)
    covariance_prior = factor @ factor.mT + torch.eye(dim)
    precision_prior = torch.cholesky_inverse(torch.linalg.cholesky(covariance_prior))
    chol_prior = torch.linalg.cholesky(precision_prior)

    g = torch.randn(*batch_shape, dim)
    gradient_precision_factor = torch.randn(*batch_shape, dim, dim)
    gradient_precision = gradient_precision_factor @ gradient_precision_factor.mT

    def objective_fn(theta: tuple[torch.Tensor, torch.Tensor], /) -> torch.Tensor:
        mean, precision = theta
        return (g * mean).sum() + (gradient_precision * precision).sum()

    mean_post, precision_post = argmin_proximal_kl(
        objective_fn,
        (mean_prior, precision_prior),
        gamma=gamma,
        parametrization="precision",
    )

    expected_mean = mean_prior - torch.einsum(
        "...ij,...j->...i", covariance_prior, g / gamma
    )
    whitened_gradient = chol_prior.mT @ gradient_precision @ chol_prior
    whitened_gradient = _symmetric(whitened_gradient)
    eigenvalues, eigenvectors = torch.linalg.eigh(whitened_gradient)
    spectral_scale = 2 / (1 + torch.sqrt(1 + 8 * eigenvalues / gamma))
    whitened_precision = (
        eigenvectors @ torch.diag_embed(spectral_scale) @ eigenvectors.mT
    )
    expected_precision = chol_prior @ whitened_precision @ chol_prior.mT

    assert (mean_post - expected_mean).abs().amax() < 1e-5
    assert (precision_post - expected_precision).abs().amax() < 1e-5

    mean_var = mean_post.detach().clone().requires_grad_(True)
    precision_var = precision_post.detach().clone().requires_grad_(True)
    objective = (
        (g * (mean_var - mean_prior)).sum()
        + (gradient_precision * (precision_var - precision_prior)).sum()
        + gamma
        * kl(
            (mean_var, precision_var),
            (mean_prior, precision_prior),
            parametrization="precision",
        ).sum()
    )
    mean_grad, precision_grad = torch.autograd.grad(
        objective,
        (mean_var, precision_var),
    )

    assert mean_grad.abs().amax() < 1e-5
    assert _symmetric(precision_grad).abs().amax() < 1e-4


def test_argmin_proximal_kl_cholesky_solves_closed_form_problem() -> None:
    r"""Test the Cholesky-form KL-proximal Gaussian update."""
    batch_shape = (2, 3)
    dim = 4
    gamma = torch.tensor(1.7)

    mean_prior = torch.randn(*batch_shape, dim)
    factor = torch.randn(*batch_shape, dim, dim)
    covariance_prior = factor @ factor.mT + torch.eye(dim)
    chol_prior = torch.linalg.cholesky(covariance_prior)

    g = torch.randn(*batch_shape, dim)
    gradient_cholesky = torch.tril(torch.randn(*batch_shape, dim, dim))

    def objective_fn(theta: tuple[torch.Tensor, torch.Tensor], /) -> torch.Tensor:
        mean, chol = theta
        return (g * mean).sum() + (gradient_cholesky * chol).sum()

    mean_post, chol_post = argmin_proximal_kl(
        objective_fn,
        (mean_prior, chol_prior),
        gamma=gamma,
        parametrization="cholesky",
    )

    expected_mean = mean_prior - torch.einsum(
        "...ij,...j->...i", covariance_prior, g / gamma
    )
    whitened_gradient = chol_prior.mT @ gradient_cholesky
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
    expected_cholesky = torch.tril(chol_prior @ whitened_cholesky)

    assert (mean_post - expected_mean).abs().amax() < 1e-5
    assert (chol_post - expected_cholesky).abs().amax() < 1e-5

    mean_var = mean_post.detach().clone().requires_grad_(True)
    chol_var = chol_post.detach().clone().requires_grad_(True)
    objective = (
        (g * (mean_var - mean_prior)).sum()
        + (gradient_cholesky * (chol_var - chol_prior)).sum()
        + gamma
        * kl(
            (mean_var, chol_var),
            (mean_prior, chol_prior),
            parametrization="cholesky",
        ).sum()
    )
    mean_grad, chol_grad = torch.autograd.grad(
        objective,
        (mean_var, chol_var),
    )

    assert mean_grad.abs().amax() < 1e-5
    assert torch.tril(chol_grad).abs().amax() < 1e-5


def test_argmin_proximal_kl_log_cholesky_solves_closed_form_problem() -> None:
    r"""Test the log-Cholesky-form KL-proximal Gaussian update."""
    batch_shape = (2, 3)
    dim = 4
    gamma = torch.tensor(1.7)

    mean_prior = torch.randn(*batch_shape, dim)
    factor = torch.randn(*batch_shape, dim, dim)
    covariance_prior = factor @ factor.mT + torch.eye(dim)
    chol_prior = torch.linalg.cholesky(covariance_prior)
    log_chol_prior = chol_prior.tril(diagonal=-1) + torch.diag_embed(
        chol_prior.diagonal(dim1=-2, dim2=-1).log()
    )

    g = torch.randn(*batch_shape, dim)
    gradient_log_cholesky = torch.tril(torch.randn(*batch_shape, dim, dim))
    diagonal_gradient = gradient_log_cholesky.diagonal(dim1=-2, dim2=-1)
    diagonal_gradient = 0.5 * gamma * torch.tanh(diagonal_gradient)
    gradient_log_cholesky = gradient_log_cholesky.tril(diagonal=-1) + torch.diag_embed(
        diagonal_gradient
    )

    def objective_fn(theta: tuple[torch.Tensor, torch.Tensor], /) -> torch.Tensor:
        mean, log_chol = theta
        return (g * mean).sum() + (gradient_log_cholesky * log_chol).sum()

    mean_post, log_chol_post = argmin_proximal_kl(
        objective_fn,
        (mean_prior, log_chol_prior),
        gamma=gamma,
        parametrization="log-cholesky",
    )

    expected_mean = mean_prior - torch.einsum(
        "...ij,...j->...i", covariance_prior, g / gamma
    )
    gradient_off = torch.tril(gradient_log_cholesky, diagonal=-1)
    linear_term = chol_prior.mT @ gradient_off
    diagonal_linear = linear_term.diagonal(dim1=-2, dim2=-1)
    diagonal_update = (
        0.5
        * (
            -diagonal_linear
            + torch.sqrt(
                diagonal_linear.square() + 4 * gamma * (gamma - diagonal_gradient)
            )
        )
        / gamma
    )
    whitened_cholesky = torch.tril(
        -linear_term / gamma, diagonal=-1
    ) + torch.diag_embed(diagonal_update)
    expected_cholesky = torch.tril(chol_prior @ whitened_cholesky)
    expected_log_cholesky = expected_cholesky.tril(diagonal=-1) + torch.diag_embed(
        expected_cholesky.diagonal(dim1=-2, dim2=-1).log()
    )

    assert (mean_post - expected_mean).abs().amax() < 1e-5
    assert (log_chol_post - expected_log_cholesky).abs().amax() < 1e-5

    mean_var = mean_post.detach().clone().requires_grad_(True)
    log_chol_var = log_chol_post.detach().clone().requires_grad_(True)
    objective = (
        (g * (mean_var - mean_prior)).sum()
        + (gradient_log_cholesky * (log_chol_var - log_chol_prior)).sum()
        + gamma
        * kl(
            (mean_var, log_chol_var),
            (mean_prior, log_chol_prior),
            parametrization="log-cholesky",
        ).sum()
    )
    mean_grad, log_chol_grad = torch.autograd.grad(
        objective,
        (mean_var, log_chol_var),
    )

    assert mean_grad.abs().amax() < 1e-5
    assert torch.tril(log_chol_grad).abs().amax() < 1e-5


def test_argmin_proximal_kl_covariance_raises_when_precision_is_not_pd() -> None:
    r"""Test that the covariance update rejects ill-posed precision shifts."""
    dim = 4
    mean_prior = torch.randn(dim)
    factor = torch.randn(dim, dim)
    covariance_prior = factor @ factor.mT + torch.eye(dim)
    precision_prior = torch.linalg.inv(covariance_prior)
    mean_gradient = torch.randn(dim)

    with pytest.raises(ValueError, match="finite minimizer"):
        argmin_proximal_kl(
            lambda theta: (
                (mean_gradient * theta[0]).sum() + (-precision_prior * theta[1]).sum()
            ),
            (mean_prior, covariance_prior),
        )


def test_argmin_proximal_kl_precision_raises_when_gradient_is_not_psd() -> None:
    r"""Test that the precision update rejects unbounded linearized objectives."""
    dim = 4
    mean_prior = torch.randn(dim)
    factor = torch.randn(dim, dim)
    covariance_prior = factor @ factor.mT + torch.eye(dim)
    precision_prior = torch.linalg.inv(covariance_prior)
    mean_gradient = torch.randn(dim)

    with pytest.raises(ValueError, match="finite minimizer"):
        argmin_proximal_kl(
            lambda theta: (
                (mean_gradient * theta[0]).sum() + (-torch.eye(dim) * theta[1]).sum()
            ),
            (mean_prior, precision_prior),
            parametrization="precision",
        )


def test_argmin_proximal_kl_log_cholesky_raises_when_diagonal_gradient_is_too_large() -> (
    None
):
    r"""Test that the log-Cholesky update rejects ill-posed diagonal gradients."""
    dim = 4
    mean_prior = torch.randn(dim)
    factor = torch.randn(dim, dim)
    covariance_prior = factor @ factor.mT + torch.eye(dim)
    chol_prior = torch.linalg.cholesky(covariance_prior)
    log_chol_prior = chol_prior.tril(diagonal=-1) + torch.diag_embed(
        chol_prior.diagonal(dim1=-2, dim2=-1).log()
    )
    mean_gradient = torch.randn(dim)
    gamma = torch.tensor(1.0)
    diagonal_gradient = gamma + 0.1 + torch.rand(dim)
    gradient_log_cholesky = torch.diag_embed(diagonal_gradient)

    with pytest.raises(ValueError, match="finite minimizer"):
        argmin_proximal_kl(
            lambda theta: (
                (mean_gradient * theta[0]).sum()
                + (gradient_log_cholesky * theta[1]).sum()
            ),
            (mean_prior, log_chol_prior),
            gamma=gamma,
            parametrization="log-cholesky",
        )


def test_argmin_proximal_kl_rejects_unknown_parametrization() -> None:
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


def test_fisher_rejects_unknown_parametrization() -> None:
    r"""Test that the public Fisher dispatch rejects unknown parametrizations."""
    dim = 4
    mean = torch.randn(dim)
    factor = torch.randn(dim, dim)
    covariance = factor @ factor.mT + torch.eye(dim)

    with pytest.raises(ValueError, match="'unknown' is not a valid CovarianceType"):
        fisher((mean, covariance), (mean, covariance), parametrization="unknown")


def test_inverse_fisher_rejects_unknown_parametrization() -> None:
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


def test_log_prob_rejects_unknown_parametrization() -> None:
    r"""Test that the public log-density dispatch rejects unknown parametrizations."""
    dim = 4
    mean = torch.randn(dim)
    factor = torch.randn(dim, dim)
    covariance = factor @ factor.mT + torch.eye(dim)

    with pytest.raises(ValueError, match="'unknown' is not a valid CovarianceType"):
        log_prob(mean, (mean, covariance), parametrization="unknown")
