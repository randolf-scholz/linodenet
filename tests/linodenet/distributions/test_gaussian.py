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

    actual = kl((mean_p, cov_p), (mean_q, cov_q))
    actual_cholesky = kl(
        (mean_p, chol_p),
        (mean_q, chol_q),
        parametrization="cholesky",
    )
    expected = kl_divergence(
        MultivariateNormal(mean_p, covariance_matrix=cov_p),
        MultivariateNormal(mean_q, covariance_matrix=cov_q),
    )

    assert torch.allclose(actual, expected)
    assert torch.allclose(actual_cholesky, expected)


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


@pytest.mark.parametrize("parametrization", ["covariance", "precision", "cholesky"])
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
        case _:
            raise AssertionError(f"Unexpected parametrization {parametrization!r}.")

    transported = fisher(theta, tangent, parametrization=parametrization)
    recovered = inverse_fisher(theta, transported, parametrization=parametrization)

    assert (recovered[0] - tangent[0]).abs().amax() < 1e-5
    assert (recovered[1] - tangent[1]).abs().amax() < 1e-5


def test_argmin_proximal_kl_solves_closed_form_problem() -> None:
    r"""Test the closed-form KL-proximal Gaussian update."""
    batch_shape = (2, 3)
    dim = 4
    lambda_ = torch.tensor(1.7)

    mean_prior = torch.randn(*batch_shape, dim)
    factor = torch.randn(*batch_shape, dim, dim)
    covariance_prior = factor @ factor.mT + torch.eye(dim)
    precision_prior = torch.cholesky_inverse(torch.linalg.cholesky(covariance_prior))

    g = torch.randn(*batch_shape, dim)
    precision_shift_factor = torch.randn(*batch_shape, dim, dim)
    precision_shift = precision_shift_factor @ precision_shift_factor.mT
    gradient_covariance = 0.5 * lambda_ * precision_shift

    mean_post, covariance_post = argmin_proximal_kl(
        (g, gradient_covariance),
        (mean_prior, covariance_prior),
        lambda_=lambda_,
    )

    expected_mean = mean_prior - torch.einsum(
        "...ij,...j->...i", covariance_prior, g / lambda_
    )
    expected_precision = 0.5 * (
        precision_prior + precision_shift + (precision_prior + precision_shift).mT
    )
    expected_covariance = torch.cholesky_inverse(
        torch.linalg.cholesky(expected_precision)
    )

    assert torch.allclose(mean_post, expected_mean)
    assert (covariance_post - expected_covariance).abs().amax() < 3e-5

    mean_var = mean_post.detach().clone().requires_grad_(True)
    covariance_var = covariance_post.detach().clone().requires_grad_(True)
    objective = (
        (g * (mean_var - mean_prior)).sum()
        + (gradient_covariance * (covariance_var - covariance_prior)).sum()
        + lambda_ * kl((mean_var, covariance_var), (mean_prior, covariance_prior)).sum()
    )
    mean_grad, covariance_grad = torch.autograd.grad(
        objective,
        (mean_var, covariance_var),
    )

    assert torch.allclose(mean_grad, torch.zeros_like(mean_grad), atol=1e-6, rtol=1e-6)
    assert _symmetric(covariance_grad).abs().amax() < 1e-5


def test_argmin_proximal_kl_raises_when_precision_is_not_pd() -> None:
    r"""Test that the proximal update rejects non-positive-definite precision."""
    dim = 4
    mean_prior = torch.randn(dim)
    factor = torch.randn(dim, dim)
    covariance_prior = factor @ factor.mT + torch.eye(dim)
    precision_prior = torch.linalg.inv(covariance_prior)

    with pytest.raises(ValueError, match="not positive definite"):
        argmin_proximal_kl(
            (torch.randn(dim), -precision_prior),
            (mean_prior, covariance_prior),
        )


def test_fisher_rejects_unknown_parametrization() -> None:
    r"""Test that the public Fisher dispatch rejects unknown parametrizations."""
    dim = 4
    mean = torch.randn(dim)
    factor = torch.randn(dim, dim)
    covariance = factor @ factor.mT + torch.eye(dim)

    with pytest.raises(ValueError, match="Expected parametrization"):
        fisher((mean, covariance), (mean, covariance), parametrization="unknown")


def test_inverse_fisher_rejects_unknown_parametrization() -> None:
    r"""Test that the public inverse Fisher dispatch rejects unknown parametrizations."""
    dim = 4
    mean = torch.randn(dim)
    factor = torch.randn(dim, dim)
    covariance = factor @ factor.mT + torch.eye(dim)

    with pytest.raises(ValueError, match="Expected parametrization"):
        inverse_fisher(
            (mean, covariance),
            (mean, covariance),
            parametrization="unknown",
        )
