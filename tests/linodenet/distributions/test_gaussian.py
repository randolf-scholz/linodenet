r"""Test Gaussian distribution utilities."""

import torch
from torch.distributions import MultivariateNormal
from torch.distributions.kl import kl_divergence

from linodenet.distributions.gaussian import kl, kl_cholesky


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
    actual_cholesky = kl_cholesky((mean_p, chol_p), (mean_q, chol_q))
    expected = kl_divergence(
        MultivariateNormal(mean_p, covariance_matrix=cov_p),
        MultivariateNormal(mean_q, covariance_matrix=cov_q),
    )

    assert torch.allclose(actual, expected)
    assert torch.allclose(actual_cholesky, expected)
