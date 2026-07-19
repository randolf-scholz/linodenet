r"""Tests for discrete Kalman filtering."""

import pytest
import torch
from torch.testing import assert_close

from linodenet_models import DiscreteKalmanFilter
from linodenet_models.kalman_filter import marginal_gaussian_log_prob


def make_model() -> DiscreteKalmanFilter:
    r"""Instantiate a small stable discrete Kalman filter."""
    torch.manual_seed(0)
    return DiscreteKalmanFilter(
        3,
        5,
        system_matrix=0.1 * torch.randn(5, 5),
        observation_matrix=torch.randn(3, 5),
        process_covariance=0.2,
        measurement_covariance=0.5,
        initial_mean=torch.randn(5),
        initial_covariance=2.0,
    )


def test_forward_returns_posterior_latent_states() -> None:
    r"""Check forward filters the event stream and yields latent posteriors."""
    model = make_model()
    steps = torch.tensor([0, 1, 3])
    context_values = torch.randn(3, 3)
    context_mask = torch.tensor(
        [
            [True, True, False],
            [False, False, False],
            [True, False, True],
        ]
    )
    context_values = context_values.masked_fill(~context_mask, torch.nan)
    query_mask = torch.zeros_like(context_mask)

    means, covs = model.forward(
        steps=steps,
        context_values=context_values,
        context_mask=context_mask,
        query_mask=query_mask,
    )

    assert means.shape == (3, 5)
    assert covs.shape == (3, 5, 5)
    assert means[context_mask.any(dim=-1)].isfinite().all()
    assert covs[context_mask.any(dim=-1)].isfinite().all()
    assert means[~context_mask.any(dim=-1)].isnan().all()
    assert covs[~context_mask.any(dim=-1)].isnan().all()
    assert_close(means, model.posterior_latent_means, equal_nan=True)
    assert_close(covs, model.posterior_latent_covariances, equal_nan=True)


def test_predict_returns_query_marginals() -> None:
    r"""Check predict decodes posterior query states in observation space."""
    model = make_model()
    context_steps = torch.tensor([0, 1, 2])
    context_values = torch.randn(3, 3)
    context_mask = torch.tensor(
        [
            [True, False, True],
            [True, True, False],
            [False, False, False],
        ]
    )
    context_values = context_values.masked_fill(~context_mask, torch.nan)
    query_steps = torch.tensor([1, 3])
    query_mask = torch.tensor([[False, True, False], [True, False, True]])

    mean, cov = model.predict(
        query_steps=query_steps,
        query_mask=query_mask,
        context_steps=context_steps,
        context_values=context_values,
        context_mask=context_mask,
    )

    assert mean.shape == (2, 3)
    assert cov.shape == (2, 3, 3)
    assert mean.isfinite().all()
    assert cov.isfinite().all()
    assert_close(mean, model.pred_means)
    assert_close(cov, model.pred_covs)


def test_sample_and_log_prob_consistent() -> None:
    r"""Check joint sampling/scoring matches standalone masked scoring."""
    model = make_model()
    context_steps = torch.tensor([[0, 1, 2], [0, 2, 0]])
    context_values = torch.randn(2, 3, 3)
    context_mask = torch.tensor(
        [
            [[True, False, True], [False, False, False], [True, True, True]],
            [[True, True, False], [True, False, True], [False, False, False]],
        ]
    )
    context_values = context_values.masked_fill(~context_mask, torch.nan)
    query_steps = torch.tensor([[1, 3], [2, 4]])
    query_mask = torch.tensor(
        [
            [[False, True, False], [True, False, True]],
            [[False, True, False], [True, True, False]],
        ]
    )

    samples, log_prob = model.sample_and_log_prob(
        (2, 3),
        query_steps=query_steps,
        query_mask=query_mask,
        context_steps=context_steps,
        context_values=context_values,
        context_mask=context_mask,
    )
    expected = marginal_gaussian_log_prob(
        samples,
        mean=model.pred_means.expand(2, 3, *model.pred_means.shape),
        cov=model.pred_covs.expand(2, 3, *model.pred_covs.shape),
        mask=query_mask.expand(2, 3, *query_mask.shape),
    )

    assert samples.shape == (2, 3, 2, 2, 3)
    assert log_prob.shape == (2, 3, 2, 2)
    assert samples[..., query_mask].isfinite().all()
    assert samples[..., ~query_mask].isnan().all()
    assert_close(log_prob, expected)


def test_predict_rejects_float_step_indices() -> None:
    r"""Check the discrete API rejects continuous-time step inputs."""
    model = make_model()
    context_values = torch.randn(2, 3)
    context_mask = torch.ones_like(context_values, dtype=torch.bool)
    query_mask = torch.ones(1, 3, dtype=torch.bool)

    with pytest.raises(TypeError, match="Long tensors"):
        model.predict(
            query_steps=torch.tensor([2.0]),
            query_mask=query_mask,
            context_steps=torch.tensor([0.0, 1.0]),
            context_values=context_values,
            context_mask=context_mask,
        )
