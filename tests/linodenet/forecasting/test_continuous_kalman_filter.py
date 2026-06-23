r"""Tests for continuous Kalman filtering."""

from typing import ClassVar, NamedTuple

import pytest
import torch
from torch.distributions import MultivariateNormal
from torch.nn import functional as F
from torch.testing import assert_close

from linodenet.forecasting.continuous_kalman_filter import (
    ContinuousKalmanFilter,
    marginal_gaussian_log_prob,
    marginal_gaussian_sample,
    marginal_gaussian_sample_and_log_prob,
)
from linodenet.forecasting.utils import BatchedCombinedArgs, BatchedDenseArgs

from .base import SequentialData, TestForecastingModel


def test_marginal_gaussian_log_prob_matches_explicit_subvectors() -> None:
    r"""Check masked Gaussian scoring equals explicit marginal scoring."""
    mean = torch.tensor([[1.0, -2.0, 0.5], [0.0, 3.0, -1.0]])
    values = torch.tensor([[1.5, 10.0, -0.25], [99.0, 2.5, 10.0]])
    cov = torch.tensor(
        [
            [
                [2.0, 0.3, 0.4],
                [0.3, 1.0, 0.2],
                [0.4, 0.2, 1.5],
            ],
            [
                [1.0, 0.1, 0.2],
                [0.1, 2.0, 0.5],
                [0.2, 0.5, 3.0],
            ],
        ]
    )
    mask = torch.tensor([[True, False, True], [False, True, False]])

    actual = marginal_gaussian_log_prob(values, mean=mean, cov=cov, mask=mask)
    expected = torch.stack(
        [
            MultivariateNormal(
                mean[0, [0, 2]],
                covariance_matrix=cov[0][[0, 2]][:, [0, 2]],
            ).log_prob(values[0, [0, 2]]),
            MultivariateNormal(
                mean[1, [1]],
                covariance_matrix=cov[1][[1]][:, [1]],
            ).log_prob(values[1, [1]]),
        ]
    )

    assert_close(actual, expected)


def test_marginal_gaussian_log_prob_zero_for_empty_mask() -> None:
    r"""Check empty marginals have unit likelihood."""
    values = torch.full((2, 3), torch.nan)
    mean = torch.full((2, 3), torch.nan)
    cov = torch.full((2, 3, 3), torch.nan)
    mask = torch.zeros(2, 3, dtype=torch.bool)

    actual = marginal_gaussian_log_prob(values, mean=mean, cov=cov, mask=mask)

    assert_close(actual, torch.zeros(2))


def test_marginal_gaussian_sample_uses_masked_marginals() -> None:
    r"""Check masked Gaussian samples have sample shape and NaN padding."""
    torch.manual_seed(0)
    mean = torch.tensor([[1.0, -2.0, 0.5], [0.0, 3.0, -1.0]])
    cov = torch.tensor(
        [
            [
                [2.0, 0.3, 0.4],
                [0.3, 1.0, 0.2],
                [0.4, 0.2, 1.5],
            ],
            [
                [1.0, 0.1, 0.2],
                [0.1, 2.0, 0.5],
                [0.2, 0.5, 3.0],
            ],
        ]
    )
    mask = torch.tensor([[True, False, True], [False, True, False]])

    samples = marginal_gaussian_sample((2, 3), mean=mean, cov=cov, mask=mask)
    log_prob = marginal_gaussian_log_prob(
        samples,
        mean=mean.expand(2, 3, *mean.shape),
        cov=cov.expand(2, 3, *cov.shape),
        mask=mask.expand(2, 3, *mask.shape),
    )

    assert samples.shape == (2, 3, *mean.shape)
    assert samples[..., mask].isfinite().all()
    assert samples[..., ~mask].isnan().all()
    assert log_prob.isfinite().all()


def test_marginal_gaussian_sample_and_log_prob_reuses_samples() -> None:
    r"""Check joint sampling/scoring matches standalone masked scoring."""
    torch.manual_seed(0)
    mean = torch.tensor([[1.0, -2.0, 0.5], [0.0, 3.0, -1.0]])
    cov = torch.tensor(
        [
            [
                [2.0, 0.3, 0.4],
                [0.3, 1.0, 0.2],
                [0.4, 0.2, 1.5],
            ],
            [
                [1.0, 0.1, 0.2],
                [0.1, 2.0, 0.5],
                [0.2, 0.5, 3.0],
            ],
        ]
    )
    mask = torch.tensor([[True, False, True], [False, True, False]])

    samples, log_prob = marginal_gaussian_sample_and_log_prob(
        (2, 3),
        mean=mean,
        cov=cov,
        mask=mask,
    )
    expected = marginal_gaussian_log_prob(
        samples,
        mean=mean.expand(2, 3, *mean.shape),
        cov=cov.expand(2, 3, *cov.shape),
        mask=mask.expand(2, 3, *mask.shape),
    )

    assert samples.shape == (2, 3, *mean.shape)
    assert log_prob.shape == (2, 3, *mean.shape[:-1])
    assert_close(log_prob, expected)


class KalmanFilterTestConfig(NamedTuple):
    r"""Configuration used by shared continuous Kalman filter tests."""

    input_size: int
    hidden_size: int


class TestKalmanFilter(TestForecastingModel[ContinuousKalmanFilter]):
    r"""Shared forecasting-model tests for continuous Kalman filters."""

    CONTEXT_SHAPE: ClassVar[tuple[int, ...]] = (3,)
    OUTPUT_SHAPE: ClassVar[tuple[int, ...]] = CONTEXT_SHAPE
    STANDARD_CONFIG: ClassVar[KalmanFilterTestConfig] = KalmanFilterTestConfig(
        input_size=CONTEXT_SHAPE[0],
        hidden_size=5,
    )

    @pytest.fixture
    def model_config(self) -> KalmanFilterTestConfig:
        r"""Configuration used to instantiate the Kalman filter under test."""
        return self.STANDARD_CONFIG

    @pytest.fixture(params=[False, True], ids=["no_missingness", "input_missingness"])
    def input_missingness(self, request: pytest.FixtureRequest) -> bool:
        r"""Whether to randomly mask half of the context values with NaN."""
        return request.param

    def make_model(self, model_config: object, /) -> ContinuousKalmanFilter:
        r"""Instantiate a continuous Kalman filter from :attr:`STANDARD_CONFIG`."""
        if not isinstance(model_config, KalmanFilterTestConfig):
            raise TypeError("model_config must be a KalmanFilterTestConfig.")

        input_size = model_config.input_size
        hidden_size = model_config.hidden_size
        return ContinuousKalmanFilter(
            input_size,
            hidden_size,
            system_matrix=0.05 * torch.randn(hidden_size, hidden_size),
            observation_matrix=torch.randn(input_size, hidden_size),
            process_noise=0.2,
            measurement_noise=0.5,
            initial_mean=torch.randn(hidden_size),
            initial_covariance=2.0 * torch.eye(hidden_size),
            initial_state_learnable=True,
            process_noise_learnable=True,
            observation_noise_learnable=True,
        )

    def forecast(
        self,
        model: ContinuousKalmanFilter,
        inputs: SequentialData,
        /,
    ) -> tuple[torch.Tensor, ...]:
        r"""Return Kalman filter predictions for sequential forecasting inputs."""
        dense = BatchedDenseArgs(
            context_times=inputs.context_times,
            context_values=inputs.context_values,
            context_mask=inputs.context_values.isfinite(),
            query_times=inputs.query_times,
            query_mask=inputs.query_mask.unsqueeze(-1).expand_as(inputs.query_values),
        )
        combined = dense.to_combined()
        posterior_mean, posterior_covariance = model(
            combined.times,
            combined.context_values,
            combined.context_mask,
            combined.query_mask,
        )
        pred_mean = (
            BatchedCombinedArgs(
                times=combined.times,
                context_values=combined.context_values,
                context_mask=combined.context_mask,
                query_mask=combined.query_mask,
                query_values=posterior_mean.masked_fill(
                    ~combined.query_mask, torch.nan
                ),
            )
            .to_dense()
            .query_values
        )
        posterior_variance = posterior_covariance.diagonal(dim1=-2, dim2=-1)
        pred_variance = (
            BatchedCombinedArgs(
                times=combined.times,
                context_values=combined.context_values,
                context_mask=combined.context_mask,
                query_mask=combined.query_mask,
                query_values=posterior_variance.masked_fill(
                    ~combined.query_mask,
                    torch.nan,
                ),
            )
            .to_dense()
            .query_values
        )

        if pred_mean is None or pred_variance is None:
            raise RuntimeError("Expected Kalman filter query predictions.")

        query_size = inputs.query_values.shape[-2]
        if pred_mean.shape[-2] < query_size:
            padding = query_size - pred_mean.shape[-2]
            pred_mean = torch.cat(
                [
                    pred_mean,
                    pred_mean.new_full(
                        (*pred_mean.shape[:-2], padding, pred_mean.shape[-1]), torch.nan
                    ),
                ],
                dim=-2,
            )
            pred_variance = torch.cat(
                [
                    pred_variance,
                    pred_variance.new_full(
                        (*pred_variance.shape[:-2], padding, pred_variance.shape[-1]),
                        torch.nan,
                    ),
                ],
                dim=-2,
            )

        assert pred_mean.shape == inputs.query_values.shape
        assert pred_variance.shape == inputs.query_values.shape
        assert pred_mean[inputs.query_mask].isfinite().all()
        assert pred_variance[inputs.query_mask].isfinite().all()
        assert posterior_covariance[combined.query_mask.any(dim=-1)].isfinite().all()
        assert pred_mean[~inputs.query_mask].isnan().all()
        assert pred_variance[~inputs.query_mask].isnan().all()
        return pred_mean, pred_variance

    def loss(
        self,
        model: ContinuousKalmanFilter,
        predictions: tuple[torch.Tensor, ...],
        targets: torch.Tensor,
    ) -> torch.Tensor:
        r"""Return a supervised loss for Kalman filter predictions."""
        del model
        pred_mean, pred_variance = predictions
        mask = targets.isfinite().all(dim=-1)
        mean_loss = F.mse_loss(pred_mean[mask], targets[mask])
        variance_loss = F.mse_loss(
            pred_variance[mask],
            torch.ones_like(pred_variance[mask]),
        )
        return mean_loss + 1e-3 * variance_loss

    def test_query_only_steps_do_not_update_latent_state(self) -> None:
        r"""Check all-missing observations leave the propagated state unchanged."""
        torch.manual_seed(0)
        model = self.make_model(self.STANDARD_CONFIG)
        times = torch.tensor([0.0, 0.5, 1.0])
        values = torch.full((3, self.STANDARD_CONFIG.input_size), torch.nan)
        context_mask = torch.zeros_like(values, dtype=torch.bool)
        query_mask = torch.ones_like(values, dtype=torch.bool)

        model(times, values, context_mask, query_mask)

        assert_close(model.post_latent_means, model.prior_latent_means)
        assert_close(model.post_latent_covs, model.prior_latent_covs)

    def test_default_system_matrix_is_skew_symmetric(self) -> None:
        r"""Check default continuous-time dynamics are norm-preserving."""
        torch.manual_seed(0)
        model = ContinuousKalmanFilter(
            self.STANDARD_CONFIG.input_size,
            self.STANDARD_CONFIG.hidden_size,
        )

        skew_residual = model.system_matrix + model.system_matrix.mT
        assert_close(skew_residual, torch.zeros_like(skew_residual))

    def test_initial_time_defaults_to_first_time_step(self) -> None:
        r"""Check default initial time matches the first time step."""
        torch.manual_seed(0)
        model = self.make_model(self.STANDARD_CONFIG)
        times = torch.tensor([0.0, 0.5, 1.0])
        values = torch.randn(3, self.STANDARD_CONFIG.input_size)
        context_mask = torch.ones_like(values, dtype=torch.bool)
        query_mask = torch.ones_like(values, dtype=torch.bool)

        default_mean, default_cov = model(times, values, context_mask, query_mask)
        explicit_mean, explicit_cov = model(
            times,
            values,
            context_mask,
            query_mask,
            initial_time=times[0],
        )

        assert_close(explicit_mean, default_mean)
        assert_close(explicit_cov, default_cov)

    def test_log_prob_returns_time_marginal_likelihoods(self) -> None:
        r"""Check Kalman log-probabilities are returned per time step."""
        torch.manual_seed(0)
        model = self.make_model(self.STANDARD_CONFIG)
        dim = self.STANDARD_CONFIG.input_size
        times = torch.tensor([0.0, 0.5, 1.0, 1.5])
        context_values = torch.randn(4, dim)
        context_mask = torch.tensor(
            [
                [True, True, False],
                [True, False, True],
                [False, False, False],
                [True, True, True],
            ]
        )
        context_values = context_values.masked_fill(~context_mask, torch.nan)
        query_mask = torch.tensor(
            [
                [False, False, False],
                [False, True, False],
                [True, False, True],
                [False, False, False],
            ]
        )
        values = torch.randn(4, dim).masked_fill(~query_mask, torch.nan)

        log_prob = model.log_prob(
            values,
            times=times,
            context_values=context_values,
            context_mask=context_mask,
            query_mask=query_mask,
        )
        mean, cov = model(
            times,
            context_values,
            context_mask,
            query_mask,
        )
        expected = marginal_gaussian_log_prob(
            values,
            mean=mean,
            cov=cov,
            mask=query_mask,
        )

        assert log_prob.shape == times.shape
        assert_close(log_prob, expected)
        assert_close(log_prob[~query_mask.any(dim=-1)], torch.zeros(2))

    def test_sample_returns_time_marginal_samples(self) -> None:
        r"""Check Kalman samples have the requested sample and query shape."""
        torch.manual_seed(0)
        model = self.make_model(self.STANDARD_CONFIG)
        dim = self.STANDARD_CONFIG.input_size
        times = torch.tensor([0.0, 0.5, 1.0])
        context_values = torch.randn(3, dim)
        context_mask = torch.tensor(
            [
                [True, False, True],
                [False, False, False],
                [True, True, False],
            ]
        )
        context_values = context_values.masked_fill(~context_mask, torch.nan)
        query_mask = torch.tensor(
            [
                [False, True, False],
                [True, False, True],
                [False, False, False],
            ]
        )

        samples = model.sample(
            5,
            times=times,
            context_values=context_values,
            context_mask=context_mask,
            query_mask=query_mask,
        )

        assert samples.shape == (5, *context_values.shape)
        assert samples[:, query_mask].isfinite().all()
        assert samples[:, ~query_mask].isnan().all()

    def test_sample_and_log_prob_returns_time_marginal_likelihoods(self) -> None:
        r"""Check Kalman samples and sample log-probabilities use query masks."""
        torch.manual_seed(0)
        model = self.make_model(self.STANDARD_CONFIG)
        dim = self.STANDARD_CONFIG.input_size
        times = torch.tensor([[0.0, 0.5, 1.0], [0.0, 0.25, 0.75]])
        context_values = torch.randn(2, 3, dim)
        context_mask = torch.tensor(
            [
                [[True, False, True], [False, False, False], [True, True, True]],
                [[True, True, False], [True, False, True], [False, False, False]],
            ]
        )
        context_values = context_values.masked_fill(~context_mask, torch.nan)
        query_mask = torch.tensor(
            [
                [[False, True, False], [True, False, True], [False, False, False]],
                [[False, False, False], [False, True, False], [True, True, False]],
            ]
        )

        samples, log_prob = model.sample_and_log_prob(
            (2, 3),
            times=times,
            context_values=context_values,
            context_mask=context_mask,
            query_mask=query_mask,
        )
        expected = marginal_gaussian_log_prob(
            samples,
            mean=model.post_target_means.expand(2, 3, *model.post_target_means.shape),
            cov=model.post_target_covs.expand(2, 3, *model.post_target_covs.shape),
            mask=query_mask.expand(2, 3, *query_mask.shape),
        )

        assert samples.shape == (2, 3, *context_values.shape)
        assert log_prob.shape == (2, 3, *times.shape)
        assert samples[..., query_mask].isfinite().all()
        assert samples[..., ~query_mask].isnan().all()
        assert_close(log_prob, expected)
        assert_close(
            log_prob[..., ~query_mask.any(dim=-1)],
            torch.zeros_like(log_prob[..., ~query_mask.any(dim=-1)]),
        )

    @pytest.mark.parametrize("size", [(), (5,), (1, 2, 3)])
    def test_sample_and_log_prob_consistent(self, size: tuple[int, ...]) -> None:
        torch.manual_seed(0)
        model = self.make_model(self.STANDARD_CONFIG)
        D = self.STANDARD_CONFIG.input_size
        times = torch.tensor([0.0, 0.5, 1.0, 1.5])
        context_values = torch.randn(4, D)
        context_mask = torch.tensor([[True] * D, [True] * D, [False] * D, [True] * D])
        context_values = context_values.masked_fill(~context_mask, torch.nan)
        query_mask = torch.zeros(4, D, dtype=torch.bool)
        query_mask[2] = True
        query_mask[3] = True

        samples, log_prob_direct = model.sample_and_log_prob(
            size,
            times=times,
            context_values=context_values,
            context_mask=context_mask,
            query_mask=query_mask,
        )
        log_prob_via_sample = model.log_prob(
            samples,
            times=times,
            context_values=context_values,
            context_mask=context_mask,
            query_mask=query_mask,
        )

        assert_close(log_prob_direct, log_prob_via_sample)
