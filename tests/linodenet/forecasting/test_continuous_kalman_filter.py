r"""Tests for continuous Kalman filtering."""

from typing import ClassVar, NamedTuple

import pytest
import torch
from torch.nn import functional as F
from torch.testing import assert_close

from linodenet.forecasting.continuous_kalman_filter import ContinuousKalmanFilter
from linodenet.forecasting.utils import BatchedCombinedArgs, BatchedDenseArgs

from .base import SequentialData, TestForecastingModel


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
            process_covariance=0.2,
            measurement_covariance=0.5,
            initial_mean=torch.randn(hidden_size),
            initial_covariance=2.0 * torch.eye(hidden_size),
            learnable=True,
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

        assert_close(model.posterior_latent_means, model.prior_latent_means)
        assert_close(model.posterior_latent_covariances, model.prior_latent_covariances)
