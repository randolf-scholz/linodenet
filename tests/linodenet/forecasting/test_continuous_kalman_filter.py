r"""Tests for continuous Kalman filtering."""

from typing import ClassVar, NamedTuple

import pytest
import torch
from torch.nn import functional as F

from linodenet.forecasting.continuous_kalman_filter import ContinuousKalmanFilter

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
        pred_mean, pred_covariance = model(
            inputs.query_times,
            (inputs.context_times, inputs.context_values),
        )

        assert pred_mean.shape == inputs.query_values.shape
        assert pred_covariance.shape == (
            *inputs.query_times.shape,
            model.input_size,
            model.input_size,
        )
        assert pred_mean[inputs.query_mask].isfinite().all()
        assert pred_covariance[inputs.query_mask].isfinite().all()
        assert pred_mean[~inputs.query_mask].isnan().all()
        assert pred_covariance[~inputs.query_mask].isnan().all()
        pred_variance = pred_covariance.diagonal(dim1=-2, dim2=-1)
        assert pred_variance.shape == inputs.query_values.shape
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
