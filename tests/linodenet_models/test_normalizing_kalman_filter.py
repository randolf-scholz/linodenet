r"""Tests for the NKF model."""

from typing import ClassVar, NamedTuple

import pytest
import torch
from torch.nn import functional as F

from linodenet.mappings.transforms.scalar import Sinh
from linodenet_models import NormalizingKalmanFilter
from linodenet_models.utils import SplitTimeData

from .base import TestDiscreteTimeModel


class NormalizingKalmanFilterTestConfig(NamedTuple):
    r"""Configuration used by shared discrete NKF tests."""

    input_size: int
    hidden_size: int


class TestNormalizingKalmanFilter(TestDiscreteTimeModel[NormalizingKalmanFilter]):
    r"""Shared forecasting-model tests for normalizing Kalman filters."""

    CONTEXT_SHAPE: ClassVar[tuple[int, ...]] = (3,)
    OUTPUT_SHAPE: ClassVar[tuple[int, ...]] = CONTEXT_SHAPE
    STANDARD_CONFIG: ClassVar[NormalizingKalmanFilterTestConfig] = (
        NormalizingKalmanFilterTestConfig(
            input_size=CONTEXT_SHAPE[0],
            hidden_size=5,
        )
    )

    @pytest.fixture
    def model_config(self) -> NormalizingKalmanFilterTestConfig:
        r"""Configuration used to instantiate the NKF under test."""
        return self.STANDARD_CONFIG

    @pytest.fixture(params=[False, True], ids=["no_missingness", "input_missingness"])
    def input_missingness(self, request: pytest.FixtureRequest) -> bool:
        r"""Whether to randomly mask half of the context values with NaN."""
        return request.param

    def make_model(self, model_config: object, /) -> NormalizingKalmanFilter:
        r"""Instantiate an NKF from :attr:`STANDARD_CONFIG`."""
        if not isinstance(model_config, NormalizingKalmanFilterTestConfig):
            raise TypeError("model_config must be a NormalizingKalmanFilterTestConfig.")

        input_size = model_config.input_size
        hidden_size = model_config.hidden_size
        return NormalizingKalmanFilter(
            input_size,
            hidden_size,
            decoder=Sinh(),
            system_matrix=0.1 * torch.randn(hidden_size, hidden_size),
            observation_matrix=torch.randn(input_size, hidden_size),
            process_covariance=0.2,
            measurement_covariance=0.5,
            initial_mean=torch.randn(hidden_size),
            initial_covariance=2.0,
            learnable=True,
        )

    def forecast(
        self,
        model: NormalizingKalmanFilter,
        inputs: SplitTimeData,
        /,
    ) -> tuple[torch.Tensor, ...]:
        r"""Return NKF predictions for sequential forecasting inputs."""
        assert inputs.target_values is not None
        pred_mean, pred_scale = model.predict(
            context_steps=inputs.context_times,
            context_values=inputs.context_values,
            context_mask=inputs.context_mask,
            query_steps=inputs.query_times,
            query_mask=inputs.query_mask,
        )

        assert pred_mean.shape == inputs.target_values.shape
        assert pred_scale.shape == inputs.target_values.shape
        assert pred_mean[inputs.query_mask].isfinite().all()
        assert pred_scale[inputs.query_mask].isfinite().all()
        assert pred_mean[~inputs.query_mask].isnan().all()
        assert pred_scale[~inputs.query_mask].isnan().all()

        return pred_mean, pred_scale

    def loss(
        self,
        model: NormalizingKalmanFilter,
        predictions: tuple[torch.Tensor, ...],
        targets: torch.Tensor,
    ) -> torch.Tensor:
        r"""Return a supervised loss for NKF point summaries."""
        del model
        pred_mean, pred_scale = predictions
        mask = targets.isfinite()
        mean_loss = F.mse_loss(pred_mean[mask], targets[mask])
        scale_loss = F.mse_loss(pred_scale[mask], torch.ones_like(pred_scale[mask]))
        return 100.0 * (mean_loss + 1e-3 * scale_loss)
