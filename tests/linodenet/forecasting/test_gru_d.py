r"""Tests for GRU-D forecasting."""

from typing import ClassVar, NamedTuple

import pytest
import torch
from torch.nn import functional as F

from linodenet.forecasting.gru_d import GRU_D

from .base import SequentialData, TestForecastingModel


class GRUDTestConfig(NamedTuple):
    r"""Configuration used by shared GRU-D forecasting-model tests."""

    input_size: int
    hidden_size: int
    output_size: int


class TestGRU_D(TestForecastingModel[GRU_D]):
    r"""Shared forecasting-model tests for GRU-D."""

    CONTEXT_SHAPE: ClassVar[tuple[int, ...]] = (4,)
    OUTPUT_SHAPE: ClassVar[tuple[int, ...]] = CONTEXT_SHAPE
    STANDARD_CONFIG: ClassVar[GRUDTestConfig] = GRUDTestConfig(
        input_size=CONTEXT_SHAPE[0],
        hidden_size=6,
        output_size=OUTPUT_SHAPE[0],
    )

    @pytest.fixture
    def model_config(self) -> GRUDTestConfig:
        r"""Configuration used to instantiate the GRU-D model under test."""
        return self.STANDARD_CONFIG

    def make_model(self, model_config: object, /) -> GRU_D:
        r"""Instantiate a GRU-D model from :attr:`STANDARD_CONFIG`."""
        if not isinstance(model_config, GRUDTestConfig):
            raise TypeError("model_config must be a GRUDTestConfig.")
        return GRU_D(
            model_config.input_size,
            model_config.hidden_size,
            empirical_mean=torch.zeros(model_config.input_size),
            output_size=model_config.output_size,
        )

    def forecast(
        self,
        model: GRU_D,
        inputs: SequentialData,
        /,
    ) -> tuple[torch.Tensor, ...]:
        r"""Return GRU-D predictions for sequential forecasting inputs."""
        predictions = model(
            inputs.context_times,
            inputs.context_values,
            inputs.query_times,
        )

        assert predictions.shape == inputs.query_values.shape
        assert predictions[inputs.query_mask].isfinite().all()
        assert predictions[~inputs.query_mask].isnan().all()
        return (predictions,)

    def loss(
        self,
        model: GRU_D,
        predictions: tuple[torch.Tensor, ...],
        targets: torch.Tensor,
    ) -> torch.Tensor:
        r"""Return mean squared error for GRU-D predictions."""
        del model
        (forecast,) = predictions
        mask = targets.isfinite()
        return F.mse_loss(forecast[mask], targets[mask])
