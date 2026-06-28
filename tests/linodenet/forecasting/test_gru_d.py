r"""Tests for GRU-D forecasting."""

from typing import ClassVar, NamedTuple

import pytest
import torch
from torch import nan
from torch.nn import functional as F

from linodenet.forecasting.gru_d import GRU_D
from linodenet.forecasting.utils import ForecastingRequest

from .base import TestForecastingModel


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

    @pytest.fixture(params=[False, True], ids=["no_missingness", "input_missingness"])
    def input_missingness(self, request: pytest.FixtureRequest) -> bool:
        r"""Whether to randomly mask half of the context values with NaN."""
        return request.param

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
        inputs: ForecastingRequest,
        /,
    ) -> tuple[torch.Tensor, ...]:
        r"""Return GRU-D predictions for sequential forecasting inputs."""
        assert inputs.target_values is not None
        query_valid = inputs.query_mask.any(dim=-1)

        combined_pred = model.predict(
            context_times=inputs.context_times,
            context_values=inputs.context_values,
            context_mask=inputs.context_mask,
            query_times=inputs.query_times,
            query_mask=inputs.query_mask,
        )  # (..., N+K, F)

        query_steps = inputs.query_mask.any(dim=-1)  # (..., N+K)
        pred = torch.full_like(inputs.target_values, nan)
        pred[query_valid] = combined_pred[query_steps]

        assert pred.shape == inputs.target_values.shape
        assert pred[query_valid].isfinite().all()
        assert pred[~query_valid].isnan().all()
        return (pred,)

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
