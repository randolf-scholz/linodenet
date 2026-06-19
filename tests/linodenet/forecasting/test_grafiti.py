r"""Tests for GraFITi components."""

from typing import ClassVar, NamedTuple

import pytest
import torch
from torch.nn import functional as F
from torch.testing import assert_close

from linodenet.forecasting.grafiti import Grafiti
from linodenet.forecasting.utils import BatchedTripletArgs

from .base import SequentialData, TestForecastingModel


class GrafitiTestConfig(NamedTuple):
    r"""Configuration used by shared GraFITi forecasting-model tests."""

    input_dim: int
    hidden_dim: int
    num_layers: int
    num_heads: int


class TestModel(TestForecastingModel[Grafiti]):
    r"""Shared forecasting-model tests for GraFITi."""

    CONTEXT_SHAPE: ClassVar[tuple[int, ...]] = (1,)
    OUTPUT_SHAPE: ClassVar[tuple[int, ...]] = (8,)
    BATCH_SHAPE: ClassVar[tuple[int, ...]] = (4,)
    STANDARD_CONFIG: ClassVar[GrafitiTestConfig] = GrafitiTestConfig(
        input_dim=CONTEXT_SHAPE[0],
        hidden_dim=OUTPUT_SHAPE[0],
        num_layers=2,
        num_heads=2,
    )

    @pytest.fixture
    def model_config(self) -> GrafitiTestConfig:
        r"""Configuration used to instantiate the GraFITi model under test."""
        return self.STANDARD_CONFIG

    def make_model(self, model_config: object, /) -> Grafiti:
        r"""Instantiate a GraFITi model from :attr:`STANDARD_CONFIG`."""
        if not isinstance(model_config, GrafitiTestConfig):
            raise TypeError("model_config must be a GrafitiTestConfig.")
        return Grafiti(
            input_dim=model_config.input_dim,
            hidden_dim=model_config.hidden_dim,
            num_layers=model_config.num_layers,
            num_heads=model_config.num_heads,
        )

    def forecast(
        self,
        model: Grafiti,
        inputs: SequentialData,
        /,
    ) -> tuple[torch.Tensor, ...]:
        r"""Return GraFITi target embeddings for sequential forecasting inputs."""
        time_points = torch.cat([inputs.context_times, inputs.query_times], dim=-1)
        time_points = time_points.nan_to_num(0.0)
        context_nan = inputs.context_values.new_full(
            (*inputs.query_times.shape, inputs.context_values.shape[-1]),
            torch.nan,
        )
        context_values = torch.cat([inputs.context_values, context_nan], dim=-2)
        target_mask = torch.cat(
            [
                torch.zeros_like(inputs.context_values, dtype=torch.bool),
                inputs.query_mask.unsqueeze(dim=-1),
            ],
            dim=-2,
        )
        embeddings = model(time_points, context_values, target_mask)

        assert embeddings.shape == inputs.query_values.shape
        assert embeddings[inputs.query_mask].isfinite().all()
        return (embeddings,)

    def loss(
        self,
        model: Grafiti,
        predictions: tuple[torch.Tensor, ...],
        targets: torch.Tensor,
    ) -> torch.Tensor:
        r"""Return mean squared error for GraFITi target embeddings."""
        (embeddings,) = predictions
        mask = targets.isfinite()
        regularizer = embeddings.new_zeros(())
        for parameter in model.parameters():
            regularizer = regularizer + parameter.sum()
        return F.mse_loss(embeddings[mask], targets[mask]) + 1e-3 * regularizer


def test_grafiti_triplet_matches_combined_forward() -> None:
    r"""Check that sparse and combined GraFITi inputs produce the same embeddings."""
    torch.manual_seed(0)
    model = Grafiti(input_dim=3, hidden_dim=8, num_layers=2, num_heads=2)
    args = BatchedTripletArgs(
        context_times=torch.tensor(
            [
                [1.0, 3.0, 5.0, 7.0, torch.nan],
                [0.0, 2.0, 4.0, torch.nan, torch.nan],
            ]
        ),
        context_channels=torch.tensor(
            [
                [0, 2, 1, 0, -1],
                [1, 0, 2, -1, -1],
            ]
        ),
        context_values=torch.tensor(
            [
                [10.0, 32.0, 51.0, 70.0, torch.nan],
                [1.0, 20.0, 22.0, torch.nan, torch.nan],
            ]
        ),
        query_times=torch.tensor(
            [
                [2.0, 4.0, 6.0, torch.nan],
                [1.0, 3.0, 5.0, 7.0],
            ]
        ),
        query_channels=torch.tensor(
            [
                [0, 1, 2, -1],
                [0, 1, 2, 1],
            ]
        ),
        query_values=torch.tensor(
            [
                [200.0, 410.0, 620.0, torch.nan],
                [100.0, 310.0, 520.0, 710.0],
            ]
        ),
    )
    combined = args.to_combined()

    expected = model(
        combined.times,
        combined.context_values,
        combined.query_mask,
    )
    actual = model.forward_triplet(
        args.context_times,
        args.context_channels,
        args.context_values,
        args.query_times,
        args.query_channels,
    )

    assert_close(actual, expected)
