r"""Tests for GraFITi components."""

from typing import ClassVar, NamedTuple

import pytest
import torch
from torch import nan
from torch.nn import functional as F
from torch.testing import assert_close

from linodenet_models.grafiti import Grafiti, gather_target_embeddings
from linodenet_models.utils import SplitTimeData

from .base import TestContinuousTimeModel


class GrafitiTestConfig(NamedTuple):
    r"""Configuration used by shared GraFITi forecasting-model tests."""

    input_dim: int
    hidden_dim: int
    num_layers: int
    num_heads: int


class TestGrafiti(TestContinuousTimeModel[Grafiti]):
    r"""Shared forecasting-model tests for GraFITi."""

    CONTEXT_SHAPE: ClassVar[tuple[int, ...]] = (1,)
    OUTPUT_SHAPE: ClassVar[tuple[int, ...]] = CONTEXT_SHAPE
    STANDARD_CONFIG: ClassVar[GrafitiTestConfig] = GrafitiTestConfig(
        input_dim=CONTEXT_SHAPE[0],
        hidden_dim=8,
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
            dim_input=model_config.input_dim,
            dim_latent=model_config.hidden_dim,
            num_layers=model_config.num_layers,
            num_heads=model_config.num_heads,
        )

    def forecast(
        self,
        model: Grafiti,
        inputs: SplitTimeData,
        /,
    ) -> tuple[torch.Tensor, ...]:
        r"""Return GraFITi predictions for sequential forecasting inputs."""
        assert inputs.target_values is not None
        query_valid = inputs.query_mask.any(dim=-1)
        time_points = torch.cat([inputs.context_times, inputs.query_times], dim=-1)
        time_points = time_points.nan_to_num(0.0)
        context_nan = inputs.context_values.new_full(
            (*inputs.query_times.shape, inputs.context_values.shape[-1]),
            nan,
        )
        context_values = torch.cat([inputs.context_values, context_nan], dim=-2)
        context_mask = torch.cat(
            [
                inputs.context_mask,
                torch.zeros_like(context_nan, dtype=torch.bool),
            ],
            dim=-2,
        )
        query_mask = torch.cat(
            [
                torch.zeros_like(inputs.context_values, dtype=torch.bool),
                inputs.query_mask,
            ],
            dim=-2,
        )
        forecasts = model.forward(
            timestamps=time_points,
            context_values=context_values,
            context_mask=context_mask,
            query_mask=query_mask,
        )
        predictions = forecasts[..., inputs.context_values.shape[-2] :, :]

        assert predictions.shape == inputs.target_values.shape
        assert predictions[query_valid].isfinite().all()
        return (predictions,)

    def loss(
        self,
        model: Grafiti,
        predictions: tuple[torch.Tensor, ...],
        targets: torch.Tensor,
    ) -> torch.Tensor:
        r"""Return mean squared error for GraFITi predictions."""
        (forecast,) = predictions
        mask = targets.isfinite()
        regularizer = forecast.new_zeros(())
        for parameter in model.parameters():
            regularizer = regularizer + parameter.sum()
        return F.mse_loss(forecast[mask], targets[mask]) + 1e-3 * regularizer


def test_gather_target_embeddings_unbatched() -> None:
    r"""Check target embedding gathering for an unbatched edge list."""
    x = torch.arange(10.0).reshape(5, 2)
    target_mask = torch.tensor([True, False, True, True, False])

    actual = gather_target_embeddings(x, target_mask=target_mask)
    expected = torch.stack([x[0], x[2], x[3]])

    assert_close(actual, expected)


def test_gather_target_embeddings_batched_pads_to_max_targets() -> None:
    r"""Check batched target gathering pads shorter target lists with NaNs."""
    x = torch.arange(16.0).reshape(2, 4, 2)
    target_mask = torch.tensor(
        [
            [True, False, True, False],
            [False, True, False, False],
        ]
    )

    actual = gather_target_embeddings(x, target_mask=target_mask)
    expected = torch.stack(
        [
            torch.stack([x[0, 0], x[0, 2]]),
            torch.stack([x[1, 1], x.new_full((2,), nan)]),
        ]
    )

    assert_close(actual, expected, equal_nan=True)


def test_grafiti_forward_embeddings_allow_duplicate_timestamps() -> None:
    r"""Check GraFITi embeddings mode handles repeated dense timestamps."""
    torch.manual_seed(0)
    model = Grafiti(
        dim_input=3,
        dim_latent=8,
        num_layers=2,
        num_heads=2,
        output_mode="embeddings",
    )
    timestamps = torch.tensor(
        [
            [1.0, 1.0, 2.0, 2.0],
            [0.0, 2.0, 2.0, 3.0],
        ]
    )
    context_values = torch.tensor([
        [[10.0, 11.0, nan], [nan, nan, nan], [nan, nan, 20.0], [nan, nan, nan]],
        [[nan, 1.0, nan], [20.0, nan, 22.0], [nan, nan, nan], [nan, 31.0, nan]],
    ])  # fmt: skip
    query_mask = torch.tensor([
        [[False, False, False], [False, False,  True], [False, False, False], [ True, False, False]],
        [[False, False, False], [False, False, False], [ True,  True, False], [False, False, False]],
    ])  # fmt: skip

    actual = model.forward(
        timestamps=timestamps,
        context_values=context_values,
        context_mask=context_values.isfinite(),
        query_mask=query_mask,
    )

    assert actual.shape == (2, 2, model.latent_dim)
    assert actual.isfinite().all()


def test_grafiti_batched_forward_allows_missing_context_values() -> None:
    r"""Check batched GraFITi handles sparse dense context values."""
    torch.manual_seed(0)
    model = Grafiti(dim_input=3, dim_latent=8, num_layers=2, num_heads=2)
    time_points = torch.tensor([
        [0.0, 1.0, 2.0, 3.0],
        [0.0, 1.5, 2.5, 4.0],
    ])  # fmt: skip
    context_values = torch.tensor([
        [[1.0,  nan,  3.0], [nan,  5.0,  nan], [7.0,  nan,  nan], [nan,  8.0,  nan]],
        [[nan, 10.0,  nan], [11.0, nan, 13.0], [nan,  nan, 14.0], [15.0, nan,  nan]],
    ])  # fmt: skip
    target_mask = torch.tensor([
        [[False, False, False], [ True, False, False], [False, False,  True], [ True, False, False]],
        [[False, False, False], [False, False, False], [False,  True, False], [False, False,  True]],
    ])  # fmt: skip

    actual = model.forward(
        timestamps=time_points,
        context_values=context_values,
        context_mask=context_values.isfinite(),
        query_mask=target_mask,
    )
    expected = actual.new_full(actual.shape, nan)
    for k in range(time_points.shape[0]):
        output = model.forward(
            timestamps=time_points[k],
            context_values=context_values[k],
            context_mask=context_values[k].isfinite(),
            query_mask=target_mask[k],
        )
        expected[k] = output

    assert actual.shape == target_mask.shape
    assert actual[target_mask].isfinite().all()
    assert_close(actual, expected, equal_nan=True)
