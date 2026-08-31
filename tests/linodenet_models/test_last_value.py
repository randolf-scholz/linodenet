r"""Tests for last-value forecasting."""

from math import nan
from typing import ClassVar

import matplotlib.pyplot as plt
import pytest
import torch
from torch import Tensor
from torch.nn import functional as F
from torch.testing import assert_close

from linodenet_models import LastValue
from linodenet_models.utils import SplitTimeData
from tests.testing import PROJECT

from .base import TestContinuousTimeModel

RESULT_DIR = PROJECT.RESULTS_DIR[__file__]


class TestLastValue(TestContinuousTimeModel[LastValue]):
    r"""Shared forecasting-model tests for LastValue."""

    CONTEXT_SHAPE: ClassVar[tuple[int, ...]] = (4,)
    OUTPUT_SHAPE: ClassVar[tuple[int, ...]] = CONTEXT_SHAPE

    @pytest.fixture(params=[False, True], ids=["no_missingness", "input_missingness"])
    def input_missingness(self, request: pytest.FixtureRequest) -> bool:
        r"""Whether to randomly mask half of the context values with NaN."""
        return request.param

    def make_model(self, model_config: object, /) -> LastValue:
        r"""Instantiate the LastValue model under test."""
        del model_config
        return LastValue()

    def forecast(self, model: LastValue, args: SplitTimeData) -> tuple[Tensor, ...]:
        r"""Return LastValue predictions for sequential forecasting inputs."""
        assert args.target_values is not None
        query_valid = args.query_mask.any(dim=-1)
        shape = (
            *args.context_values.shape[:-2],
            1,
            *args.context_values.shape[-1:],
        )
        initial_state = args.context_values.new_zeros(shape)
        pred = model(
            query_times=args.query_times,
            context_times=args.context_times,
            context_values=args.context_values,
            context_mask=args.context_mask,
            initial_state=initial_state,
        )

        assert pred.shape == args.target_values.shape
        assert pred[query_valid].isfinite().all()
        assert pred[~query_valid].isnan().all()
        return (pred,)

    def loss(
        self, model: LastValue, predictions: tuple[Tensor, ...], targets: Tensor
    ) -> Tensor:
        r"""Return mean squared error for LastValue predictions."""
        del model
        (forecast,) = predictions
        mask = targets.isfinite()
        return F.mse_loss(forecast[mask], targets[mask])

    def test_training_unbatched(self, *_arg: object, **_kwargs: object) -> None:
        r"""LastValue is stateless and has no trainable parameters."""

    def test_training_batched(self, *_arg: object, **_kwargs: object) -> None:
        r"""LastValue is stateless and has no trainable parameters."""


def test_last_value_carries_features_independently() -> None:
    r"""LastValue skips missing features independently."""
    model = LastValue()
    query_times = torch.tensor([0.5, 1.5, 2.5, 3.5])
    context_times = torch.tensor([0.0, 1.0, 2.0, 3.0])
    context_values = torch.tensor(
        [
            [1.0, nan],
            [nan, 2.0],
            [3.0, nan],
            [nan, 4.0],
        ]
    )

    actual = model(
        query_times=query_times,
        context_times=context_times,
        context_values=context_values,
        context_mask=context_values.isfinite(),
    )
    expected = torch.tensor(
        [
            [1.0, nan],
            [1.0, 2.0],
            [3.0, 2.0],
            [3.0, 4.0],
        ]
    )

    assert_close(actual, expected, equal_nan=True)


def test_last_value_supports_batched_inputs() -> None:
    r"""LastValue preserves batch dimensions."""
    model = LastValue()
    query_times = torch.tensor([[0.5, 1.5], [1.5, 2.5]])
    context_times = torch.tensor([[0.0, 1.0], [1.0, 2.0]])
    context_values = torch.tensor(
        [
            [[1.0, nan], [nan, 2.0]],
            [[3.0, 4.0], [5.0, nan]],
        ]
    )

    actual = model(
        query_times=query_times,
        context_times=context_times,
        context_values=context_values,
        context_mask=context_values.isfinite(),
    )
    expected = torch.tensor(
        [
            [[1.0, nan], [1.0, 2.0]],
            [[3.0, 4.0], [5.0, 4.0]],
        ]
    )

    assert_close(actual, expected, equal_nan=True)


def test_last_value_uses_initial_state_before_observations() -> None:
    r"""LastValue falls back to the initial state before any observation."""
    model = LastValue()
    query_times = torch.tensor([-1.0, 0.5, 1.5, 2.5])
    context_times = torch.tensor([0.0, 2.0])
    context_values = torch.tensor(
        [
            [1.0, nan],
            [nan, 4.0],
        ]
    )
    initial_state = torch.tensor([0.0, -1.0])

    actual = model(
        query_times=query_times,
        context_times=context_times,
        context_values=context_values,
        context_mask=context_values.isfinite(),
        initial_state=initial_state,
    )
    expected = torch.tensor(
        [
            [0.0, -1.0],
            [1.0, -1.0],
            [1.0, -1.0],
            [1.0, 4.0],
        ]
    )

    assert_close(actual, expected, equal_nan=True)


def test_last_value_defaults_to_nan_initial_state() -> None:
    r"""LastValue defaults to NaN before any observation."""
    model = LastValue()
    query_times = torch.tensor([-1.0, 1.0])
    context_times = torch.tensor([0.0])
    context_values = torch.tensor([[1.0, nan]])

    actual = model(
        query_times=query_times,
        context_times=context_times,
        context_values=context_values,
        context_mask=context_values.isfinite(),
    )
    expected = torch.tensor(
        [
            [nan, nan],
            [1.0, nan],
        ]
    )

    assert_close(actual, expected, equal_nan=True)


def test_last_value_supports_trailing_nan_padding() -> None:
    r"""LastValue ignores trailing NaN padding in query and context times."""
    model = LastValue()
    query_times = torch.tensor([-1.0, 1.0, 3.0, nan])
    context_times = torch.tensor([0.0, 2.0, nan, nan])
    context_values = torch.tensor(
        [
            [1.0, nan],
            [nan, 3.0],
            [nan, nan],
            [nan, nan],
        ]
    )
    initial_state = torch.tensor([0.0, -1.0])

    actual = model(
        query_times=query_times,
        context_times=context_times,
        context_values=context_values,
        context_mask=context_values.isfinite(),
        initial_state=initial_state,
    )
    expected = torch.tensor(
        [
            [0.0, -1.0],
            [1.0, -1.0],
            [1.0, 3.0],
            [nan, nan],
        ]
    )

    assert_close(actual, expected, equal_nan=True)


def test_last_value_visual_forecast() -> None:
    r"""Plot sparse channel observations and their last-value forecast."""
    torch.manual_seed(0)
    model = LastValue()
    num_channels = 3
    horizon = 10.0
    samples_per_channel = 6

    amplitude = 0.75 + 0.5 * torch.rand(num_channels)
    frequency = 0.5 + torch.rand(num_channels)
    phase = 2 * torch.pi * torch.rand(num_channels)

    sample_times = horizon * torch.rand(num_channels, samples_per_channel)
    sample_times, _ = sample_times.sort(dim=-1)
    sample_values = amplitude[:, None] * torch.sin(
        frequency[:, None] * sample_times + phase[:, None]
    )

    context_times = sample_times.flatten().unique(sorted=True)
    context_values = torch.full((len(context_times), num_channels), torch.nan)
    for channel, times in enumerate(sample_times):
        indices = torch.searchsorted(context_times, times)
        context_values[indices, channel] = sample_values[channel]

    query_times = torch.linspace(0.0, horizon, 1000)
    forecast = model(
        query_times=query_times,
        context_times=context_times,
        context_values=context_values,
        context_mask=context_values.isfinite(),
        initial_state=torch.randn(num_channels),
    )

    tab10 = plt.colormaps["tab10"]
    colors = [tab10(index) for index in range(num_channels)]
    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    for channel, color in enumerate(colors):
        ax.plot(
            query_times,
            forecast[:, channel],
            color=color,
            label=f"channel {channel} forecast",
        )
        ax.plot(
            sample_times[channel],
            sample_values[channel],
            linestyle="",
            marker="o",
            color=color,
            label=f"channel {channel} data",
        )

    ax.set_xlabel("time")
    ax.set_ylabel("value")
    ax.set_title("Last-value forecast with feature missingness")
    ax.set_xlim(0.0, horizon)
    ax.legend(ncols=3, fontsize="small")
    fig.savefig(RESULT_DIR / "last_value_forecast.png", dpi=200)
    # plt.show()
    plt.close(fig)
