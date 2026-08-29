r"""Tests for forecasting utility containers."""

from dataclasses import replace
from typing import Literal, cast

import pytest
import torch
from torch import Tensor, nan
from torch.testing import assert_close
from typing_extensions import TypedDict

from linodenet_models.utils import (
    DiscreteTimeEventBatch,
    EventBatch,
    MergedTimeData,
    SplitTimeData,
    TripletTimeData,
    merged_to_split,
    merged_to_triplet,
    split_to_merged,
    split_to_triplet,
    triplet_to_merged,
    triplet_to_split,
)

from .base import make_continuous_time_request


def test_discrete_time_event_batch_uses_zero_step_padding() -> None:
    r"""Check integer event batches use masks for validity and zero step padding."""
    context_steps = torch.tensor([1, 3, 0])
    context_mask = torch.tensor(
        [
            [True, False],
            [False, True],
            [False, False],
        ]
    )
    context_values = torch.tensor([[10.0, nan], [nan, 31.0], [nan, nan]])
    query_steps = torch.tensor([2, 3, 0])
    query_mask = torch.tensor(
        [
            [True, False],
            [True, True],
            [False, False],
        ]
    )
    target_values = torch.tensor([[20.0, nan], [30.0, 31.0], [nan, nan]])

    batch = DiscreteTimeEventBatch.from_request(
        context_times=context_steps,
        context_values=context_values,
        context_mask=context_mask,
        query_times=query_steps,
        query_mask=query_mask,
        target_values=target_values,
    )

    assert_close(batch.steps, torch.tensor([1, 2, 3, 3, 0, 0]))
    assert torch.equal(
        batch.context_mask,
        torch.tensor(
            [
                [True, False],
                [False, False],
                [False, True],
                [False, False],
                [False, False],
                [False, False],
            ]
        ),
    )
    assert torch.equal(
        batch.query_mask,
        torch.tensor(
            [
                [False, False],
                [True, False],
                [False, False],
                [True, True],
                [False, False],
                [False, False],
            ]
        ),
    )
    assert_close(batch.query_mask[batch.query_indices], query_mask)


class CanonicalTestData(TypedDict, closed=True):
    split: SplitTimeData
    merged: MergedTimeData
    triplet: TripletTimeData


class TensorViewData(TypedDict, closed=True):
    split: dict[str, Tensor]
    merged: dict[str, Tensor]
    triplet: dict[str, Tensor]


def _repeat_batch_elements(data: TensorViewData, repeats: int, /) -> TensorViewData:
    return cast(
        "TensorViewData",
        {
            data_format: {
                name: tensor.repeat(repeats, *([1] * (tensor.ndim - 1)))
                for name, tensor in tensors.items()
            }
            for data_format, tensors in data.items()
        },
    )


def _reshape_single_batch(
    data: TensorViewData,
    batch_shape: tuple[int, ...],
    /,
) -> TensorViewData:
    return cast(
        "TensorViewData",
        {
            data_format: {
                name: tensor.reshape(*batch_shape, *tensor.shape[1:])
                for name, tensor in tensors.items()
            }
            for data_format, tensors in data.items()
        },
    )


type BatchType = Literal["unbatched", "single", "multi"]
type DataType = Literal["simple", "sparse", "general"]
type DataFormat = Literal["split", "merged", "triplet"]

UNBATCHED_TEST_DATA: CanonicalTestData = {
    "split": SplitTimeData(
        context_times=torch.tensor([1.0, 3.0]),
        context_values=torch.tensor([[10.0, nan, 12.0], [nan, 30.0, 32.0]]),
        context_mask=torch.tensor([[True, False, True], [False, True, True]]),
        query_times=torch.tensor([2.0, 4.0]),
        query_mask=torch.tensor([[True, False, True], [True, True, True]]),
        target_values=torch.tensor([[20.0, nan, 22.0], [40.0, 41.0, 42.0]]),
        static_covariates=torch.tensor([5.0, 6.0]),
    ),
    "merged": MergedTimeData(
        timestamps=torch.tensor([1.0, 2.0, 3.0, 4.0]),
        context_values=torch.tensor([
            [10.0, nan, 12.0],
            [20.0, nan, 22.0],
            [nan, 30.0, 32.0],
            [40.0, 41.0, 42.0],
        ]),
        context_mask=torch.tensor([
            [True, False, True],
            [False, False, False],
            [False, True, True],
            [False, False, False],
        ]),
        query_mask=torch.tensor([
            [False, False, False],
            [True, False, True],
            [False, False, False],
            [True, True, True],
        ]),
        target_values=torch.tensor([
            [10.0, nan, 12.0],
            [20.0, nan, 22.0],
            [nan, 30.0, 32.0],
            [40.0, 41.0, 42.0],
        ]),
        static_covariates=torch.tensor([5.0, 6.0]),
    ),
    "triplet": TripletTimeData(
        context_times=torch.tensor([1.0, 1.0, 3.0, 3.0]),
        context_channels=torch.tensor([0, 2, 1, 2]),
        context_values=torch.tensor([10.0, 12.0, 30.0, 32.0]),
        query_times=torch.tensor([2.0, 2.0, 4.0, 4.0, 4.0]),
        query_channels=torch.tensor([0, 2, 0, 1, 2]),
        target_values=torch.tensor([20.0, 22.0, 40.0, 41.0, 42.0]),
        static_covariates=torch.tensor([5.0, 6.0]),
    ),
}  # fmt: skip
BATCHED_TEST_DATA: CanonicalTestData = {
    "split": SplitTimeData(
        context_times=torch.tensor([
            [1.0, 3.0],
            [0.0, nan],
        ]),
        context_values=torch.tensor([
            [[10.0, nan, 12.0], [nan, 30.0, 32.0]],
            [[nan, 1.0, 2.0], [nan, nan, nan]],
        ]),
        context_mask=torch.tensor([
            [[True, False, True], [False, True, True]],
            [[False, True, True], [False, False, False]],
        ]),
        query_times=torch.tensor([[2.0, 4.0], [5.0, nan]]),
        query_mask=torch.tensor([
            [[True, False, True], [True, True, True]],
            [[False, True, True], [False, False, False]],
        ]),
        target_values=torch.tensor([
            [[20.0, nan, 22.0], [40.0, 41.0, 42.0]],
            [[nan, 51.0, 52.0], [nan, nan, nan]],
        ]),
        static_covariates=torch.tensor([[5.0, 6.0], [7.0, 8.0]]),
    ),
    "merged": MergedTimeData(
        timestamps=torch.tensor([
                [1.0, 2.0, 3.0, 4.0],
                [0.0, 5.0, nan, nan],
        ]),
        context_values=torch.tensor([
            [[10.0, nan, 12.0],
             [20.0, nan, 22.0],
             [nan, 30.0, 32.0],
             [40.0, 41.0, 42.0]],

            [[nan, 1.0, 2.0],
             [nan, 51.0, 52.0],
             [nan, nan, nan],
             [nan, nan, nan]],
        ]),
        context_mask=torch.tensor([
            [[True, False, True],
             [False, False, False],
             [False, True, True],
             [False, False, False]],

            [[False, True, True],
             [False, False, False],
             [False, False, False],
             [False, False, False]],
        ]),
        query_mask=torch.tensor([
            [[False, False, False],
             [True, False, True],
             [False, False, False],
             [True, True, True]],

            [[False, False, False],
             [False, True, True],
             [False, False, False],
             [False, False, False]],
        ]),
        target_values=torch.tensor([
            [[10.0, nan, 12.0],
             [20.0, nan, 22.0],
             [nan, 30.0, 32.0],
             [40.0, 41.0, 42.0]],

            [[nan, 1.0, 2.0],
             [nan, 51.0, 52.0],
             [nan, nan, nan],
             [nan, nan, nan]],
        ]),
        static_covariates=torch.tensor([[5.0, 6.0], [7.0, 8.0]]),
    ),
    "triplet": TripletTimeData(
        context_times=torch.tensor([
            [1.0, 1.0, 3.0, 3.0],
            [0.0, 0.0, nan, nan],
        ]),
        context_channels=torch.tensor([
            [0, 2, 1, 2],
            [1, 2, -1, -1],
        ]),
        context_values=torch.tensor([
            [10.0, 12.0, 30.0, 32.0],
            [1.0, 2.0, nan, nan],
        ]),
        query_times=torch.tensor([
            [2.0, 2.0, 4.0, 4.0, 4.0],
            [5.0, 5.0, nan, nan, nan],
        ]),
        query_channels=torch.tensor([
            [0, 2, 0, 1, 2],
            [1, 2, -1, -1, -1],
        ]),
        target_values=torch.tensor([
            [20.0, 22.0, 40.0, 41.0, 42.0],
            [51.0, 52.0, nan, nan, nan],
        ]),
        static_covariates=torch.tensor([
            [5.0, 6.0],
            [7.0, 8.0],
        ]),
    ),
}  # fmt: skip

BATCH_SHAPES: dict[BatchType, tuple[int, ...]] = {
    "unbatched": (),
    "single": (6,),
    "multi": (1, 2, 3),
}

UNBATCHED_SIMPLE_DATA: TensorViewData = {
    "split": {
        "context_times": torch.tensor([1.0, 3.0]),
        "context_values": torch.tensor([[10.0, nan, 12.0], [nan, 30.0, 32.0]]),
        "context_mask": torch.tensor([[True, False, True], [False, True, True]]),
        "query_times": torch.tensor([2.0, 4.0]),
        "query_mask": torch.tensor([[True, False, True], [True, True, True]]),
        "target_values": torch.tensor([[20.0, nan, 22.0], [40.0, 41.0, 42.0]]),
        "static_covariates": torch.tensor([5.0, 6.0]),
    },
    "merged": {
        "timestamps": torch.tensor([1.0, 2.0, 3.0, 4.0]),
        "context_values": torch.tensor([
            [10.0, nan, 12.0],
            [nan, nan, nan],
            [nan, 30.0, 32.0],
            [nan, nan, nan],
        ]),
        "context_mask": torch.tensor([
            [True, False, True],
            [False, False, False],
            [False, True, True],
            [False, False, False],
        ]),
        "query_mask": torch.tensor([
            [False, False, False],
            [True, False, True],
            [False, False, False],
            [True, True, True],
        ]),
        "target_values": torch.tensor([
            [nan, nan, nan],
            [20.0, nan, 22.0],
            [nan, nan, nan],
            [40.0, 41.0, 42.0],
        ]),
        "static_covariates": torch.tensor([5.0, 6.0]),
    },
    "triplet": {
        "context_times": torch.tensor([1.0, 1.0, 3.0, 3.0]),
        "context_channels": torch.tensor([0, 2, 1, 2]),
        "context_values": torch.tensor([10.0, 12.0, 30.0, 32.0]),
        "query_times": torch.tensor([2.0, 2.0, 4.0, 4.0, 4.0]),
        "query_channels": torch.tensor([0, 2, 0, 1, 2]),
        "target_values": torch.tensor([20.0, 22.0, 40.0, 41.0, 42.0]),
        "static_covariates": torch.tensor([5.0, 6.0]),
    },
}  # fmt: skip
UNBATCHED_SPARSE_DATA: TensorViewData = {
    "split": {
        "context_times": torch.tensor([1.0, 1.0, 3.0]),
        "context_values": torch.tensor([
            [10.0, nan, nan],
            [nan, nan, 12.0],
            [nan, 30.0, nan],
        ]),
        "context_mask": torch.tensor([
            [True, False, False],
            [False, False, True],
            [False, True, False],
        ]),
        "query_times": torch.tensor([2.0, 4.0, 4.0]),
        "query_mask": torch.tensor([
            [True, False, False],
            [False, True, False],
            [False, False, True],
        ]),
        "target_values": torch.tensor([
            [20.0, nan, nan],
            [nan, 41.0, nan],
            [nan, nan, 42.0],
        ]),
        "static_covariates": torch.tensor([7.0, 8.0]),
    },
    "merged": {
        "timestamps": torch.tensor([1.0, 1.0, 2.0, 3.0, 4.0, 4.0]),
        "context_values": torch.tensor([
            [10.0, nan, nan],
            [nan, nan, 12.0],
            [nan, nan, nan],
            [nan, 30.0, nan],
            [nan, nan, nan],
            [nan, nan, nan],
        ]),
        "context_mask": torch.tensor([
            [True, False, False],
            [False, False, True],
            [False, False, False],
            [False, True, False],
            [False, False, False],
            [False, False, False],
        ]),
        "query_mask": torch.tensor([
            [False, False, False],
            [False, False, False],
            [True, False, False],
            [False, False, False],
            [False, True, False],
            [False, False, True],
        ]),
        "target_values": torch.tensor([
            [nan, nan, nan],
            [nan, nan, nan],
            [20.0, nan, nan],
            [nan, nan, nan],
            [nan, 41.0, nan],
            [nan, nan, 42.0],
        ]),
        "static_covariates": torch.tensor([7.0, 8.0]),
    },
    "triplet": {
        "context_times": torch.tensor([1.0, 1.0, 3.0]),
        "context_channels": torch.tensor([0, 2, 1]),
        "context_values": torch.tensor([10.0, 12.0, 30.0]),
        "query_times": torch.tensor([2.0, 4.0, 4.0]),
        "query_channels": torch.tensor([0, 1, 2]),
        "target_values": torch.tensor([20.0, 41.0, 42.0]),
        "static_covariates": torch.tensor([7.0, 8.0]),
    },
}  # fmt: skip
UNBATCHED_GENERAL_DATA: TensorViewData = {
    "split": {
        "context_times": torch.tensor([1.0, 1.0, 2.0]),
        "context_values": torch.tensor([
            [10.0, nan, nan],
            [11.0, 12.0, nan],
            [nan, 20.0, 21.0],
        ]),
        "context_mask": torch.tensor([
            [True, False, False],
            [True, True, False],
            [False, True, True],
        ]),
        "query_times": torch.tensor([3.0, 3.0, 4.0]),
        "query_mask": torch.tensor([
            [True, False, False],
            [True, True, False],
            [False, False, True],
        ]),
        "target_values": torch.tensor([
            [30.0, nan, nan],
            [31.0, 32.0, nan],
            [nan, nan, 40.0],
        ]),
        "static_covariates": torch.tensor([9.0, 10.0]),
    },
    "merged": {
        "timestamps": torch.tensor([1.0, 1.0, 2.0, 3.0, 3.0, 4.0]),
        "context_values": torch.tensor([
            [10.0, nan, nan],
            [11.0, 12.0, nan],
            [nan, 20.0, 21.0],
            [nan, nan, nan],
            [nan, nan, nan],
            [nan, nan, nan],
        ]),
        "context_mask": torch.tensor([
            [True, False, False],
            [True, True, False],
            [False, True, True],
            [False, False, False],
            [False, False, False],
            [False, False, False],
        ]),
        "query_mask": torch.tensor([
            [False, False, False],
            [False, False, False],
            [False, False, False],
            [True, False, False],
            [True, True, False],
            [False, False, True],
        ]),
        "target_values": torch.tensor([
            [nan, nan, nan],
            [nan, nan, nan],
            [nan, nan, nan],
            [30.0, nan, nan],
            [31.0, 32.0, nan],
            [nan, nan, 40.0],
        ]),
        "static_covariates": torch.tensor([9.0, 10.0]),
    },
    "triplet": {
        "context_times": torch.tensor([1.0, 1.0, 1.0, 2.0, 2.0]),
        "context_channels": torch.tensor([0, 0, 1, 1, 2]),
        "context_values": torch.tensor([10.0, 11.0, 12.0, 20.0, 21.0]),
        "query_times": torch.tensor([3.0, 3.0, 3.0, 4.0]),
        "query_channels": torch.tensor([0, 0, 1, 2]),
        "target_values": torch.tensor([30.0, 31.0, 32.0, 40.0]),
        "static_covariates": torch.tensor([9.0, 10.0]),
    },
}  # fmt: skip

BATCHED_SIMPLE_DATA: TensorViewData = {
    "split": {
        "context_times": torch.tensor([
            [1.0, 3.0],
            [0.0, nan],
        ]),
        "context_values": torch.tensor([
            [[10.0, nan, 12.0], [nan, 30.0, 32.0]],
            [[nan, 1.0, 2.0], [nan, nan, nan]],
        ]),
        "context_mask": torch.tensor([
            [[True, False, True], [False, True, True]],
            [[False, True, True], [False, False, False]],
        ]),
        "query_times": torch.tensor([
            [2.0, 4.0],
            [5.0, nan],
        ]),
        "query_mask": torch.tensor([
            [[True, False, True], [True, True, True]],
            [[False, True, True], [False, False, False]],
        ]),
        "target_values": torch.tensor([
            [[20.0, nan, 22.0], [40.0, 41.0, 42.0]],
            [[nan, 51.0, 52.0], [nan, nan, nan]],
        ]),
        "static_covariates": torch.tensor([
            [5.0, 6.0],
            [7.0, 8.0],
        ]),
    },
    "merged": {
        "timestamps": torch.tensor([
            [1.0, 2.0, 3.0, 4.0],
            [0.0, 5.0, nan, nan],
        ]),
        "context_values": torch.tensor([
            [[10.0, nan, 12.0], [nan, nan, nan], [nan, 30.0, 32.0], [nan, nan, nan]],
            [[nan, 1.0, 2.0], [nan, nan, nan], [nan, nan, nan], [nan, nan, nan]],
        ]),
        "target_values": torch.tensor([
            [[nan, nan, nan], [20.0, nan, 22.0], [nan, nan, nan], [40.0, 41.0, 42.0]],
            [[nan, nan, nan], [nan, 51.0, 52.0], [nan, nan, nan], [nan, nan, nan]],
        ]),
        "context_mask": torch.tensor([
            [[True, False, True],
             [False, False, False],
             [False, True, True],
             [False, False, False]],
            [[False, True, True],
             [False, False, False],
             [False, False, False],
             [False, False, False]],
        ]),
        "query_mask": torch.tensor([
            [[False, False, False],
             [True, False, True],
             [False, False, False],
             [True, True, True]],

            [[False, False, False],
             [False, True, True],
             [False, False, False],
             [False, False, False]],
        ]),
        "static_covariates": torch.tensor([
            [5.0, 6.0],
            [7.0, 8.0],
        ]),
    },
    "triplet": {
        "context_times": torch.tensor([
            [1.0, 1.0, 3.0, 3.0],
            [0.0, 0.0, nan, nan],
        ]),
        "context_channels": torch.tensor([
            [0, 2, 1, 2],
            [1, 2, -1, -1],
        ]),
        "context_values": torch.tensor([
            [10.0, 12.0, 30.0, 32.0],
            [1.0, 2.0, nan, nan],
        ]),
        "query_times": torch.tensor([
            [2.0, 2.0, 4.0, 4.0, 4.0],
            [5.0, 5.0, nan, nan, nan],
        ]),
        "query_channels": torch.tensor([
            [0, 2, 0, 1, 2],
            [1, 2, -1, -1, -1],
        ]),
        "target_values": torch.tensor([
            [20.0, 22.0, 40.0, 41.0, 42.0],
            [51.0, 52.0, nan, nan, nan],
        ]),
        "static_covariates": torch.tensor([
            [5.0, 6.0],
            [7.0, 8.0],
        ]),
    },
}  # fmt: skip
BATCHED_SPARSE_DATA: TensorViewData = {
    "split": {
        "context_times": torch.tensor([
            [1.0, 1.0, 3.0],
            [0.0, 2.0, nan],
        ]),
        "context_values": torch.tensor([
            [[10.0, nan, nan], [nan, nan, 12.0], [nan, 30.0, nan]],
            [[nan, 1.0, nan], [2.0, nan, nan], [nan, nan, nan]],
        ]),
        "context_mask": torch.tensor([
            [[True, False, False], [False, False, True], [False, True, False]],
            [[False, True, False], [True, False, False], [False, False, False]],
        ]),
        "query_times": torch.tensor([
            [2.0, 4.0, 4.0],
            [1.0, 1.0, 3.0],
        ]),
        "query_mask": torch.tensor([
            [[True, False, False], [False, True, False], [False, False, True]],
            [[False, True, False], [False, False, True], [True, False, False]],
        ]),
        "target_values": torch.tensor([
            [[20.0, nan, nan], [nan, 41.0, nan], [nan, nan, 42.0]],
            [[nan, 11.0, nan], [nan, nan, 12.0], [13.0, nan, nan]],
        ]),
        "static_covariates": torch.tensor([
            [7.0, 8.0],
            [9.0, 10.0],
        ]),
    },
    "merged": {
        "timestamps": torch.tensor([
            [1.0, 1.0, 2.0, 3.0, 4.0, 4.0],
            [0.0, 1.0, 1.0, 2.0, 3.0, nan],
        ]),
        "context_values": torch.tensor([
            [[10.0, nan, nan],
             [nan, nan, 12.0],
             [nan, nan, nan],
             [nan, 30.0, nan],
             [nan, nan, nan],
             [nan, nan, nan]],

            [[nan, 1.0, nan],
             [nan, nan, nan],
             [nan, nan, nan],
             [2.0, nan, nan],
             [nan, nan, nan],
             [nan, nan, nan]],
        ]),
        "target_values": torch.tensor([
            [[nan, nan, nan],
             [nan, nan, nan],
             [20.0, nan, nan],
             [nan, nan, nan],
             [nan, 41.0, nan],
             [nan, nan, 42.0]],

            [[nan, nan, nan],
             [nan, 11.0, nan],
             [nan, nan, 12.0],
             [nan, nan, nan],
             [13.0, nan, nan],
             [nan, nan, nan]],
        ]),
        "context_mask": torch.tensor([
            [[True, False, False],
             [False, False, True],
             [False, False, False],
             [False, True, False],
             [False, False, False],
             [False, False, False]],

            [[False, True, False],
             [False, False, False],
             [False, False, False],
             [True, False, False],
             [False, False, False],
             [False, False, False]],
        ]),
        "query_mask": torch.tensor([
            [[False, False, False],
             [False, False, False],
             [True, False, False],
             [False, False, False],
             [False, True, False],
             [False, False, True]],

            [[False, False, False],
             [False, True, False],
             [False, False, True],
             [False, False, False],
             [True, False, False],
             [False, False, False]],
        ]),
        "static_covariates": torch.tensor([
            [7.0, 8.0],
            [9.0, 10.0],
        ]),
    },
    "triplet": {
        "context_times": torch.tensor([
            [1.0, 1.0, 3.0],
            [0.0, 2.0, nan],
        ]),
        "context_channels": torch.tensor([
            [0, 2, 1],
            [1, 0, -1],
        ]),
        "context_values": torch.tensor([
            [10.0, 12.0, 30.0],
            [1.0, 2.0, nan],
        ]),
        "query_times": torch.tensor([
            [2.0, 4.0, 4.0],
            [1.0, 1.0, 3.0],
        ]),
        "query_channels": torch.tensor([
            [0, 1, 2],
            [1, 2, 0],
        ]),
        "target_values": torch.tensor([
            [20.0, 41.0, 42.0],
            [11.0, 12.0, 13.0],
        ]),
        "static_covariates": torch.tensor([
            [7.0, 8.0],
            [9.0, 10.0],
        ]),
    },
}  # fmt: skip
BATCHED_GENERAL_DATA: TensorViewData = {
    "split": {
        "context_times": torch.tensor([
            [1.0, 1.0, 2.0],
            [0.0, 0.0, nan],
        ]),
        "context_values": torch.tensor([
            [[10.0, nan, nan], [11.0, 12.0, nan], [nan, 20.0, 21.0]],
            [[nan, 1.0, nan], [nan, 2.0, 3.0], [nan, nan, nan]],
        ]),
        "context_mask": torch.tensor([
            [[True, False, False], [True, True, False], [False, True, True]],
            [[False, True, False], [False, True, True], [False, False, False]],
        ]),
        "query_times": torch.tensor([
            [3.0, 3.0, 4.0],
            [5.0, 5.0, 6.0],
        ]),
        "query_mask": torch.tensor([
            [[True, False, False], [True, True, False], [False, False, True]],
            [[False, True, False], [False, True, True], [True, False, False]],
        ]),
        "target_values": torch.tensor([
            [[30.0, nan, nan], [31.0, 32.0, nan], [nan, nan, 40.0]],
            [[nan, 51.0, nan], [nan, 52.0, 53.0], [60.0, nan, nan]],
        ]),
        "static_covariates": torch.tensor([
            [9.0, 10.0],
            [11.0, 12.0],
        ]),
    },
    "merged": {
        "timestamps": torch.tensor([
            [1.0, 1.0, 2.0, 3.0, 3.0, 4.0],
            [0.0, 0.0, 5.0, 5.0, 6.0, nan],
        ]),
        "context_values": torch.tensor([
            [[10.0, nan, nan],
             [11.0, 12.0, nan],
             [nan, 20.0, 21.0],
             [nan, nan, nan],
             [nan, nan, nan],
             [nan, nan, nan]],

            [[nan, 1.0, nan],
             [nan, 2.0, 3.0],
             [nan, nan, nan],
             [nan, nan, nan],
             [nan, nan, nan],
             [nan, nan, nan]],
        ]),
        "context_mask": torch.tensor([
            [[True, False, False],
             [True, True, False],
             [False, True, True],
             [False, False, False],
             [False, False, False],
             [False, False, False]],

            [[False, True, False],
             [False, True, True],
             [False, False, False],
             [False, False, False],
             [False, False, False],
             [False, False, False]],
        ]),
        "query_mask": torch.tensor([
            [[False, False, False],
             [False, False, False],
             [False, False, False],
             [True, False, False],
             [True, True, False],
             [False, False, True]],

            [[False, False, False],
             [False, False, False],
             [False, True, False],
             [False, True, True],
             [True, False, False],
             [False, False, False]],
        ]),
        "target_values": torch.tensor([
            [[nan, nan, nan],
             [nan, nan, nan],
             [nan, nan, nan],
             [30.0, nan, nan],
             [31.0, 32.0, nan],
             [nan, nan, 40.0]],

            [[nan, nan, nan],
             [nan, nan, nan],
             [nan, 51.0, nan],
             [nan, 52.0, 53.0],
             [60.0, nan, nan],
             [nan, nan, nan]],
        ]),
        "static_covariates": torch.tensor([
            [9.0, 10.0],
            [11.0, 12.0],
        ]),
    },
    "triplet": {
        "context_times": torch.tensor([
            [1.0, 1.0, 1.0, 2.0, 2.0],
            [0.0, 0.0, 0.0, nan, nan],
        ]),
        "context_channels": torch.tensor([
            [0, 0, 1, 1, 2],
            [1, 1, 2, -1, -1],
        ]),
        "context_values": torch.tensor([
            [10.0, 11.0, 12.0, 20.0, 21.0],
            [1.0, 2.0, 3.0, nan, nan],
        ]),
        "query_times": torch.tensor([
            [3.0, 3.0, 3.0, 4.0],
            [5.0, 5.0, 5.0, 6.0],
        ]),
        "query_channels": torch.tensor([
            [0, 0, 1, 2],
            [1, 1, 2, 0],
        ]),
        "target_values": torch.tensor([
            [30.0, 31.0, 32.0, 40.0],
            [51.0, 52.0, 53.0, 60.0],
        ]),
        "static_covariates": torch.tensor([
            [9.0, 10.0],
            [11.0, 12.0],
        ]),
    },
}  # fmt: skip

BATCHED_SIMPLE_DATA = _repeat_batch_elements(BATCHED_SIMPLE_DATA, 3)
BATCHED_SPARSE_DATA = _repeat_batch_elements(BATCHED_SPARSE_DATA, 3)
BATCHED_GENERAL_DATA = _repeat_batch_elements(BATCHED_GENERAL_DATA, 3)

MULTI_BATCHED_SIMPLE_DATA: TensorViewData = _reshape_single_batch(
    BATCHED_SIMPLE_DATA, BATCH_SHAPES["multi"]
)
MULTI_BATCHED_SPARSE_DATA: TensorViewData = _reshape_single_batch(
    BATCHED_SPARSE_DATA, BATCH_SHAPES["multi"]
)
MULTI_BATCHED_GENERAL_DATA: TensorViewData = _reshape_single_batch(
    BATCHED_GENERAL_DATA, BATCH_SHAPES["multi"]
)


def _to_batch_last(data: TensorViewData) -> TensorViewData:
    return {
        "split": {
            key: (
                tensor
                if key == "static_covariates"
                else tensor.movedim(-1, 0)
                if key in {"context_times", "query_times"}
                else tensor.movedim(-2, 0)
            )
            for key, tensor in data["split"].items()
        },
        "merged": {
            key: (
                tensor
                if key == "static_covariates"
                else tensor.movedim(-1, 0)
                if key == "timestamps"
                else tensor.movedim(-2, 0)
            )
            for key, tensor in data["merged"].items()
        },
        "triplet": {
            key: tensor if key == "static_covariates" else tensor.movedim(-1, 0)
            for key, tensor in data["triplet"].items()
        },
    }


_RAW_TEST_DATA: dict[tuple[DataType, BatchType, bool], TensorViewData] = {
    # batch_first
    ("general", "multi", True): MULTI_BATCHED_GENERAL_DATA,
    ("general", "single", True): BATCHED_GENERAL_DATA,
    ("general", "unbatched", True): UNBATCHED_GENERAL_DATA,
    ("simple", "multi", True): MULTI_BATCHED_SIMPLE_DATA,
    ("simple", "single", True): BATCHED_SIMPLE_DATA,
    ("simple", "unbatched", True): UNBATCHED_SIMPLE_DATA,
    ("sparse", "multi", True): MULTI_BATCHED_SPARSE_DATA,
    ("sparse", "single", True): BATCHED_SPARSE_DATA,
    ("sparse", "unbatched", True): UNBATCHED_SPARSE_DATA,
    # batch_last
    ("general", "multi", False): _to_batch_last(MULTI_BATCHED_GENERAL_DATA),
    ("general", "single", False): _to_batch_last(BATCHED_GENERAL_DATA),
    ("general", "unbatched", False): _to_batch_last(UNBATCHED_GENERAL_DATA),
    ("simple", "multi", False): _to_batch_last(MULTI_BATCHED_SIMPLE_DATA),
    ("simple", "single", False): _to_batch_last(BATCHED_SIMPLE_DATA),
    ("simple", "unbatched", False): _to_batch_last(UNBATCHED_SIMPLE_DATA),
    ("sparse", "multi", False): _to_batch_last(MULTI_BATCHED_SPARSE_DATA),
    ("sparse", "single", False): _to_batch_last(BATCHED_SPARSE_DATA),
    ("sparse", "unbatched", False): _to_batch_last(UNBATCHED_SPARSE_DATA),
}


@pytest.mark.parametrize("case", _RAW_TEST_DATA)
def test_initialization(case) -> None:
    data: TensorViewData = _RAW_TEST_DATA[case]
    batch_first = case[-1]
    split_data = data["split"]
    merged_data = data["merged"]
    triplet_data = data["triplet"]

    SplitTimeData(
        context_times=split_data["context_times"],
        context_values=split_data["context_values"],
        context_mask=split_data["context_mask"],
        query_times=split_data["query_times"],
        query_mask=split_data["query_mask"],
        target_values=split_data["target_values"],
        static_covariates=split_data["static_covariates"],
        batch_first=batch_first,
    )
    MergedTimeData(
        timestamps=merged_data["timestamps"],
        context_mask=merged_data["context_mask"],
        context_values=merged_data["context_values"],
        query_mask=merged_data["query_mask"],
        target_values=merged_data["target_values"],
        static_covariates=merged_data["static_covariates"],
        batch_first=batch_first,
    )
    TripletTimeData(
        context_times=triplet_data["context_times"],
        context_channels=triplet_data["context_channels"],
        context_values=triplet_data["context_values"],
        query_times=triplet_data["query_times"],
        query_channels=triplet_data["query_channels"],
        target_values=triplet_data["target_values"],
        static_covariates=triplet_data["static_covariates"],
        batch_first=batch_first,
    )


def _init_time_data(data: TensorViewData, batch_first: bool) -> CanonicalTestData:
    split_data = data["split"]
    merged_data = data["merged"]
    triplet_data = data["triplet"]

    return {
        "split": SplitTimeData(
            context_times=split_data["context_times"],
            context_values=split_data["context_values"],
            context_mask=split_data["context_mask"],
            query_times=split_data["query_times"],
            query_mask=split_data["query_mask"],
            target_values=split_data["target_values"],
            static_covariates=split_data["static_covariates"],
            batch_first=batch_first,
        ),
        "merged": MergedTimeData(
            timestamps=merged_data["timestamps"],
            context_mask=merged_data["context_mask"],
            context_values=merged_data["context_values"],
            query_mask=merged_data["query_mask"],
            target_values=merged_data["target_values"],
            static_covariates=merged_data["static_covariates"],
            batch_first=batch_first,
        ),
        "triplet": TripletTimeData(
            context_times=triplet_data["context_times"],
            context_channels=triplet_data["context_channels"],
            context_values=triplet_data["context_values"],
            query_times=triplet_data["query_times"],
            query_channels=triplet_data["query_channels"],
            target_values=triplet_data["target_values"],
            static_covariates=triplet_data["static_covariates"],
            batch_first=batch_first,
        ),
    }


TEST_DATA: dict[tuple[DataType, BatchType, bool], CanonicalTestData] = {
    key: _init_time_data(value, batch_first=key[-1])
    for key, value in _RAW_TEST_DATA.items()
}


BATCH_PARAMETERS = pytest.mark.parametrize(
    ("batch_shape", "batch_first"),
    [
        pytest.param(batch_shape, batch_first, id=f"{batch_type}-{batch_first=}")
        for batch_type, batch_shape in BATCH_SHAPES.items()
        for batch_first in (True, False)
    ],
)


def _simple_test_data(
    batch_shape: tuple[int, ...], batch_first: bool, /
) -> CanonicalTestData:
    batch_type = cast(
        "BatchType",
        next(
            batch_type
            for batch_type, candidate_shape in BATCH_SHAPES.items()
            if candidate_shape == batch_shape
        ),
    )
    return TEST_DATA["simple", batch_type, batch_first]


class TestModuleTestData:
    @staticmethod
    def _has_duplicate_time_channel_pair(times: Tensor, channels: Tensor, /) -> bool:
        valid = times.isfinite() & channels.ge(0)
        if not valid.any():
            return False

        pairs = torch.stack([times[valid], channels[valid].to(times.dtype)], dim=-1)
        return len(torch.unique(pairs, dim=0)) < len(pairs)

    @pytest.mark.parametrize(
        "data",
        [UNBATCHED_SIMPLE_DATA, BATCHED_SIMPLE_DATA],
    )
    def test_simple_data_has_strictly_increasing_split_timestamps(
        self, data: TensorViewData
    ) -> None:
        split = data["split"]

        for times in split["context_times"].unsqueeze(0).flatten(end_dim=-2):
            valid = times[times.isfinite()]
            assert valid.diff().gt(0.0).all()

        for times in split["query_times"].unsqueeze(0).flatten(end_dim=-2):
            valid = times[times.isfinite()]
            assert valid.diff().gt(0.0).all()

    @pytest.mark.parametrize(
        "data",
        [UNBATCHED_SPARSE_DATA, BATCHED_SPARSE_DATA],
    )
    def test_sparse_data_has_one_observation_per_split_row(
        self, data: TensorViewData
    ) -> None:
        split = data["split"]

        for times, mask in zip(
            split["context_times"].unsqueeze(0).flatten(end_dim=-2),
            split["context_mask"].unsqueeze(0).flatten(end_dim=-3),
            strict=True,
        ):
            valid = times.isfinite()
            counts = mask.sum(dim=-1)
            assert torch.equal(counts[valid], torch.ones_like(counts[valid]))

        for times, mask in zip(
            split["query_times"].unsqueeze(0).flatten(end_dim=-2),
            split["query_mask"].unsqueeze(0).flatten(end_dim=-3),
            strict=True,
        ):
            valid = times.isfinite()
            counts = mask.sum(dim=-1)
            assert torch.equal(counts[valid], torch.ones_like(counts[valid]))

    @pytest.mark.parametrize(
        "data",
        [UNBATCHED_GENERAL_DATA, BATCHED_GENERAL_DATA],
    )
    def test_general_data_has_duplicate_time_channel_pairs(
        self, data: TensorViewData
    ) -> None:
        triplet = data["triplet"]

        for times, channels in zip(
            triplet["context_times"].unsqueeze(0).flatten(end_dim=-2),
            triplet["context_channels"].unsqueeze(0).flatten(end_dim=-2),
            strict=True,
        ):
            assert self._has_duplicate_time_channel_pair(times, channels)

        for times, channels in zip(
            triplet["query_times"].unsqueeze(0).flatten(end_dim=-2),
            triplet["query_channels"].unsqueeze(0).flatten(end_dim=-2),
            strict=True,
        ):
            assert self._has_duplicate_time_channel_pair(times, channels)


class TestEventBatch:
    @pytest.mark.parametrize("batch_shape", BATCH_SHAPES.values())
    def test_query_indices_random(self, batch_shape: tuple[int, ...]) -> None:
        req = make_continuous_time_request(
            rng=3141,
            batch_shape=batch_shape,
            min_steps=1,
            max_steps=4,
            context_shape=(3,),
            output_shape=(4,),
            input_missingness=True,
            target_missingness=True,
        )
        assert req.target_values is not None
        event = EventBatch.from_request(
            context_times=req.context_times,
            context_values=req.context_values,
            context_mask=req.context_mask,
            query_times=req.query_times,
            query_mask=req.query_mask,
            target_values=req.target_values,
        )
        assert event.target_values is not None
        assert_close(
            event.target_values[event.query_indices],
            req.target_values,
            atol=0.0,
            rtol=0.0,
            equal_nan=True,
        )

    @pytest.mark.parametrize("batch_shape", BATCH_SHAPES.values())
    def test_query_indices_batch_last(self, batch_shape: tuple[int, ...]) -> None:
        req = make_continuous_time_request(
            rng=3141,
            batch_shape=batch_shape,
            min_steps=1,
            max_steps=4,
            context_shape=(3,),
            output_shape=(4,),
            input_missingness=True,
            target_missingness=True,
        )
        assert req.target_values is not None

        # Rearrange to batch_first=False layout: seq dim moves from position -1/-2 to 0.
        # Shapes: (*batch_shape, N) -> (N, *batch_shape), (*batch_shape, N, D) -> (N, *batch_shape, D)
        ctx_times = req.context_times.movedim(-1, 0)  # (N, *batch_shape)
        ctx_values = req.context_values.movedim(-2, 0)  # (N, *batch_shape, D)
        ctx_mask = req.context_mask.movedim(-2, 0)  # (N, *batch_shape, D)
        qry_times = req.query_times.movedim(-1, 0)  # (K, *batch_shape)
        qry_mask = req.query_mask.movedim(-2, 0)  # (K, *batch_shape, F)
        tgt_values = req.target_values.movedim(-2, 0)  # (K, *batch_shape, F)

        event = EventBatch.from_request(
            context_times=ctx_times,
            context_values=ctx_values,
            context_mask=ctx_mask,
            query_times=qry_times,
            query_mask=qry_mask,
            target_values=tgt_values,
            batch_first=False,
        )
        assert event.target_values is not None
        assert_close(
            event.target_values[event.query_indices],
            tgt_values,
            atol=0.0,
            rtol=0.0,
            equal_nan=True,
        )


class TestSplitTimeData:
    def test_eq_uses_tensor_value_comparison(self) -> None:
        lhs = SplitTimeData(
            context_times=torch.tensor([1.0, nan]),
            context_values=torch.tensor([[10.0, nan], [nan, nan]]),
            context_mask=torch.tensor([[True, False], [False, False]]),
            query_times=torch.tensor([2.0]),
            query_mask=torch.tensor([[True, False]]),
            target_values=torch.tensor([[20.0, nan]]),
            static_covariates=torch.tensor([3.0, nan]),
        )
        rhs = SplitTimeData(
            context_times=torch.tensor([1.0, nan]),
            context_values=torch.tensor([[10.0, nan], [nan, nan]]),
            context_mask=torch.tensor([[True, False], [False, False]]),
            query_times=torch.tensor([2.0]),
            query_mask=torch.tensor([[True, False]]),
            target_values=torch.tensor([[20.0, nan]]),
            static_covariates=torch.tensor([3.0, nan]),
        )
        other = SplitTimeData(
            context_times=torch.tensor([1.0, nan]),
            context_values=torch.tensor([[10.0, nan], [nan, nan]]),
            context_mask=torch.tensor([[True, False], [False, False]]),
            query_times=torch.tensor([2.0]),
            query_mask=torch.tensor([[False, True]]),
            target_values=torch.tensor([[nan, 20.0]]),
            static_covariates=torch.tensor([3.0, nan]),
        )

        assert lhs == rhs
        assert lhs != other

    def test_query_mask_clears_masked_values(self) -> None:
        arg = SplitTimeData(
            context_times=torch.tensor([1.0]),
            context_values=torch.tensor([[10.0, 11.0]]),
            context_mask=torch.tensor([[True, True]]),
            query_times=torch.tensor([2.0]),
            query_mask=torch.tensor([[True, False]]),
            target_values=torch.tensor([[20.0, 21.0]]),
        )
        assert_close(arg.target_values, torch.tensor([[20.0, nan]]), equal_nan=True)

        batched = SplitTimeData(
            context_times=torch.tensor([[1.0]]),
            context_values=torch.tensor([[[10.0, 11.0]]]),
            context_mask=torch.tensor([[[True, True]]]),
            query_times=torch.tensor([[2.0, nan]]),
            query_mask=torch.tensor([[[True, False], [False, False]]]),
            target_values=torch.tensor([[[20.0, 21.0], [nan, nan]]]),
        )
        assert_close(
            batched.target_values,
            torch.tensor([[[20.0, nan], [nan, nan]]]),
            equal_nan=True,
        )

    def test_context_mask_clears_masked_values(self) -> None:
        arg = SplitTimeData(
            context_times=torch.tensor([1.0]),
            context_values=torch.tensor([[10.0, 11.0]]),
            context_mask=torch.tensor([[True, False]]),
            query_times=torch.tensor([2.0]),
            query_mask=torch.tensor([[True, False]]),
        )

        assert_close(arg.context_values, torch.tensor([[10.0, nan]]), equal_nan=True)

    def test_is_simple(self) -> None:
        arg = SplitTimeData(
            context_times=torch.tensor([1.0, 3.0]),
            context_values=torch.tensor(
                [
                    [10.0, nan],
                    [nan, 30.0],
                ]
            ),
            context_mask=torch.tensor(
                [
                    [True, False],
                    [False, True],
                ]
            ),
            query_times=torch.tensor([2.0, 4.0]),
            query_mask=torch.tensor([[True], [True]]),
        )

        assert arg.is_simple()
        assert not replace(arg, context_times=torch.tensor([1.0, 1.0])).is_simple()
        assert not replace(arg, query_times=torch.tensor([2.0, 2.0])).is_simple()
        assert not SplitTimeData(
            context_times=torch.tensor([[1.0, nan], [2.0, nan]]),
            context_values=torch.tensor([[[10.0], [nan]], [[20.0], [nan]]]),
            context_mask=torch.tensor([[[True], [False]], [[True], [False]]]),
            query_times=torch.tensor([[3.0], [4.0]]),
            query_mask=torch.tensor([[[True]], [[True]]]),
        ).is_simple()

    def test_is_trimmed(self) -> None:
        assert SplitTimeData(
            context_times=torch.tensor([[1.0, 2.0], [3.0, nan]]),
            context_values=torch.tensor([[[10.0], [20.0]], [[30.0], [nan]]]),
            context_mask=torch.tensor([[[True], [True]], [[True], [False]]]),
            query_times=torch.tensor([[4.0], [5.0]]),
            query_mask=torch.tensor([[[True]], [[True]]]),
        ).is_trimmed()
        assert not SplitTimeData(
            context_times=torch.tensor([[1.0, nan], [2.0, nan]]),
            context_values=torch.tensor([[[10.0], [nan]], [[20.0], [nan]]]),
            context_mask=torch.tensor([[[True], [False]], [[True], [False]]]),
            query_times=torch.tensor([[3.0], [4.0]]),
            query_mask=torch.tensor([[[True]], [[True]]]),
        ).is_trimmed()

    @BATCH_PARAMETERS
    def test_to_split(self, batch_shape: tuple[int, ...], batch_first: bool) -> None:
        data = _simple_test_data(batch_shape, batch_first)

        actual = data["split"].to_split()

        assert actual.batch_shape == batch_shape
        assert actual.batch_first is batch_first
        assert actual == data["split"]

    @BATCH_PARAMETERS
    def test_to_merged(self, batch_shape: tuple[int, ...], batch_first: bool) -> None:
        data = _simple_test_data(batch_shape, batch_first)

        actual = data["split"].to_merged()

        assert actual.batch_shape == batch_shape
        assert actual.batch_first is batch_first
        assert actual == data["merged"]

    @BATCH_PARAMETERS
    def test_to_triplet(self, batch_shape: tuple[int, ...], batch_first: bool) -> None:
        data = _simple_test_data(batch_shape, batch_first)

        actual = data["split"].to_triplet()

        assert actual.batch_shape == batch_shape
        assert actual.batch_first is batch_first
        assert actual == data["triplet"]

    @BATCH_PARAMETERS
    def test_roundtrip_split(
        self, batch_shape: tuple[int, ...], batch_first: bool
    ) -> None:
        original = _simple_test_data(batch_shape, batch_first)["split"]

        actual = original.to_split().to_split()

        assert actual.batch_shape == batch_shape
        assert actual.batch_first is batch_first
        assert actual == original

    @BATCH_PARAMETERS
    def test_roundtrip_merged(
        self, batch_shape: tuple[int, ...], batch_first: bool
    ) -> None:
        original = _simple_test_data(batch_shape, batch_first)["split"]

        actual = original.to_merged().to_split()

        assert actual.batch_shape == batch_shape
        assert actual.batch_first is batch_first
        assert actual == original

    @BATCH_PARAMETERS
    def test_roundtrip_triplet(
        self, batch_shape: tuple[int, ...], batch_first: bool
    ) -> None:
        original = _simple_test_data(batch_shape, batch_first)["split"]

        actual = original.to_triplet().to_split()

        assert actual.batch_shape == batch_shape
        assert actual.batch_first is batch_first
        assert actual == original

    def test_rejects_batch_first_mismatch(self) -> None:
        original = TEST_DATA["simple", "single", True]["split"]

        with pytest.raises(ValueError, match=r"arg\.batch_first"):
            split_to_merged(original, batch_first=False)
        with pytest.raises(ValueError, match=r"arg\.batch_first"):
            split_to_triplet(original, batch_first=False)

    @pytest.mark.parametrize("batch_shape", BATCH_SHAPES.values())
    def test_roundtrip_batch_first(self, batch_shape: tuple[int, ...]) -> None:
        original = _simple_test_data(batch_shape, True)["split"]

        batch_last = original.to_batch_last()
        actual = batch_last.to_batch_first()

        assert batch_last.batch_shape == batch_shape
        assert not batch_last.batch_first
        assert batch_last == _simple_test_data(batch_shape, False)["split"]
        assert actual.batch_shape == batch_shape
        assert actual.batch_first
        assert actual == original

    @pytest.mark.parametrize("batch_shape", BATCH_SHAPES.values())
    def test_roundtrip_batch_last(self, batch_shape: tuple[int, ...]) -> None:
        original = _simple_test_data(batch_shape, False)["split"]

        batch_first = original.to_batch_first()
        actual = batch_first.to_batch_last()

        assert batch_first.batch_shape == batch_shape
        assert batch_first.batch_first
        assert batch_first == _simple_test_data(batch_shape, True)["split"]
        assert actual.batch_shape == batch_shape
        assert not actual.batch_first
        assert actual == original

    @pytest.mark.parametrize("batch_first", [True, False])
    def test_roundtrip_unbatch(self, batch_first: bool) -> None:
        original = TEST_DATA["simple", "single", batch_first]["split"]

        unbatched = original.unbatch()
        actual = SplitTimeData.from_unbatched(unbatched, batch_first=batch_first)

        assert all(arg.batch_first is batch_first for arg in unbatched)
        assert actual.batch_shape == original.batch_shape
        assert actual.batch_first is batch_first
        assert actual == original

    @pytest.mark.parametrize("batch_first", [True, False])
    def test_roundtrip_from_unbatched(self, batch_first: bool) -> None:
        originals = [
            TEST_DATA[data_type, "unbatched", batch_first]["split"]
            for data_type in ("simple", "sparse")
        ]

        batched = SplitTimeData.from_unbatched(originals, batch_first=batch_first)
        actual = batched.unbatch()

        assert batched.batch_shape == (len(originals),)
        assert batched.batch_first is batch_first
        assert all(arg.batch_first is batch_first for arg in actual)
        assert actual == originals

    def test_to_triplet_batched_without_mask(self) -> None:
        context_times = torch.tensor([[[1.0], [2.0]]])
        context_values = torch.tensor([[[[1.0, nan]], [[2.0, 3.0]]]])
        query_times = torch.tensor([[[4.0, nan], [5.0, 6.0]]])

        original = SplitTimeData(
            context_times=context_times,
            context_values=context_values,
            context_mask=context_values.isfinite(),
            query_times=query_times,
            query_mask=query_times.isfinite()
            .unsqueeze(-1)
            .expand(*query_times.shape, 2),
        )
        actual = original.to_triplet()
        actual.validate()

        expected = TripletTimeData(
            context_times=torch.tensor([[[1.0, nan], [2.0, 2.0]]]),
            context_channels=torch.tensor([[[0, -1], [0, 1]]]),
            context_values=torch.tensor([[[1.0, nan], [2.0, 3.0]]]),
            query_times=torch.tensor([[[4.0, 4.0, nan, nan], [5.0, 5.0, 6.0, 6.0]]]),
            query_channels=torch.tensor([[[0, 1, -1, -1], [0, 1, 0, 1]]]),
        )

        actual = TripletTimeData(
            context_times=actual.context_times,
            context_channels=actual.context_channels,
            context_values=actual.context_values,
            query_times=actual.query_times,
            query_channels=actual.query_channels,
            target_values=actual.target_values,
            static_covariates=actual.static_covariates,
        )

        assert actual == expected

    def test_roundtrip_to_triplet_duplicates(self) -> None:
        original = SplitTimeData(
            context_times=torch.tensor([
                [1.0, 1.0, 2.0],
                [0.0, 0.0, nan],
            ]),
            context_values=torch.tensor([
                [[10.0,  nan,  nan],
                 [ nan, 11.0,  nan],
                 [20.0,  nan, 22.0]],
                [[ 1.0,  nan,  nan],
                 [ nan,  2.0,  3.0],
                 [ nan,  nan,  nan]],
            ]),
            context_mask=torch.tensor([
                [[ True, False, False],
                 [False,  True, False],
                 [ True, False,  True]],
                [[ True, False, False],
                 [False,  True,  True],
                 [False, False, False]],
            ]),
            query_times=torch.tensor([
                [5.0, 6.0, 7.0],
                [8.0, 9.0, nan],
            ]),
            query_mask=torch.tensor([
                [[ True, False, False],
                 [False,  True, False],
                 [ True, False,  True]],
                [[ True, False, False],
                 [False,  True,  True],
                 [False, False, False]],
            ]),
            target_values=torch.tensor([
                [[50.0,  nan,  nan],
                 [ nan, 61.0,  nan],
                 [70.0,  nan, 72.0]],
                [[80.0,  nan,  nan],
                 [ nan, 91.0, 92.0],
                 [ nan,  nan,  nan]],
            ]),
        )  # fmt: skip

        triplet = original.to_triplet()
        actual = triplet.to_split(context_dim=3, query_dim=3)
        expected = SplitTimeData(
            context_times=torch.tensor([
                [1.0, 2.0],
                [0.0, nan],
            ]),
            context_values=torch.tensor([
                [[10.0, 11.0,  nan],
                 [20.0,  nan, 22.0]],
                [[ 1.0,  2.0,  3.0],
                 [ nan,  nan,  nan]],
            ]),
            context_mask=torch.tensor([
                [[ True,  True, False],
                 [ True, False,  True]],
                [[ True,  True,  True],
                 [False, False, False]],
            ]),
            query_times=torch.tensor([
                [5.0, 6.0, 7.0],
                [8.0, 9.0, nan],
            ]),
            query_mask=torch.tensor([
                [[ True, False, False],
                 [False,  True, False],
                 [ True, False,  True]],
                [[ True, False, False],
                 [False,  True,  True],
                 [False, False, False]],
            ]),
            target_values=torch.tensor([
                [[50.0,  nan,  nan],
                 [ nan, 61.0,  nan],
                 [70.0,  nan, 72.0]],
                [[80.0,  nan,  nan],
                 [ nan, 91.0, 92.0],
                 [ nan,  nan,  nan]],
            ]),
        )  # fmt: skip

        assert actual == expected
        assert actual.to_triplet() == triplet

    @pytest.mark.parametrize("batch_shape", BATCH_SHAPES.values())
    def test_roundtrip_to_merged_distinct_dims(
        self, batch_shape: tuple[int, ...]
    ) -> None:
        original = make_continuous_time_request(
            rng=3141,
            batch_shape=batch_shape,
            min_steps=4,
            max_steps=4,
            context_shape=(3,),
            output_shape=(4,),
            input_missingness=True,
            target_missingness=True,
        )

        combined = original.to_merged()
        assert combined.context_values.shape[-1] == 3
        assert combined.query_mask.shape[-1] == 4

        actual = combined.to_split()

        assert actual == original


class TestMergedTimeData:
    def test_eq_uses_tensor_value_comparison(self) -> None:
        lhs = MergedTimeData(
            timestamps=torch.tensor([1.0, 2.0, nan]),
            context_values=torch.tensor(
                [
                    [10.0, nan],
                    [nan, nan],
                    [nan, nan],
                ]
            ),
            context_mask=torch.tensor(
                [
                    [True, False],
                    [False, False],
                    [False, False],
                ]
            ),
            query_mask=torch.tensor(
                [
                    [False, False],
                    [True, False],
                    [False, False],
                ]
            ),
            target_values=torch.tensor(
                [
                    [nan, nan],
                    [20.0, nan],
                    [nan, nan],
                ]
            ),
            static_covariates=torch.tensor([3.0, nan]),
        )
        rhs = MergedTimeData(
            timestamps=torch.tensor([1.0, 2.0, nan]),
            context_values=torch.tensor(
                [
                    [10.0, nan],
                    [nan, nan],
                    [nan, nan],
                ]
            ),
            context_mask=torch.tensor(
                [
                    [True, False],
                    [False, False],
                    [False, False],
                ]
            ),
            query_mask=torch.tensor(
                [
                    [False, False],
                    [True, False],
                    [False, False],
                ]
            ),
            target_values=torch.tensor(
                [
                    [nan, nan],
                    [20.0, nan],
                    [nan, nan],
                ]
            ),
            static_covariates=torch.tensor([3.0, nan]),
        )
        other = MergedTimeData(
            timestamps=torch.tensor([1.0, 2.0, nan]),
            context_values=torch.tensor(
                [
                    [10.0, nan],
                    [nan, nan],
                    [nan, nan],
                ]
            ),
            context_mask=torch.tensor(
                [
                    [True, False],
                    [False, False],
                    [False, False],
                ]
            ),
            query_mask=torch.tensor(
                [
                    [False, False],
                    [False, True],
                    [False, False],
                ]
            ),
            target_values=torch.tensor(
                [
                    [nan, nan],
                    [nan, 20.0],
                    [nan, nan],
                ]
            ),
            static_covariates=torch.tensor([3.0, nan]),
        )

        assert lhs == rhs
        assert lhs != other

    def test_rejects_non_increasing_query_times(self) -> None:
        with pytest.raises(AssertionError):
            MergedTimeData(
                timestamps=torch.tensor([1.0, 1.0]),
                context_values=torch.tensor([
                    [1.0, nan],
                    [2.0, 3.0],
                ]),
                context_mask=torch.tensor([
                    [ True, False],
                    [False, False],
                ]),
                query_mask=torch.tensor([
                    [False,  True],
                    [ True, False],
                ]),
                target_values=torch.tensor([
                    [1.0, nan],
                    [2.0, 3.0],
                ]),
            )  # fmt: skip

    def test_rejects_mixed_query_value_availability(self) -> None:
        with pytest.raises(AssertionError):
            MergedTimeData(
                timestamps=torch.tensor([1.0, 2.0]),
                context_values=torch.tensor([
                    [1.0, nan],
                    [2.0, 3.0],
                ]),
                context_mask=torch.tensor([
                    [ True, False],
                    [False, False],
                ]),
                query_mask=torch.tensor([
                    [False,  True],
                    [ True, False],
                ]),
                target_values=torch.tensor([
                    [1.0, nan],
                    [2.0, 3.0],
                ]),
            )  # fmt: skip

    def test_is_simple(self) -> None:
        arg = MergedTimeData(
            timestamps=torch.tensor([1.0, 2.0, 4.0]),
            context_values=torch.tensor(
                [
                    [10.0, nan],
                    [nan, nan],
                    [nan, 40.0],
                ]
            ),
            context_mask=torch.tensor(
                [
                    [True, False],
                    [False, False],
                    [False, True],
                ]
            ),
            query_mask=torch.tensor(
                [
                    [False, False],
                    [True, False],
                    [False, False],
                ]
            ),
        )

        assert arg.is_simple()
        assert not replace(arg, timestamps=torch.tensor([1.0, 2.0, 2.0])).is_simple()
        assert not MergedTimeData(
            timestamps=torch.tensor([[1.0, nan], [2.0, nan]]),
            context_values=torch.tensor(
                [
                    [[10.0], [nan]],
                    [[20.0], [nan]],
                ]
            ),
            context_mask=torch.tensor(
                [
                    [[True], [False]],
                    [[True], [False]],
                ]
            ),
            query_mask=torch.tensor(
                [
                    [[False], [False]],
                    [[False], [False]],
                ]
            ),
        ).is_simple()

    def test_is_trimmed(self) -> None:
        assert MergedTimeData(
            timestamps=torch.tensor([[1.0, 2.0], [3.0, nan]]),
            context_values=torch.tensor(
                [
                    [[10.0], [nan]],
                    [[30.0], [nan]],
                ]
            ),
            context_mask=torch.tensor(
                [
                    [[True], [False]],
                    [[True], [False]],
                ]
            ),
            query_mask=torch.tensor(
                [
                    [[False], [True]],
                    [[False], [False]],
                ]
            ),
        ).is_trimmed()
        assert not MergedTimeData(
            timestamps=torch.tensor([[1.0, nan], [2.0, nan]]),
            context_values=torch.tensor(
                [
                    [[10.0], [nan]],
                    [[20.0], [nan]],
                ]
            ),
            context_mask=torch.tensor(
                [
                    [[True], [False]],
                    [[True], [False]],
                ]
            ),
            query_mask=torch.tensor(
                [
                    [[False], [False]],
                    [[False], [False]],
                ]
            ),
        ).is_trimmed()

    @BATCH_PARAMETERS
    def test_to_split(self, batch_shape: tuple[int, ...], batch_first: bool) -> None:
        data = _simple_test_data(batch_shape, batch_first)

        actual = data["merged"].to_split()

        assert actual.batch_shape == batch_shape
        assert actual.batch_first is batch_first
        assert actual == data["split"]

    @BATCH_PARAMETERS
    def test_to_merged(self, batch_shape: tuple[int, ...], batch_first: bool) -> None:
        data = _simple_test_data(batch_shape, batch_first)

        actual = data["merged"].to_merged()

        assert actual.batch_shape == batch_shape
        assert actual.batch_first is batch_first
        assert actual == data["merged"]

    @BATCH_PARAMETERS
    def test_to_triplet(self, batch_shape: tuple[int, ...], batch_first: bool) -> None:
        data = _simple_test_data(batch_shape, batch_first)

        actual = data["merged"].to_triplet()

        assert actual.batch_shape == batch_shape
        assert actual.batch_first is batch_first
        assert actual == data["triplet"]

    @BATCH_PARAMETERS
    def test_roundtrip_split(
        self, batch_shape: tuple[int, ...], batch_first: bool
    ) -> None:
        original = _simple_test_data(batch_shape, batch_first)["merged"]

        actual = original.to_split().to_merged()

        assert actual.batch_shape == batch_shape
        assert actual.batch_first is batch_first
        assert actual == original

    @BATCH_PARAMETERS
    def test_roundtrip_merged(
        self, batch_shape: tuple[int, ...], batch_first: bool
    ) -> None:
        original = _simple_test_data(batch_shape, batch_first)["merged"]

        actual = original.to_merged().to_merged()

        assert actual.batch_shape == batch_shape
        assert actual.batch_first is batch_first
        assert actual == original

    @BATCH_PARAMETERS
    def test_roundtrip_triplet(
        self, batch_shape: tuple[int, ...], batch_first: bool
    ) -> None:
        original = _simple_test_data(batch_shape, batch_first)["merged"]

        actual = original.to_triplet().to_merged()

        assert actual.batch_shape == batch_shape
        assert actual.batch_first is batch_first
        assert actual == original

    def test_rejects_batch_first_mismatch(self) -> None:
        original = TEST_DATA["simple", "single", True]["merged"]

        with pytest.raises(ValueError, match=r"arg\.batch_first"):
            merged_to_split(original, batch_first=False)
        with pytest.raises(ValueError, match=r"arg\.batch_first"):
            merged_to_triplet(original, batch_first=False)

    @pytest.mark.parametrize("batch_shape", BATCH_SHAPES.values())
    def test_roundtrip_batch_first(self, batch_shape: tuple[int, ...]) -> None:
        original = _simple_test_data(batch_shape, True)["merged"]

        batch_last = original.to_batch_last()
        actual = batch_last.to_batch_first()

        assert batch_last.batch_shape == batch_shape
        assert not batch_last.batch_first
        assert batch_last == _simple_test_data(batch_shape, False)["merged"]
        assert actual.batch_shape == batch_shape
        assert actual.batch_first
        assert actual == original

    @pytest.mark.parametrize("batch_shape", BATCH_SHAPES.values())
    def test_roundtrip_batch_last(self, batch_shape: tuple[int, ...]) -> None:
        original = _simple_test_data(batch_shape, False)["merged"]

        batch_first = original.to_batch_first()
        actual = batch_first.to_batch_last()

        assert batch_first.batch_shape == batch_shape
        assert batch_first.batch_first
        assert batch_first == _simple_test_data(batch_shape, True)["merged"]
        assert actual.batch_shape == batch_shape
        assert not actual.batch_first
        assert actual == original

    @pytest.mark.parametrize("batch_first", [True, False])
    def test_roundtrip_unbatch(self, batch_first: bool) -> None:
        original = TEST_DATA["simple", "single", batch_first]["merged"]

        unbatched = original.unbatch()
        actual = MergedTimeData.from_unbatched(unbatched, batch_first=batch_first)

        assert all(arg.batch_first is batch_first for arg in unbatched)
        assert actual.batch_shape == original.batch_shape
        assert actual.batch_first is batch_first
        assert actual == original

    @pytest.mark.parametrize("batch_first", [True, False])
    def test_roundtrip_from_unbatched(self, batch_first: bool) -> None:
        originals = [
            TEST_DATA[data_type, "unbatched", batch_first]["merged"]
            for data_type in ("simple", "sparse")
        ]

        batched = MergedTimeData.from_unbatched(originals, batch_first=batch_first)
        actual = batched.unbatch()

        assert batched.batch_shape == (len(originals),)
        assert batched.batch_first is batch_first
        assert all(arg.batch_first is batch_first for arg in actual)
        assert actual == originals

    @BATCH_PARAMETERS
    def test_to_split_without_target_values(
        self, batch_shape: tuple[int, ...], batch_first: bool
    ) -> None:
        data = _simple_test_data(batch_shape, batch_first)
        original = replace(data["merged"], target_values=None)

        actual = original.to_split()

        assert actual.batch_shape == batch_shape
        assert actual.batch_first is batch_first
        assert actual == replace(data["split"], target_values=None)

    @pytest.mark.parametrize("batch_shape", BATCH_SHAPES.values())
    def test_query_and_context_indices_match_split_time(
        self, batch_shape: tuple[int, ...]
    ) -> None:
        original = make_continuous_time_request(
            rng=3141,
            batch_shape=batch_shape,
            min_steps=1,
            max_steps=4,
            context_shape=(3,),
            output_shape=(4,),
            input_missingness=True,
            target_missingness=True,
        )
        assert original.target_values is not None

        merged_data = original.to_merged()
        assert merged_data.target_values is not None

        query_indices = merged_data.query_indices
        context_indices = merged_data.context_indices

        assert_close(
            original.query_times,
            merged_data.timestamps[query_indices],
            atol=0.0,
            rtol=0.0,
            equal_nan=True,
        )
        assert_close(
            original.query_mask,
            merged_data.query_mask[query_indices],
            atol=0.0,
            rtol=0.0,
            equal_nan=True,
        )
        assert_close(
            original.target_values,
            merged_data.target_values[query_indices],
            atol=0.0,
            rtol=0.0,
            equal_nan=True,
        )
        assert_close(
            original.context_times,
            merged_data.timestamps[context_indices],
            atol=0.0,
            rtol=0.0,
            equal_nan=True,
        )
        assert_close(
            original.context_mask,
            merged_data.context_mask[context_indices],
            atol=0.0,
            rtol=0.0,
            equal_nan=True,
        )
        assert_close(
            original.context_values,
            merged_data.context_values[context_indices],
            atol=0.0,
            rtol=0.0,
            equal_nan=True,
        )


class TestTripletTimeData:
    def test_eq_uses_tensor_value_comparison(self) -> None:
        lhs = TripletTimeData(
            context_times=torch.tensor([1.0, nan]),
            context_channels=torch.tensor([0, -1]),
            context_values=torch.tensor([10.0, nan]),
            query_times=torch.tensor([2.0, nan]),
            query_channels=torch.tensor([1, -1]),
            target_values=torch.tensor([20.0, nan]),
            static_covariates=torch.tensor([3.0, nan]),
        )
        rhs = TripletTimeData(
            context_times=torch.tensor([1.0, nan]),
            context_channels=torch.tensor([0, -1]),
            context_values=torch.tensor([10.0, nan]),
            query_times=torch.tensor([2.0, nan]),
            query_channels=torch.tensor([1, -1]),
            target_values=torch.tensor([20.0, nan]),
            static_covariates=torch.tensor([3.0, nan]),
        )
        other = TripletTimeData(
            context_times=torch.tensor([1.0, nan]),
            context_channels=torch.tensor([1, -1]),
            context_values=torch.tensor([10.0, nan]),
            query_times=torch.tensor([2.0, nan]),
            query_channels=torch.tensor([1, -1]),
            target_values=torch.tensor([20.0, nan]),
            static_covariates=torch.tensor([3.0, nan]),
        )

        assert lhs == rhs
        assert lhs != other

    def test_is_simple(self) -> None:
        arg = TripletTimeData(
            context_times=torch.tensor([1.0, 1.0, 2.0]),
            context_channels=torch.tensor([0, 1, 0]),
            context_values=torch.tensor([10.0, 11.0, 20.0]),
            query_times=torch.tensor([3.0, 3.0, 4.0]),
            query_channels=torch.tensor([0, 1, 0]),
            target_values=torch.tensor([30.0, 31.0, 40.0]),
        )

        assert arg.is_simple()
        assert replace(
            arg,
            context_channels=torch.tensor([0, 0, 0]),
        ).is_simple()
        assert TripletTimeData(
            context_times=torch.tensor([1.0]),
            context_channels=torch.tensor([0]),
            context_values=torch.tensor([10.0]),
            query_times=torch.tensor([2.0, 2.0]),
            query_channels=torch.tensor([1, 1]),
            target_values=torch.tensor([20.0, 21.0]),
            validate_args=False,
        ).is_simple()
        assert not TripletTimeData(
            context_times=torch.tensor([[1.0, nan], [2.0, nan]]),
            context_channels=torch.tensor([[0, -1], [1, -1]]),
            context_values=torch.tensor([[10.0, nan], [20.0, nan]]),
            query_times=torch.tensor([[3.0], [4.0]]),
            query_channels=torch.tensor([[0], [0]]),
            target_values=torch.tensor([[30.0], [40.0]]),
        ).is_simple()

    @pytest.mark.parametrize("case", [key for key in TEST_DATA if key[0] == "simple"])
    def test_query_indices_recover_simple_split_target_layout(
        self, case: tuple[DataType, BatchType, bool]
    ) -> None:
        split = TEST_DATA[case]["split"]
        triplet = split.to_triplet()
        y = triplet.target_values
        assert y is not None

        actual = y[triplet.query_indices].masked_fill(~split.query_mask, nan)

        assert_close(actual, split.target_values, atol=0.0, rtol=0.0, equal_nan=True)

    def test_is_trimmed(self) -> None:
        assert TripletTimeData(
            context_times=torch.tensor([[1.0, 2.0], [3.0, nan]]),
            context_channels=torch.tensor([[0, 0], [1, -1]]),
            context_values=torch.tensor([[10.0, 20.0], [30.0, nan]]),
            query_times=torch.tensor([[4.0], [5.0]]),
            query_channels=torch.tensor([[0], [1]]),
            target_values=torch.tensor([[40.0], [50.0]]),
        ).is_trimmed()
        assert not TripletTimeData(
            context_times=torch.tensor([[1.0, nan], [2.0, nan]]),
            context_channels=torch.tensor([[0, -1], [1, -1]]),
            context_values=torch.tensor([[10.0, nan], [20.0, nan]]),
            query_times=torch.tensor([[3.0], [4.0]]),
            query_channels=torch.tensor([[0], [1]]),
            target_values=torch.tensor([[30.0], [40.0]]),
        ).is_trimmed()

    @BATCH_PARAMETERS
    def test_to_split(self, batch_shape: tuple[int, ...], batch_first: bool) -> None:
        data = _simple_test_data(batch_shape, batch_first)

        actual = data["triplet"].to_split()

        assert actual.batch_shape == batch_shape
        assert actual.batch_first is batch_first
        assert actual == data["split"]

    @BATCH_PARAMETERS
    def test_to_merged(self, batch_shape: tuple[int, ...], batch_first: bool) -> None:
        data = _simple_test_data(batch_shape, batch_first)

        actual = data["triplet"].to_merged()

        assert actual.batch_shape == batch_shape
        assert actual.batch_first is batch_first
        assert actual == data["merged"]

    @BATCH_PARAMETERS
    def test_to_triplet(self, batch_shape: tuple[int, ...], batch_first: bool) -> None:
        data = _simple_test_data(batch_shape, batch_first)

        actual = data["triplet"].to_triplet()

        assert actual.batch_shape == batch_shape
        assert actual.batch_first is batch_first
        assert actual == data["triplet"]

    @BATCH_PARAMETERS
    def test_roundtrip_split(
        self, batch_shape: tuple[int, ...], batch_first: bool
    ) -> None:
        original = _simple_test_data(batch_shape, batch_first)["triplet"]

        actual = original.to_split().to_triplet()

        assert actual.batch_shape == batch_shape
        assert actual.batch_first is batch_first
        assert actual == original

    @BATCH_PARAMETERS
    def test_roundtrip_merged(
        self, batch_shape: tuple[int, ...], batch_first: bool
    ) -> None:
        original = _simple_test_data(batch_shape, batch_first)["triplet"]

        actual = original.to_merged().to_triplet()

        assert actual.batch_shape == batch_shape
        assert actual.batch_first is batch_first
        assert actual == original

    @BATCH_PARAMETERS
    def test_roundtrip_triplet(
        self, batch_shape: tuple[int, ...], batch_first: bool
    ) -> None:
        original = _simple_test_data(batch_shape, batch_first)["triplet"]

        actual = original.to_triplet().to_triplet()

        assert actual.batch_shape == batch_shape
        assert actual.batch_first is batch_first
        assert actual == original

    def test_rejects_batch_first_mismatch(self) -> None:
        original = TEST_DATA["simple", "single", True]["triplet"]

        with pytest.raises(ValueError, match=r"arg\.batch_first"):
            triplet_to_split(original, batch_first=False)
        with pytest.raises(ValueError, match=r"arg\.batch_first"):
            triplet_to_merged(original, batch_first=False)

    @pytest.mark.parametrize("batch_shape", BATCH_SHAPES.values())
    def test_roundtrip_batch_first(self, batch_shape: tuple[int, ...]) -> None:
        original = _simple_test_data(batch_shape, True)["triplet"]

        batch_last = original.to_batch_last()
        actual = batch_last.to_batch_first()

        assert batch_last.batch_shape == batch_shape
        assert not batch_last.batch_first
        assert batch_last == _simple_test_data(batch_shape, False)["triplet"]
        assert actual.batch_shape == batch_shape
        assert actual.batch_first
        assert actual == original

    @pytest.mark.parametrize("batch_shape", BATCH_SHAPES.values())
    def test_roundtrip_batch_last(self, batch_shape: tuple[int, ...]) -> None:
        original = _simple_test_data(batch_shape, False)["triplet"]

        batch_first = original.to_batch_first()
        actual = batch_first.to_batch_last()

        assert batch_first.batch_shape == batch_shape
        assert batch_first.batch_first
        assert batch_first == _simple_test_data(batch_shape, True)["triplet"]
        assert actual.batch_shape == batch_shape
        assert not actual.batch_first
        assert actual == original

    @pytest.mark.parametrize("batch_first", [True, False])
    def test_roundtrip_unbatch(self, batch_first: bool) -> None:
        original = TEST_DATA["simple", "single", batch_first]["triplet"]

        unbatched = original.unbatch()
        actual = TripletTimeData.from_unbatched(unbatched, batch_first=batch_first)

        assert all(arg.batch_first is batch_first for arg in unbatched)
        assert actual.batch_shape == original.batch_shape
        assert actual.batch_first is batch_first
        assert actual == original

    @pytest.mark.parametrize("batch_first", [True, False])
    def test_roundtrip_from_unbatched(self, batch_first: bool) -> None:
        originals = [
            TEST_DATA[data_type, "unbatched", batch_first]["triplet"]
            for data_type in ("simple", "sparse")
        ]

        batched = TripletTimeData.from_unbatched(originals, batch_first=batch_first)
        actual = batched.unbatch()

        assert batched.batch_shape == (len(originals),)
        assert batched.batch_first is batch_first
        assert all(arg.batch_first is batch_first for arg in actual)
        assert actual == originals

    def test_to_split_uses_channel_order_for_equal_timestamps(self) -> None:
        original = TripletTimeData(
            context_times=torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0]),
            context_channels=torch.tensor([0, 1, 0, 3, 2]),
            context_values=torch.tensor([2.0, 2.0, 3.0, 7.0, 6.0]),
            query_times=torch.tensor([2.0, 2.0]),
            query_channels=torch.tensor([0, 0]),
            target_values=torch.tensor([8.0, 9.0]),
        )

        actual = original.to_split(context_dim=4, query_dim=1)
        expected = SplitTimeData(
            context_times=torch.tensor([1.0, 1.0, 1.0]),
            context_values=torch.tensor([
                [2.0, 2.0, nan, nan],
                [3.0, nan, nan, 7.0],
                [nan, nan, 6.0, nan],
            ]),
            context_mask=torch.tensor([
                [ True,  True, False, False],
                [ True, False, False,  True],
                [False, False,  True, False],
            ]),
            query_times=torch.tensor([2.0, 2.0]),
            query_mask=torch.tensor([[True], [True]]),
            target_values=torch.tensor([[8.0], [9.0]]),
        )  # fmt: skip

        assert actual == expected
        assert actual.to_triplet() == original

    def test_to_split_unbatched_with_dims(self) -> None:
        original = TripletTimeData(
            context_times=torch.tensor([1.0, 2.0]),
            context_channels=torch.tensor([0, 1]),
            context_values=torch.tensor([10.0, 20.0]),
            query_times=torch.tensor([3.0]),
            query_channels=torch.tensor([1]),
            target_values=torch.tensor([30.0]),
        )

        actual = original.to_split(context_dim=3, query_dim=4)
        expected = SplitTimeData(
            context_times=torch.tensor([1.0, 2.0]),
            context_values=torch.tensor([
                [10.0,  nan, nan],
                [ nan, 20.0, nan],
            ]),
            context_mask=torch.tensor([
                [ True, False, False],
                [False,  True, False],
            ]),
            query_times=torch.tensor([3.0]),
            query_mask=torch.tensor([[False, True, False, False]]),
            target_values=torch.tensor([[nan, 30.0, nan, nan]]),
        )  # fmt: skip

        assert actual == expected

    def test_to_split_batched_without_values(self) -> None:
        original = TripletTimeData(
            context_times=torch.tensor([
                [1.0, nan],
                [2.0, 2.0],
            ]),
            context_channels=torch.tensor([
                [0, -1],
                [0,  1],
            ]),
            context_values=torch.tensor([
                [10.0,  nan],
                [20.0, 21.0],
            ]),
            query_times=torch.tensor([
                [3.0, 4.0],
                [5.0, nan],
            ]),
            query_channels=torch.tensor([
                [0,  1],
                [1, -1],
            ]),
        )  # fmt: skip

        actual = original.to_split(context_dim=2, query_dim=2)
        expected = SplitTimeData(
            context_times=torch.tensor([
                [1.0],
                [2.0],
            ]),
            context_values=torch.tensor([
                [[10.0,  nan]],
                [[20.0, 21.0]],
            ]),
            context_mask=torch.tensor([
                [[ True, False]],
                [[ True,  True]],
            ]),
            query_times=torch.tensor([
                [3.0, 4.0],
                [5.0, nan],
            ]),
            query_mask=torch.tensor([
                [[ True, False], [False,  True]],
                [[False,  True], [False, False]],
            ]),
        )  # fmt: skip

        assert actual == expected

    def test_roundtrip_from_unbatched_without_target_values(self) -> None:
        args = [
            TripletTimeData(
                context_times=torch.tensor([1.0]),
                context_channels=torch.tensor([0]),
                context_values=torch.tensor([10.0]),
                query_times=torch.tensor([3.0, 4.0]),
                query_channels=torch.tensor([0, 1]),
            ),
            TripletTimeData(
                context_times=torch.tensor([5.0, 5.0]),
                context_channels=torch.tensor([0, 1]),
                context_values=torch.tensor([50.0, 51.0]),
                query_times=torch.tensor([6.0]),
                query_channels=torch.tensor([2]),
            ),
        ]

        actual = TripletTimeData.from_unbatched(args, batch_first=True)
        expected = TripletTimeData(
            context_times=torch.tensor([
                [1.0, nan],
                [5.0, 5.0],
            ]),
            context_channels=torch.tensor([
                [0, -1],
                [0,  1],
            ]),
            context_values=torch.tensor([
                [10.0,  nan],
                [50.0, 51.0],
            ]),
            query_times=torch.tensor([
                [3.0, 4.0],
                [6.0, nan],
            ]),
            query_channels=torch.tensor([
                [0,  1],
                [2, -1],
            ]),
        )  # fmt: skip

        assert actual == expected

        unbatched = actual.unbatch()
        assert isinstance(unbatched, list)
        assert len(unbatched) == len(args)
        for actual, expected in zip(unbatched, args, strict=True):
            assert actual == expected

    @pytest.mark.parametrize("batch_shape", BATCH_SHAPES.values())
    def test_from_request_random(self, batch_shape: tuple[int, ...]) -> None:
        req = make_continuous_time_request(
            rng=3141,
            batch_shape=batch_shape,
            min_steps=1,
            max_steps=4,
            context_shape=(3,),
            output_shape=(4,),
            input_missingness=True,
            target_missingness=True,
        )

        actual = TripletTimeData.from_request(
            context_times=req.context_times,
            context_values=req.context_values,
            context_mask=req.context_mask,
            query_times=req.query_times,
            query_mask=req.query_mask,
            target_values=req.target_values,
            static_covariates=req.static_covariates,
        )
        expected = req.to_triplet()

        actual = TripletTimeData(
            context_times=actual.context_times,
            context_channels=actual.context_channels,
            context_values=actual.context_values,
            query_times=actual.query_times,
            query_channels=actual.query_channels,
            target_values=actual.target_values,
            static_covariates=actual.static_covariates,
        )

        assert actual == expected

    @pytest.mark.parametrize("batch_shape", BATCH_SHAPES.values())
    def test_from_request_batch_last(self, batch_shape: tuple[int, ...]) -> None:
        req = make_continuous_time_request(
            rng=3141,
            batch_shape=batch_shape,
            min_steps=1,
            max_steps=4,
            context_shape=(3,),
            output_shape=(4,),
            input_missingness=True,
            target_missingness=True,
            batch_first=False,
        )

        actual = TripletTimeData.from_request(
            context_times=req.context_times,
            context_values=req.context_values,
            context_mask=req.context_mask,
            query_times=req.query_times,
            query_mask=req.query_mask,
            target_values=req.target_values,
            static_covariates=req.static_covariates,
            batch_first=False,
        )
        expected = req.to_triplet()
        expected.validate()

        assert actual == expected
