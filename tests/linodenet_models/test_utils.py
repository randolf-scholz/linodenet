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
    JointTimeData,
    SplitTimeData,
    TripletTimeData,
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
        context_steps=context_steps,
        context_values=context_values,
        context_mask=context_mask,
        query_steps=query_steps,
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
    joint: JointTimeData
    triplet: TripletTimeData


class TensorViewData(TypedDict, closed=True):
    split: dict[str, Tensor]
    joint: dict[str, Tensor]
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
type DataFormat = Literal["split", "joint", "triplet"]

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
    "joint": JointTimeData(
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
    "joint": JointTimeData(
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
    "joint": {
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
    "joint": {
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
    "joint": {
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
    "joint": {
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
    "joint": {
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
    "joint": {
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
        "joint": {
            key: (
                tensor
                if key == "static_covariates"
                else tensor.movedim(-1, 0)
                if key == "timestamps"
                else tensor.movedim(-2, 0)
            )
            for key, tensor in data["joint"].items()
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
    joint_data = data["joint"]
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
    JointTimeData(
        timestamps=joint_data["timestamps"],
        context_mask=joint_data["context_mask"],
        context_values=joint_data["context_values"],
        query_mask=joint_data["query_mask"],
        target_values=joint_data["target_values"],
        static_covariates=joint_data["static_covariates"],
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
    joint_data = data["joint"]
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
        "joint": JointTimeData(
            timestamps=joint_data["timestamps"],
            context_mask=joint_data["context_mask"],
            context_values=joint_data["context_values"],
            query_mask=joint_data["query_mask"],
            target_values=joint_data["target_values"],
            static_covariates=joint_data["static_covariates"],
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


def _make_random_batched_triplet(batch_shape: tuple[int, ...], /) -> TripletTimeData:
    num_samples = 1
    for size in batch_shape:
        num_samples *= size

    generator = torch.Generator().manual_seed(
        1729 + sum((k + 1) * size for k, size in enumerate(batch_shape))
    )
    context_dim = 3
    query_dim = 4
    num_context_steps = 4
    num_query_steps = 3
    num_context = num_context_steps * context_dim
    num_query = num_query_steps * query_dim

    context_times = torch.full((num_samples, num_context), nan)
    context_channels = torch.full((num_samples, num_context), -1, dtype=torch.long)
    context_values = torch.full((num_samples, num_context), nan)
    query_times = torch.full((num_samples, num_query), nan)
    query_channels = torch.full((num_samples, num_query), -1, dtype=torch.long)
    target_values = torch.full((num_samples, num_query), nan)

    for sample in range(num_samples):
        context_steps = (
            num_context_steps
            if sample == 0
            else int(torch.randint(1, num_context_steps + 1, (), generator=generator))
        )
        index = 0
        for step in range(context_steps):
            if sample == 0:
                channels = torch.arange(context_dim)
            else:
                mask = torch.rand(context_dim, generator=generator) < 0.7
                if not mask.any():
                    mask[int(torch.randint(context_dim, (), generator=generator))] = (
                        True
                    )
                channels = mask.nonzero(as_tuple=True)[0]

            for channel in channels:
                context_times[sample, index] = float(step + 1)
                context_channels[sample, index] = channel
                context_values[sample, index] = float(
                    sample * 100 + step * 10 + channel
                )
                index += 1

        query_steps = (
            num_query_steps
            if sample == 0
            else int(torch.randint(1, num_query_steps + 1, (), generator=generator))
        )
        index = 0
        for step in range(query_steps):
            if sample == 0:
                channels = torch.arange(query_dim)
            else:
                mask = torch.rand(query_dim, generator=generator) < 0.7
                if not mask.any():
                    mask[int(torch.randint(query_dim, (), generator=generator))] = True
                channels = mask.nonzero(as_tuple=True)[0]

            for channel in channels:
                query_times[sample, index] = float(10 + step)
                query_channels[sample, index] = channel
                target_values[sample, index] = float(
                    1000 + sample * 100 + step * 10 + channel
                )
                index += 1

    static_covariates = torch.randn((num_samples, 2), generator=generator)
    return TripletTimeData(
        context_times=context_times.reshape(*batch_shape, num_context),
        context_channels=context_channels.reshape(*batch_shape, num_context),
        context_values=context_values.reshape(*batch_shape, num_context),
        query_times=query_times.reshape(*batch_shape, num_query),
        query_channels=query_channels.reshape(*batch_shape, num_query),
        target_values=target_values.reshape(*batch_shape, num_query),
        static_covariates=static_covariates.reshape(*batch_shape, 2),
    )


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
        self,
        data: TensorViewData,
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
        self,
        data: TensorViewData,
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
        self,
        data: TensorViewData,
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

    def test_to_triplet_unbatched(self) -> None:
        original = SplitTimeData(
            context_times=torch.tensor([1.0, 2.0]),
            context_values=torch.tensor([
                [10.0, nan, 11.0],
                [nan, 20.0, 21.0],
            ]),
            context_mask=torch.tensor([
                [ True, False,  True],
                [False,  True,  True],
            ]),
            query_times=torch.tensor([3.0, 4.0]),
            query_mask=torch.tensor([[True, False], [True, True]]),
            target_values=torch.tensor([[30.0, nan], [40.0, 41.0]]),
            static_covariates=torch.tensor([5.0, 6.0]),
        )  # fmt: skip

        actual = original.to_triplets()
        expected = TripletTimeData(
            context_times=torch.tensor([1.0, 1.0, 2.0, 2.0]),
            context_channels=torch.tensor([0, 2, 1, 2]),
            context_values=torch.tensor([10.0, 11.0, 20.0, 21.0]),
            query_times=torch.tensor([3.0, 4.0, 4.0]),
            query_channels=torch.tensor([0, 0, 1]),
            target_values=torch.tensor([30.0, 40.0, 41.0]),
            static_covariates=torch.tensor([5.0, 6.0]),
        )

        assert actual == expected

    def test_to_combined_unbatched(self) -> None:
        original = SplitTimeData(
            context_times=torch.tensor([1.0, 3.0]),
            context_values=torch.tensor([
                [10.0,  nan],
                [ nan, 30.0],
            ]),
            context_mask=torch.tensor([
                [ True, False],
                [False,  True],
            ]),
            query_times=torch.tensor([2.0, 4.0]),
            query_mask=torch.tensor([
                [True, False],
                [True,  True],
            ]),
            target_values=torch.tensor([
                [20.0,  nan],
                [40.0, 41.0],
            ]),
            static_covariates=torch.tensor([5.0, 6.0]),
        )  # fmt: skip

        actual = original.to_joint_time()
        expected = JointTimeData(
            timestamps=torch.tensor([1.0, 2.0, 3.0, 4.0]),
            context_values=torch.tensor([
                [10.0,  nan],
                [20.0,  nan],
                [ nan, 30.0],
                [40.0, 41.0],
            ]),
            context_mask=torch.tensor([
                [ True, False],
                [False, False],
                [False,  True],
                [False, False],
            ]),
            query_mask=torch.tensor([
                [False, False],
                [ True, False],
                [False, False],
                [ True,  True],
            ]),
            target_values=torch.tensor([
                [10.0,  nan],
                [20.0,  nan],
                [ nan, 30.0],
                [40.0, 41.0],
            ]),
            static_covariates=torch.tensor([5.0, 6.0]),
        )  # fmt: skip

        assert actual == expected

    def test_to_combined_roundtrip_unbatched_distinct_dims(self) -> None:
        original = SplitTimeData(
            context_times=torch.tensor([1.0, 3.0]),
            context_values=torch.tensor([
                [10.0,  nan],
                [ nan, 30.0],
            ]),
            context_mask=torch.tensor([
                [ True, False],
                [False,  True],
            ]),
            query_times=torch.tensor([2.0, 4.0]),
            query_mask=torch.tensor([
                [ True, False,  True],
                [False,  True,  True],
            ]),
            target_values=torch.tensor([
                [20.0,  nan, 22.0],
                [ nan, 41.0, 42.0],
            ]),
        )  # fmt: skip

        combined = original.to_joint_time()
        assert combined.context_values.shape == (4, 2)
        assert combined.query_mask.shape == (4, 3)

        actual = combined.to_split_time()

        assert actual == original

    def test_batched_roundtrip(self) -> None:
        args = [
            SplitTimeData(
                context_times=torch.tensor([1.0, 2.0]),
                context_values=torch.tensor([[1.0, nan], [2.0, 3.0]]),
                context_mask=torch.tensor([[True, False], [True, True]]),
                query_times=torch.tensor([5.0]),
                query_mask=torch.tensor([[True, False]]),
                target_values=torch.tensor([[9.0, nan]]),
                static_covariates=torch.tensor([1.0, 2.0]),
            ),
            SplitTimeData(
                context_times=torch.tensor([3.0]),
                context_values=torch.tensor([[4.0, 5.0]]),
                context_mask=torch.tensor([[True, True]]),
                query_times=torch.tensor([6.0, 7.0, 8.0]),
                query_mask=torch.tensor([[False, True], [True, False], [True, True]]),
                target_values=torch.tensor([[nan, 6.0], [7.0, nan], [8.0, 9.0]]),
                static_covariates=torch.tensor([3.0, 4.0]),
            ),
        ]

        actual = SplitTimeData.from_unbatched(args)
        expected = SplitTimeData(
            context_times=torch.tensor([
                [1.0, 2.0],
                [3.0, nan],
            ]),
            context_values=torch.tensor([
                [[1.0, nan], [2.0, 3.0]],
                [[4.0, 5.0], [nan, nan]],
            ]),
            context_mask=torch.tensor([
                [[ True, False], [ True,  True]],
                [[ True,  True], [False, False]],
            ]),
            query_times=torch.tensor([
                [5.0, nan, nan],
                [6.0, 7.0, 8.0],
            ]),
            query_mask=torch.tensor([
                [[ True, False], [False, False], [False, False]],
                [[False,  True], [ True, False], [ True,  True]],
            ]),
            target_values=torch.tensor([
                [[9.0, nan], [nan, nan], [nan, nan]],
                [[nan, 6.0], [7.0, nan], [8.0, 9.0]],
            ]),
            static_covariates=torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        )  # fmt: skip

        assert actual == expected

        unbatched = actual.unbatch()

        assert isinstance(unbatched, list)
        assert len(unbatched) == len(args)
        for actual, expected in zip(unbatched, args, strict=True):
            assert actual == expected

    def test_to_triplet_batched_with_mask(self) -> None:
        original = SplitTimeData(
            context_times=torch.tensor([
                [1.0, 2.0, nan],
                [0.0, 1.0, 2.0],
            ]),
            context_values=torch.tensor([
                [[10.0,  nan],
                 [ nan, 20.0],
                 [ nan,  nan]],
                [[ nan,  1.0],
                 [ 2.0,  3.0],
                 [ 4.0,  nan]],
            ]),
            context_mask=torch.tensor([
                [[ True, False],
                 [False,  True],
                 [False, False]],
                [[False,  True],
                 [ True,  True],
                 [ True, False]],
            ]),
            query_times=torch.tensor([
                [3.0, nan],
                [4.0, 5.0],
            ]),
            query_mask=torch.tensor([
                [[ True, False], [False, False]],
                [[False,  True], [ True,  True]],
            ]),
            target_values=torch.tensor([
                [[30.0,  nan], [ nan,  nan]],
                [[ nan, 40.0], [50.0, 60.0]],
            ]),
        )  # fmt: skip
        actual = original.to_triplets()

        expected = TripletTimeData(
            context_times=torch.tensor([
                [1.0, 2.0, nan, nan],
                [0.0, 1.0, 1.0, 2.0],
            ]),
            context_channels=torch.tensor([
                [0, 1, -1, -1],
                [1, 0,  1,  0],
            ]),
            context_values=torch.tensor([
                [10.0, 20.0, nan, nan],
                [ 1.0,  2.0, 3.0, 4.0],
            ]),
            query_times=torch.tensor([
                [3.0, nan, nan],
                [4.0, 5.0, 5.0],
            ]),
            query_channels=torch.tensor([
                [0, -1, -1],
                [1,  0,  1]
            ]),
            target_values=torch.tensor([
                [30.0,  nan,  nan],
                [40.0, 50.0, 60.0],
            ]),
        )  # fmt: skip

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
        actual = original.to_triplets()
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

    def test_to_triplet_roundtrip_batched_duplicates(self) -> None:
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

        triplet = original.to_triplets()
        actual = triplet.to_split_time(context_dim=3, query_dim=3)
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
        assert actual.to_triplets() == triplet

    @pytest.mark.parametrize("batch_shape", BATCH_SHAPES.values())
    def test_to_triplet_roundtrip(
        self,
        batch_shape: tuple[int, ...],
    ) -> None:
        original = make_continuous_time_request(
            rng=3141,
            batch_shape=batch_shape,
            min_steps=4,
            max_steps=4,
            context_shape=(3,),
            output_shape=(3,),
            input_missingness=True,
            target_missingness=True,
        )

        actual = original.to_triplets().to_split_time(
            context_dim=original.context_values.shape[-1],
            query_dim=3,
        )

        assert actual == original

    @pytest.mark.parametrize("batch_shape", BATCH_SHAPES.values())
    def test_to_combined_roundtrip_distinct_dims(
        self,
        batch_shape: tuple[int, ...],
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

        combined = original.to_joint_time()
        assert combined.context_values.shape[-1] == 3
        assert combined.query_mask.shape[-1] == 4

        actual = combined.to_split_time()

        assert actual == original


class TestJointTimeData:
    def test_eq_uses_tensor_value_comparison(self) -> None:
        lhs = JointTimeData(
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
        rhs = JointTimeData(
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
        other = JointTimeData(
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
            JointTimeData(
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
            JointTimeData(
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
        arg = JointTimeData(
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
        assert not JointTimeData(
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
        assert JointTimeData(
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
        assert not JointTimeData(
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

    def test_batched_roundtrip(self) -> None:
        expected = BATCHED_TEST_DATA["joint"]
        args = expected.unbatch()

        actual = JointTimeData.from_unbatched(args)

        assert actual == expected

        unbatched = actual.unbatch()
        assert isinstance(unbatched, list)
        assert len(unbatched) == len(args)
        for actual, expected in zip(unbatched, args, strict=True):
            assert actual == expected

    def test_to_dense_unbatched(self) -> None:
        original = UNBATCHED_TEST_DATA["joint"]

        actual = original.to_split_time()
        expected = UNBATCHED_TEST_DATA["split"]

        assert actual == expected

    def test_to_dense_unbatched_without_target_values(self) -> None:
        original = replace(UNBATCHED_TEST_DATA["joint"], target_values=None)

        actual = original.to_split_time()
        expected = replace(UNBATCHED_TEST_DATA["split"], target_values=None)

        assert actual == expected

    def test_to_dense_batched_without_target_values(self) -> None:
        original = replace(BATCHED_TEST_DATA["joint"], target_values=None)

        actual = original.to_split_time()
        expected = replace(BATCHED_TEST_DATA["split"], target_values=None)

        assert actual == expected

    @pytest.mark.parametrize("batch_shape", BATCH_SHAPES.values())
    def test_to_dense_roundtrip(self, batch_shape: tuple[int, ...]) -> None:
        original = make_continuous_time_request(
            rng=3141,
            batch_shape=batch_shape,
            min_steps=4,
            max_steps=4,
            context_shape=(3,),
            output_shape=(3,),
            input_missingness=True,
            target_missingness=True,
        ).to_joint_time()

        actual = original.to_split_time().to_joint_time()

        assert actual == original

    @pytest.mark.parametrize("batch_shape", BATCH_SHAPES.values())
    def test_to_triplet_roundtrip(self, batch_shape: tuple[int, ...]) -> None:
        original = make_continuous_time_request(
            rng=3141,
            batch_shape=batch_shape,
            min_steps=4,
            max_steps=4,
            context_shape=(3,),
            output_shape=(3,),
            input_missingness=True,
            target_missingness=True,
        ).to_joint_time()

        actual = original.to_triplets().to_joint_time()

        assert actual == original

    @pytest.mark.parametrize("batch_shape", BATCH_SHAPES.values())
    def test_query_and_context_indices_match_split_time(
        self,
        batch_shape: tuple[int, ...],
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

        joint = original.to_joint_time()
        assert joint.target_values is not None

        query_indices = joint.query_indices
        context_indices = joint.context_indices

        assert_close(
            original.query_times,
            joint.timestamps[query_indices],
            atol=0.0,
            rtol=0.0,
            equal_nan=True,
        )
        assert_close(
            original.query_mask,
            joint.query_mask[query_indices],
            atol=0.0,
            rtol=0.0,
            equal_nan=True,
        )
        assert_close(
            original.target_values,
            joint.target_values[query_indices],
            atol=0.0,
            rtol=0.0,
            equal_nan=True,
        )
        assert_close(
            original.context_times,
            joint.timestamps[context_indices],
            atol=0.0,
            rtol=0.0,
            equal_nan=True,
        )
        assert_close(
            original.context_mask,
            joint.context_mask[context_indices],
            atol=0.0,
            rtol=0.0,
            equal_nan=True,
        )
        assert_close(
            original.context_values,
            joint.context_values[context_indices],
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
        assert not replace(
            arg,
            context_channels=torch.tensor([0, 0, 0]),
        ).is_simple()
        assert not TripletTimeData(
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

    @pytest.mark.parametrize(
        "case",
        [key for key in TEST_DATA if key[0] == "simple"],
    )
    def test_query_indices_recover_simple_split_target_layout(
        self,
        case: tuple[DataType, BatchType, bool],
    ) -> None:
        split = TEST_DATA[case]["split"]
        triplet = split.to_triplets()
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

    def test_to_dense_unbatched(self) -> None:
        original = TripletTimeData(
            context_times=torch.tensor([1.0, 1.0, 2.0, 2.0]),
            context_channels=torch.tensor([0, 2, 1, 2]),
            context_values=torch.tensor([10.0, 11.0, 20.0, 21.0]),
            query_times=torch.tensor([3.0, 4.0, 4.0]),
            query_channels=torch.tensor([0, 0, 1]),
            target_values=torch.tensor([30.0, 40.0, 41.0]),
            static_covariates=torch.tensor([5.0, 6.0]),
        )

        actual = original.to_split_time()
        expected = SplitTimeData(
            context_times=torch.tensor([1.0, 2.0]),
            context_values=torch.tensor([
                [10.0, nan, 11.0],
                [nan, 20.0, 21.0],
            ]),
            context_mask=torch.tensor([
                [ True, False,  True],
                [False,  True,  True],
            ]),
            query_times=torch.tensor([3.0, 4.0]),
            query_mask=torch.tensor([[True, False], [True, True]]),
            target_values=torch.tensor([[30.0, nan], [40.0, 41.0]]),
            static_covariates=torch.tensor([5.0, 6.0]),
        )  # fmt: skip

        assert actual == expected

    def test_to_dense_unbatched_with_dims(self) -> None:
        original = TripletTimeData(
            context_times=torch.tensor([1.0, 2.0]),
            context_channels=torch.tensor([0, 1]),
            context_values=torch.tensor([10.0, 20.0]),
            query_times=torch.tensor([3.0]),
            query_channels=torch.tensor([1]),
            target_values=torch.tensor([30.0]),
        )

        actual = original.to_split_time(context_dim=3, query_dim=4)
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

    def test_batched_roundtrip(self) -> None:
        args = [
            TripletTimeData(
                context_times=torch.tensor([1.0, 1.0, 2.0]),
                context_channels=torch.tensor([0, 2, 1]),
                context_values=torch.tensor([10.0, 11.0, 20.0]),
                query_times=torch.tensor([3.0, 4.0]),
                query_channels=torch.tensor([0, 1]),
                target_values=torch.tensor([30.0, 40.0]),
                static_covariates=torch.tensor([1.0, 2.0]),
            ),
            TripletTimeData(
                context_times=torch.tensor([5.0]),
                context_channels=torch.tensor([1]),
                context_values=torch.tensor([50.0]),
                query_times=torch.tensor([6.0, 6.0, 7.0]),
                query_channels=torch.tensor([0, 2, 1]),
                target_values=torch.tensor([60.0, 62.0, 71.0]),
                static_covariates=torch.tensor([3.0, 4.0]),
            ),
        ]

        actual = TripletTimeData.from_unbatched(args)
        expected = TripletTimeData(
            context_times=torch.tensor([
                [1.0, 1.0, 2.0],
                [5.0, nan, nan],
            ]),
            context_channels=torch.tensor([
                [0,  2,  1],
                [1, -1, -1],
            ]),
            context_values=torch.tensor([
                [10.0, 11.0, 20.0],
                [50.0,  nan,  nan],
            ]),
            query_times=torch.tensor([
                [3.0, 4.0, nan],
                [6.0, 6.0, 7.0],
            ]),
            query_channels=torch.tensor([
                [0,  1, -1],
                [0,  2,  1],
            ]),
            target_values=torch.tensor([
                [30.0, 40.0,  nan],
                [60.0, 62.0, 71.0],
            ]),
            static_covariates=torch.tensor([
                [1.0, 2.0],
                [3.0, 4.0],
            ]),
        )  # fmt: skip

        assert actual == expected

        unbatched = actual.unbatch()
        assert isinstance(unbatched, list)
        assert len(unbatched) == len(args)
        for actual, expected in zip(unbatched, args, strict=True):
            assert actual == expected

    def test_batched_roundtrip_without_values(self) -> None:
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

        actual = TripletTimeData.from_unbatched(args)
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

    def test_to_dense_batched(self) -> None:
        original = TripletTimeData(
            context_times=torch.tensor([
                [1.0, 1.0, 2.0, nan],
                [0.0, 1.0, 1.0, 2.0],
            ]),
            context_channels=torch.tensor([
                [0, 1,  0, -1],
                [1, 0,  1,  0],
            ]),
            context_values=torch.tensor([
                [10.0, 11.0, 20.0,  nan],
                [ 1.0,  2.0,  3.0,  4.0],
            ]),
            query_times=torch.tensor([
                [3.0, 4.0, nan],
                [5.0, 5.0, 6.0],
            ]),
            query_channels=torch.tensor([
                [0, 1, -1],
                [1, 0,  1],
            ]),
            target_values=torch.tensor([
                [30.0, 40.0,  nan],
                [51.0, 50.0, 61.0],
            ]),
            static_covariates=torch.tensor([
                [7.0, 8.0],
                [9.0, 10.0],
            ]),
        )  # fmt: skip

        actual = original.to_split_time(context_dim=2, query_dim=2)
        expected = SplitTimeData(
            context_times=torch.tensor([
                [1.0, 2.0, nan],
                [0.0, 1.0, 2.0],
            ]),
            context_values=torch.tensor([
                [[10.0, 11.0], [20.0,  nan], [ nan,  nan]],
                [[ nan,  1.0], [ 2.0,  3.0], [ 4.0,  nan]],
            ]),
            context_mask=torch.tensor([
                [[ True,  True], [ True, False], [False, False]],
                [[False,  True], [ True,  True], [ True, False]],
            ]),
            query_times=torch.tensor([
                [3.0, 4.0],
                [5.0, 6.0],
            ]),
            query_mask=torch.tensor([
                [[ True, False], [False,  True]],
                [[ True,  True], [False,  True]],
            ]),
            target_values=torch.tensor([
                [[30.0,  nan], [ nan, 40.0]],
                [[50.0, 51.0], [ nan, 61.0]],
            ]),
            static_covariates=torch.tensor([
                [7.0, 8.0],
                [9.0, 10.0],
            ]),
        )  # fmt: skip

        assert actual == expected

    def test_to_dense_batched_without_values(self) -> None:
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

        actual = original.to_split_time(context_dim=2, query_dim=2)
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

    @pytest.mark.parametrize("batch_shape", BATCH_SHAPES.values())
    def test_to_dense_roundtrip(
        self,
        batch_shape: tuple[int, ...],
    ) -> None:
        original = _make_random_batched_triplet(batch_shape)

        actual = original.to_split_time(context_dim=3, query_dim=4).to_triplets()

        assert actual == original

    @pytest.mark.parametrize("batch_shape", BATCH_SHAPES.values())
    def test_to_combined_roundtrip(self, batch_shape: tuple[int, ...]) -> None:
        original = _make_random_batched_triplet(batch_shape)

        actual = original.to_joint_time().to_triplets()

        assert actual == original

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
        expected = req.to_triplets()

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
        expected = req.to_triplets()
        expected.validate()

        assert actual == expected


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
