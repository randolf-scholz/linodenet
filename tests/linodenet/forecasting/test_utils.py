r"""Tests for forecasting utility containers."""

import torch
from torch.testing import assert_close

from linodenet.forecasting.utils import BatchedDenseArgs


def test_batched_dense_args_to_triplet_with_query_mask() -> None:
    context_times = torch.tensor(
        [
            [1.0, 2.0, torch.nan],
            [0.0, 1.0, 2.0],
        ]
    )
    context_values = torch.tensor(
        [
            [
                [10.0, torch.nan],
                [torch.nan, 20.0],
                [torch.nan, torch.nan],
            ],
            [
                [torch.nan, 1.0],
                [2.0, 3.0],
                [4.0, torch.nan],
            ],
        ]
    )
    query_times = torch.tensor(
        [
            [3.0, torch.nan],
            [4.0, 5.0],
        ]
    )
    query_mask = torch.tensor(
        [
            [[True, False], [False, False]],
            [[False, True], [True, True]],
        ]
    )

    result = BatchedDenseArgs(
        context_times=context_times,
        context_values=context_values,
        query_times=query_times,
        query_mask=query_mask,
    ).to_triplet()

    assert_close(
        result.context_times,
        torch.tensor(
            [
                [1.0, 2.0, torch.nan, torch.nan],
                [0.0, 1.0, 1.0, 2.0],
            ]
        ),
        equal_nan=True,
    )
    assert_close(
        result.context_channels,
        torch.tensor([[0, 1, -1, -1], [1, 0, 1, 0]]),
    )
    assert_close(
        result.context_values,
        torch.tensor(
            [
                [10.0, 20.0, torch.nan, torch.nan],
                [1.0, 2.0, 3.0, 4.0],
            ]
        ),
        equal_nan=True,
    )
    assert_close(
        result.query_times,
        torch.tensor(
            [
                [3.0, torch.nan, torch.nan],
                [4.0, 5.0, 5.0],
            ]
        ),
        equal_nan=True,
    )
    assert_close(result.query_channels, torch.tensor([[0, -1, -1], [1, 0, 1]]))
    assert result.query_values.isnan().all()


def test_batched_dense_args_to_triplet_without_query_mask() -> None:
    context_times = torch.tensor([[[1.0], [2.0]]])
    context_values = torch.tensor([[[[1.0, torch.nan]], [[2.0, 3.0]]]])
    query_times = torch.tensor([[[4.0, torch.nan], [5.0, 6.0]]])

    result = BatchedDenseArgs(
        context_times=context_times,
        context_values=context_values,
        query_times=query_times,
    ).to_triplet()

    assert_close(
        result.context_times,
        torch.tensor([[[1.0, torch.nan], [2.0, 2.0]]]),
        equal_nan=True,
    )
    assert_close(result.context_channels, torch.tensor([[[0, -1], [0, 1]]]))
    assert_close(
        result.context_values,
        torch.tensor([[[1.0, torch.nan], [2.0, 3.0]]]),
        equal_nan=True,
    )
    assert_close(
        result.query_times,
        torch.tensor([[[4.0, 4.0, torch.nan, torch.nan], [5.0, 5.0, 6.0, 6.0]]]),
        equal_nan=True,
    )
    assert_close(
        result.query_channels,
        torch.tensor([[[0, 1, -1, -1], [0, 1, 0, 1]]]),
    )
    assert result.query_values.isnan().all()
