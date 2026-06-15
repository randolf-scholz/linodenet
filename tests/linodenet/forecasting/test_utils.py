r"""Tests for forecasting utility containers."""

import torch
from torch.testing import assert_close

from linodenet.forecasting.utils import BatchedDenseArgs, DenseArg


def test_batched_dense_args_from_unbatched_and_unbatch_roundtrip() -> None:
    args = [
        DenseArg(
            context_times=torch.tensor([1.0, 2.0]),
            context_values=torch.tensor([[1.0, torch.nan], [2.0, 3.0]]),
            query_times=torch.tensor([5.0]),
            query_mask=torch.tensor([[True, False]]),
            query_values=torch.tensor([[9.0, torch.nan]]),
            static_covariates=torch.tensor([1.0, 2.0]),
        ),
        DenseArg(
            context_times=torch.tensor([3.0]),
            context_values=torch.tensor([[4.0, 5.0]]),
            query_times=torch.tensor([6.0, 7.0, 8.0]),
            query_mask=torch.tensor([[False, True], [True, False], [True, True]]),
            query_values=torch.tensor([[torch.nan, 6.0], [7.0, torch.nan], [8.0, 9.0]]),
            static_covariates=torch.tensor([3.0, 4.0]),
        ),
    ]

    batched = BatchedDenseArgs.from_unbatched(args)

    assert_close(
        batched.context_times,
        torch.tensor([[1.0, 2.0], [3.0, torch.nan]]),
        equal_nan=True,
    )
    assert_close(
        batched.query_times,
        torch.tensor([[5.0, torch.nan, torch.nan], [6.0, 7.0, 8.0]]),
        equal_nan=True,
    )
    assert_close(
        batched.query_mask,
        torch.tensor(
            [
                [[True, False], [False, False], [False, False]],
                [[False, True], [True, False], [True, True]],
            ]
        ),
    )
    assert_close(
        batched.query_values,
        torch.tensor(
            [
                [[9.0, torch.nan], [torch.nan, torch.nan], [torch.nan, torch.nan]],
                [[torch.nan, 6.0], [7.0, torch.nan], [8.0, 9.0]],
            ]
        ),
        equal_nan=True,
    )
    assert_close(batched.static_covariates, torch.tensor([[1.0, 2.0], [3.0, 4.0]]))

    unbatched = batched.unbatch()

    assert isinstance(unbatched, list)
    assert len(unbatched) == len(args)
    for actual, expected in zip(unbatched, args, strict=True):
        assert_close(actual.context_times, expected.context_times)
        assert_close(actual.context_values, expected.context_values, equal_nan=True)
        assert_close(actual.query_times, expected.query_times)
        assert_close(actual.query_mask, expected.query_mask)
        assert_close(actual.query_values, expected.query_values, equal_nan=True)
        assert_close(actual.static_covariates, expected.static_covariates)


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
    query_values = torch.tensor(
        [
            [[30.0, torch.nan], [torch.nan, torch.nan]],
            [[torch.nan, 40.0], [50.0, 60.0]],
        ]
    )

    result = BatchedDenseArgs(
        context_times=context_times,
        context_values=context_values,
        query_times=query_times,
        query_mask=query_mask,
        query_values=query_values,
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
    assert_close(
        result.query_values,
        torch.tensor(
            [
                [30.0, torch.nan, torch.nan],
                [40.0, 50.0, 60.0],
            ]
        ),
        equal_nan=True,
    )


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
