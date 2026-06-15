r"""Tests for forecasting utility containers."""

import torch
from torch import nan
from torch.testing import assert_close

from linodenet.forecasting.utils import (
    BatchedDenseArgs,
    BatchedTripletArgs,
    DenseArg,
    TripletArg,
)


def test_dense_arg_to_triplet() -> None:
    original = DenseArg(
        context_times=torch.tensor([1.0, 2.0]),
        context_values=torch.tensor(
            [
                [10.0, nan, 11.0],
                [nan, 20.0, 21.0],
            ]
        ),
        query_times=torch.tensor([3.0, 4.0]),
        query_mask=torch.tensor([[True, False], [True, True]]),
        query_values=torch.tensor([[30.0, nan], [40.0, 41.0]]),
        static_covariates=torch.tensor([5.0, 6.0]),
    )

    actual = original.to_triplet()
    expected = TripletArg(
        context_times=torch.tensor([1.0, 1.0, 2.0, 2.0]),
        context_channels=torch.tensor([0, 2, 1, 2]),
        context_values=torch.tensor([10.0, 11.0, 20.0, 21.0]),
        query_times=torch.tensor([3.0, 4.0, 4.0]),
        query_channels=torch.tensor([0, 0, 1]),
        query_values=torch.tensor([30.0, 40.0, 41.0]),
        static_covariates=torch.tensor([5.0, 6.0]),
    )

    assert torch.equal(actual.context_times, expected.context_times)
    assert torch.equal(actual.context_channels, expected.context_channels)
    assert torch.equal(actual.context_values, expected.context_values)
    assert torch.equal(actual.query_times, expected.query_times)
    assert torch.equal(actual.query_channels, expected.query_channels)
    assert torch.equal(actual.query_values, expected.query_values)
    assert torch.equal(actual.static_covariates, expected.static_covariates)


def test_triplet_arg_to_dense() -> None:
    original = TripletArg(
        context_times=torch.tensor([1.0, 1.0, 2.0, 2.0]),
        context_channels=torch.tensor([0, 2, 1, 2]),
        context_values=torch.tensor([10.0, 11.0, 20.0, 21.0]),
        query_times=torch.tensor([3.0, 4.0, 4.0]),
        query_channels=torch.tensor([0, 0, 1]),
        query_values=torch.tensor([30.0, 40.0, 41.0]),
        static_covariates=torch.tensor([5.0, 6.0]),
    )

    result = original.to_dense()
    expected = DenseArg(
        context_times=torch.tensor([1.0, 2.0]),
        context_values=torch.tensor(
            [
                [10.0, nan, 11.0],
                [nan, 20.0, 21.0],
            ]
        ),
        query_times=torch.tensor([3.0, 4.0]),
        query_mask=torch.tensor([[True, False], [True, True]]),
        query_values=torch.tensor([[30.0, nan], [40.0, 41.0]]),
        static_covariates=torch.tensor([5.0, 6.0]),
    )

    assert_close(result.context_times, expected.context_times, atol=0.0, rtol=0.0)
    assert_close(
        result.context_values,
        expected.context_values,
        atol=0.0,
        rtol=0.0,
        equal_nan=True,
    )
    assert_close(result.query_times, expected.query_times, atol=0.0, rtol=0.0)
    assert torch.equal(result.query_mask, expected.query_mask)
    assert_close(
        result.query_values, expected.query_values, atol=0.0, rtol=0.0, equal_nan=True
    )
    assert_close(
        result.static_covariates, expected.static_covariates, atol=0.0, rtol=0.0
    )


def test_dense_triplet_dense_roundtrip() -> None:
    original = DenseArg(
        context_times=torch.tensor([1.0, 2.0, 3.0]),
        context_values=torch.tensor(
            [
                [10.0, nan],
                [nan, 20.0],
                [30.0, 31.0],
            ]
        ),
        query_times=torch.tensor([4.0, 5.0]),
        query_values=torch.tensor([[40.0, 41.0], [50.0, 51.0]]),
        static_covariates=torch.tensor([7.0]),
    )

    triplet = original.to_triplet()
    actual = triplet.to_dense()

    assert_close(actual.context_times, original.context_times, atol=0.0, rtol=0.0)
    assert_close(
        actual.context_values,
        original.context_values,
        atol=0.0,
        rtol=0.0,
        equal_nan=True,
    )
    assert_close(actual.query_times, original.query_times, atol=0.0, rtol=0.0)
    assert actual.query_mask is None
    assert_close(actual.query_values, original.query_values, atol=0.0, rtol=0.0)
    assert_close(
        actual.static_covariates, original.static_covariates, atol=0.0, rtol=0.0
    )

    triplet_result = actual.to_triplet()
    assert_close(
        triplet_result.context_times, triplet.context_times, atol=0.0, rtol=0.0
    )
    assert torch.equal(triplet_result.context_channels, triplet.context_channels)
    assert_close(
        triplet_result.context_values, triplet.context_values, atol=0.0, rtol=0.0
    )
    assert_close(triplet_result.query_times, triplet.query_times, atol=0.0, rtol=0.0)
    assert torch.equal(triplet_result.query_channels, triplet.query_channels)
    assert_close(triplet_result.query_values, triplet.query_values, atol=0.0, rtol=0.0)


def test_batched_dense_args_from_unbatched_and_unbatch_roundtrip() -> None:
    args = [
        DenseArg(
            context_times=torch.tensor([1.0, 2.0]),
            context_values=torch.tensor([[1.0, nan], [2.0, 3.0]]),
            query_times=torch.tensor([5.0]),
            query_mask=torch.tensor([[True, False]]),
            query_values=torch.tensor([[9.0, nan]]),
            static_covariates=torch.tensor([1.0, 2.0]),
        ),
        DenseArg(
            context_times=torch.tensor([3.0]),
            context_values=torch.tensor([[4.0, 5.0]]),
            query_times=torch.tensor([6.0, 7.0, 8.0]),
            query_mask=torch.tensor([[False, True], [True, False], [True, True]]),
            query_values=torch.tensor([[nan, 6.0], [7.0, nan], [8.0, 9.0]]),
            static_covariates=torch.tensor([3.0, 4.0]),
        ),
    ]

    actual = BatchedDenseArgs.from_unbatched(args)
    expected = BatchedDenseArgs(
        context_times=torch.tensor([[1.0, 2.0], [3.0, nan]]),
        context_values=torch.tensor(
            [
                [[1.0, nan], [2.0, 3.0]],
                [[4.0, 5.0], [nan, nan]],
            ]
        ),
        query_times=torch.tensor([[5.0, nan, nan], [6.0, 7.0, 8.0]]),
        query_mask=torch.tensor(
            [
                [[True, False], [False, False], [False, False]],
                [[False, True], [True, False], [True, True]],
            ]
        ),
        query_values=torch.tensor(
            [
                [[9.0, nan], [nan, nan], [nan, nan]],
                [[nan, 6.0], [7.0, nan], [8.0, 9.0]],
            ]
        ),
        static_covariates=torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
    )

    assert_close(
        actual.context_times, expected.context_times, atol=0.0, rtol=0.0, equal_nan=True
    )
    assert_close(
        actual.context_values,
        expected.context_values,
        atol=0.0,
        rtol=0.0,
        equal_nan=True,
    )
    assert_close(
        actual.query_times, expected.query_times, atol=0.0, rtol=0.0, equal_nan=True
    )
    assert torch.equal(actual.query_mask, expected.query_mask)
    assert_close(
        actual.query_values, expected.query_values, atol=0.0, rtol=0.0, equal_nan=True
    )
    assert_close(
        actual.static_covariates, expected.static_covariates, atol=0.0, rtol=0.0
    )

    unbatched = actual.unbatch()

    assert isinstance(unbatched, list)
    assert len(unbatched) == len(args)
    for actual, expected in zip(unbatched, args, strict=True):
        assert_close(actual.context_times, expected.context_times, atol=0.0, rtol=0.0)
        assert_close(
            actual.context_values,
            expected.context_values,
            atol=0.0,
            rtol=0.0,
            equal_nan=True,
        )
        assert_close(actual.query_times, expected.query_times, atol=0.0, rtol=0.0)
        assert torch.equal(actual.query_mask, expected.query_mask)
        assert_close(
            actual.query_values,
            expected.query_values,
            atol=0.0,
            rtol=0.0,
            equal_nan=True,
        )
        assert_close(
            actual.static_covariates, expected.static_covariates, atol=0.0, rtol=0.0
        )


def test_batched_dense_args_to_triplet_with_query_mask() -> None:
    context_times = torch.tensor(
        [
            [1.0, 2.0, nan],
            [0.0, 1.0, 2.0],
        ]
    )
    context_values = torch.tensor(
        [
            [
                [10.0, nan],
                [nan, 20.0],
                [nan, nan],
            ],
            [
                [nan, 1.0],
                [2.0, 3.0],
                [4.0, nan],
            ],
        ]
    )
    query_times = torch.tensor(
        [
            [3.0, nan],
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
            [[30.0, nan], [nan, nan]],
            [[nan, 40.0], [50.0, 60.0]],
        ]
    )

    actual = BatchedDenseArgs(
        context_times=context_times,
        context_values=context_values,
        query_times=query_times,
        query_mask=query_mask,
        query_values=query_values,
    ).to_triplet()

    expected = BatchedTripletArgs(
        context_times=torch.tensor([
            [1.0, 2.0, nan, nan],
            [0.0, 1.0, 1.0      , 2.0],
        ]),
        context_channels=torch.tensor([[0, 1, -1, -1], [1, 0, 1, 0]]),
        context_values=torch.tensor([
            [10.0, 20.0, nan, nan],
            [ 1.0,  2.0, 3.0      , 4.0],
        ]),
        query_times=torch.tensor(
            [
                [3.0, nan, nan],
                [4.0, 5.0, 5.0],
            ]
        ),
        query_channels=torch.tensor([[0, -1, -1], [1, 0, 1]]),
        query_values=torch.tensor(
            [
                [30.0, nan, nan],
                [40.0, 50.0, 60.0],
            ]
        ),
    )  # fmt: skip

    assert_close(
        actual.context_times, expected.context_times, atol=0.0, rtol=0.0, equal_nan=True
    )
    assert torch.equal(actual.context_channels, expected.context_channels)
    assert_close(
        actual.context_values,
        expected.context_values,
        atol=0.0,
        rtol=0.0,
        equal_nan=True,
    )
    assert_close(
        actual.query_times, expected.query_times, atol=0.0, rtol=0.0, equal_nan=True
    )
    assert torch.equal(actual.query_channels, expected.query_channels)
    assert_close(
        actual.query_values, expected.query_values, atol=0.0, rtol=0.0, equal_nan=True
    )


def test_batched_dense_args_to_triplet_without_query_mask() -> None:
    context_times = torch.tensor([[[1.0], [2.0]]])
    context_values = torch.tensor([[[[1.0, nan]], [[2.0, 3.0]]]])
    query_times = torch.tensor([[[4.0, nan], [5.0, 6.0]]])

    actual = BatchedDenseArgs(
        context_times=context_times,
        context_values=context_values,
        query_times=query_times,
    ).to_triplet()

    expected = BatchedTripletArgs(
        context_times=torch.tensor([[[1.0, nan], [2.0, 2.0]]]),
        context_channels=torch.tensor([[[0, -1], [0, 1]]]),
        context_values=torch.tensor([[[1.0, nan], [2.0, 3.0]]]),
        query_times=torch.tensor([[[4.0, 4.0, nan, nan], [5.0, 5.0, 6.0, 6.0]]]),
        query_channels=torch.tensor([[[0, 1, -1, -1], [0, 1, 0, 1]]]),
        query_values=torch.full((1, 2, 4), nan),
    )

    assert_close(
        actual.context_times, expected.context_times, atol=0.0, rtol=0.0, equal_nan=True
    )
    assert torch.equal(actual.context_channels, expected.context_channels)
    assert_close(
        actual.context_values,
        expected.context_values,
        atol=0.0,
        rtol=0.0,
        equal_nan=True,
    )
    assert_close(
        actual.query_times, expected.query_times, atol=0.0, rtol=0.0, equal_nan=True
    )
    assert torch.equal(actual.query_channels, expected.query_channels)
    assert_close(
        actual.query_values, expected.query_values, atol=0.0, rtol=0.0, equal_nan=True
    )
