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


def _assert_dense_equal(actual: DenseArg, expected: DenseArg, /) -> None:
    assert_close(
        actual.context_times,
        expected.context_times,
        atol=0.0,
        rtol=0.0,
        equal_nan=True,
    )
    assert_close(
        actual.context_values,
        expected.context_values,
        atol=0.0,
        rtol=0.0,
        equal_nan=True,
    )
    assert_close(
        actual.query_times,
        expected.query_times,
        atol=0.0,
        rtol=0.0,
        equal_nan=True,
    )

    if actual.query_mask is None or expected.query_mask is None:
        assert actual.query_mask is expected.query_mask
    else:
        assert torch.equal(actual.query_mask, expected.query_mask)

    if actual.query_values is None or expected.query_values is None:
        assert actual.query_values is expected.query_values
    else:
        assert_close(
            actual.query_values,
            expected.query_values,
            atol=0.0,
            rtol=0.0,
            equal_nan=True,
        )

    if actual.static_covariates is None or expected.static_covariates is None:
        assert actual.static_covariates is expected.static_covariates
    else:
        assert_close(
            actual.static_covariates,
            expected.static_covariates,
            atol=0.0,
            rtol=0.0,
            equal_nan=True,
        )


def _assert_batched_dense_equal(
    actual: BatchedDenseArgs,
    expected: BatchedDenseArgs,
    /,
) -> None:
    assert_close(
        actual.context_times,
        expected.context_times,
        atol=0.0,
        rtol=0.0,
        equal_nan=True,
    )
    assert_close(
        actual.context_values,
        expected.context_values,
        atol=0.0,
        rtol=0.0,
        equal_nan=True,
    )
    assert_close(
        actual.query_times,
        expected.query_times,
        atol=0.0,
        rtol=0.0,
        equal_nan=True,
    )

    if actual.query_mask is None or expected.query_mask is None:
        assert actual.query_mask is expected.query_mask
    else:
        assert torch.equal(actual.query_mask, expected.query_mask)

    if actual.query_values is None or expected.query_values is None:
        assert actual.query_values is expected.query_values
    else:
        assert_close(
            actual.query_values,
            expected.query_values,
            atol=0.0,
            rtol=0.0,
            equal_nan=True,
        )

    if actual.static_covariates is None or expected.static_covariates is None:
        assert actual.static_covariates is expected.static_covariates
    else:
        assert_close(
            actual.static_covariates,
            expected.static_covariates,
            atol=0.0,
            rtol=0.0,
            equal_nan=True,
        )


def _assert_triplet_equal(actual: TripletArg, expected: TripletArg, /) -> None:
    assert_close(
        actual.context_times,
        expected.context_times,
        atol=0.0,
        rtol=0.0,
        equal_nan=True,
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
        actual.query_times,
        expected.query_times,
        atol=0.0,
        rtol=0.0,
        equal_nan=True,
    )
    assert actual.query_channels is not None
    assert expected.query_channels is not None
    assert torch.equal(actual.query_channels, expected.query_channels)

    if actual.query_values is None or expected.query_values is None:
        assert actual.query_values is expected.query_values
    else:
        assert_close(
            actual.query_values,
            expected.query_values,
            atol=0.0,
            rtol=0.0,
            equal_nan=True,
        )

    if actual.static_covariates is None or expected.static_covariates is None:
        assert actual.static_covariates is expected.static_covariates
    else:
        assert_close(
            actual.static_covariates,
            expected.static_covariates,
            atol=0.0,
            rtol=0.0,
            equal_nan=True,
        )


def _assert_batched_triplet_equal(
    actual: BatchedTripletArgs,
    expected: BatchedTripletArgs,
    /,
) -> None:
    assert_close(
        actual.context_times,
        expected.context_times,
        atol=0.0,
        rtol=0.0,
        equal_nan=True,
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
        actual.query_times,
        expected.query_times,
        atol=0.0,
        rtol=0.0,
        equal_nan=True,
    )
    assert torch.equal(actual.query_channels, expected.query_channels)

    if actual.query_values is None or expected.query_values is None:
        assert actual.query_values is expected.query_values
    else:
        assert_close(
            actual.query_values,
            expected.query_values,
            atol=0.0,
            rtol=0.0,
            equal_nan=True,
        )

    if actual.static_covariates is None or expected.static_covariates is None:
        assert actual.static_covariates is expected.static_covariates
    else:
        assert_close(
            actual.static_covariates,
            expected.static_covariates,
            atol=0.0,
            rtol=0.0,
            equal_nan=True,
        )


class TestDense:
    def test_to_triplet_unbatched(self) -> None:
        original = DenseArg(
            context_times=torch.tensor([1.0, 2.0]),
            context_values=torch.tensor([
                [10.0, nan, 11.0],
                [nan, 20.0, 21.0],
            ]),
            query_times=torch.tensor([3.0, 4.0]),
            query_mask=torch.tensor([[True, False], [True, True]]),
            query_values=torch.tensor([[30.0, nan], [40.0, 41.0]]),
            static_covariates=torch.tensor([5.0, 6.0]),
        )  # fmt: skip

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

        _assert_triplet_equal(actual, expected)

    def test_to_triplet_roundtrip_unbatched(self) -> None:
        original = DenseArg(
            context_times=torch.tensor([1.0, 2.0, 3.0]),
            context_values=torch.tensor([
                    [10.0, nan],
                    [nan, 20.0],
                    [30.0, 31.0],
            ]),
            query_times=torch.tensor([4.0, 5.0]),
            query_values=torch.tensor([[40.0, 41.0], [50.0, 51.0]]),
            static_covariates=torch.tensor([7.0]),
        )  # fmt: skip

        triplet = original.to_triplet()
        actual = triplet.to_dense()

        _assert_dense_equal(actual, original)

        triplet_result = actual.to_triplet()
        _assert_triplet_equal(triplet_result, triplet)

    def test_batched_roundtrip(self) -> None:
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
            context_times=torch.tensor([
                [1.0, 2.0],
                [3.0, nan],
            ]),
            context_values=torch.tensor([
                [[1.0, nan], [2.0, 3.0]],
                [[4.0, 5.0], [nan, nan]],
            ]),
            query_times=torch.tensor([
                [5.0, nan, nan],
                [6.0, 7.0, 8.0],
            ]),
            query_mask=torch.tensor([
                [[ True, False], [False, False], [False, False]],
                [[False,  True], [ True, False], [ True,  True]],
            ]),
            query_values=torch.tensor([
                [[9.0, nan], [nan, nan], [nan, nan]],
                [[nan, 6.0], [7.0, nan], [8.0, 9.0]],
            ]),
            static_covariates=torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        )  # fmt: skip

        _assert_batched_dense_equal(actual, expected)

        unbatched = actual.unbatch()

        assert isinstance(unbatched, list)
        assert len(unbatched) == len(args)
        for actual, expected in zip(unbatched, args, strict=True):
            _assert_dense_equal(actual, expected)

    def test_to_triplet_batched_with_mask(self) -> None:
        original = BatchedDenseArgs(
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
            query_times=torch.tensor([
                [3.0, nan],
                [4.0, 5.0],
            ]),
            query_mask=torch.tensor([
                [[ True, False], [False, False]],
                [[False,  True], [ True,  True]],
            ]),
            query_values=torch.tensor([
                [[30.0,  nan], [ nan,  nan]],
                [[ nan, 40.0], [50.0, 60.0]],
            ]),
        )  # fmt: skip
        actual = original.to_triplet()

        expected = BatchedTripletArgs(
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
            query_values=torch.tensor([
                [30.0,  nan,  nan],
                [40.0, 50.0, 60.0],
            ]),
        )  # fmt: skip

        _assert_batched_triplet_equal(actual, expected)

    def test_to_triplet_batched_without_mask(self) -> None:
        context_times = torch.tensor([[[1.0], [2.0]]])
        context_values = torch.tensor([[[[1.0, nan]], [[2.0, 3.0]]]])
        query_times = torch.tensor([[[4.0, nan], [5.0, 6.0]]])

        original = BatchedDenseArgs(
            context_times=context_times,
            context_values=context_values,
            query_times=query_times,
        )
        actual = original.to_triplet()

        expected = BatchedTripletArgs(
            context_times=torch.tensor([[[1.0, nan], [2.0, 2.0]]]),
            context_channels=torch.tensor([[[0, -1], [0, 1]]]),
            context_values=torch.tensor([[[1.0, nan], [2.0, 3.0]]]),
            query_times=torch.tensor([[[4.0, 4.0, nan, nan], [5.0, 5.0, 6.0, 6.0]]]),
            query_channels=torch.tensor([[[0, 1, -1, -1], [0, 1, 0, 1]]]),
        )

        _assert_batched_triplet_equal(actual, expected)


class TestTriplet:
    def test_to_dense_unbatched(self) -> None:
        original = TripletArg(
            context_times=torch.tensor([1.0, 1.0, 2.0, 2.0]),
            context_channels=torch.tensor([0, 2, 1, 2]),
            context_values=torch.tensor([10.0, 11.0, 20.0, 21.0]),
            query_times=torch.tensor([3.0, 4.0, 4.0]),
            query_channels=torch.tensor([0, 0, 1]),
            query_values=torch.tensor([30.0, 40.0, 41.0]),
            static_covariates=torch.tensor([5.0, 6.0]),
        )

        actual = original.to_dense()
        expected = DenseArg(
            context_times=torch.tensor([1.0, 2.0]),
            context_values=torch.tensor([
                [10.0, nan, 11.0],
                [nan, 20.0, 21.0],
            ]),
            query_times=torch.tensor([3.0, 4.0]),
            query_mask=torch.tensor([[True, False], [True, True]]),
            query_values=torch.tensor([[30.0, nan], [40.0, 41.0]]),
            static_covariates=torch.tensor([5.0, 6.0]),
        )  # fmt: skip

        _assert_dense_equal(actual, expected)

    def test_batched_roundtrip(self) -> None:
        args = [
            TripletArg(
                context_times=torch.tensor([1.0, 1.0, 2.0]),
                context_channels=torch.tensor([0, 2, 1]),
                context_values=torch.tensor([10.0, 11.0, 20.0]),
                query_times=torch.tensor([3.0, 4.0]),
                query_channels=torch.tensor([0, 1]),
                query_values=torch.tensor([30.0, 40.0]),
                static_covariates=torch.tensor([1.0, 2.0]),
            ),
            TripletArg(
                context_times=torch.tensor([5.0]),
                context_channels=torch.tensor([1]),
                context_values=torch.tensor([50.0]),
                query_times=torch.tensor([6.0, 6.0, 7.0]),
                query_channels=torch.tensor([0, 2, 1]),
                query_values=torch.tensor([60.0, 62.0, 71.0]),
                static_covariates=torch.tensor([3.0, 4.0]),
            ),
        ]

        actual = BatchedTripletArgs.from_unbatched(args)
        expected = BatchedTripletArgs(
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
            query_values=torch.tensor([
                [30.0, 40.0,  nan],
                [60.0, 62.0, 71.0],
            ]),
            static_covariates=torch.tensor([
                [1.0, 2.0],
                [3.0, 4.0],
            ]),
        )  # fmt: skip

        _assert_batched_triplet_equal(actual, expected)

        unbatched = actual.unbatch()
        assert isinstance(unbatched, list)
        assert len(unbatched) == len(args)
        for actual, expected in zip(unbatched, args, strict=True):
            _assert_triplet_equal(actual, expected)

    def test_batched_roundtrip_without_values(self) -> None:
        args = [
            TripletArg(
                context_times=torch.tensor([1.0]),
                context_channels=torch.tensor([0]),
                context_values=torch.tensor([10.0]),
                query_times=torch.tensor([3.0, 4.0]),
                query_channels=torch.tensor([0, 1]),
            ),
            TripletArg(
                context_times=torch.tensor([5.0, 5.0]),
                context_channels=torch.tensor([0, 1]),
                context_values=torch.tensor([50.0, 51.0]),
                query_times=torch.tensor([6.0]),
                query_channels=torch.tensor([2]),
            ),
        ]

        actual = BatchedTripletArgs.from_unbatched(args)
        expected = BatchedTripletArgs(
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

        _assert_batched_triplet_equal(actual, expected)

        unbatched = actual.unbatch()
        assert isinstance(unbatched, list)
        assert len(unbatched) == len(args)
        for actual, expected in zip(unbatched, args, strict=True):
            _assert_triplet_equal(actual, expected)
