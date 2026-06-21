r"""Tests for forecasting utility containers."""

import pytest
import torch
from torch import Tensor, nan
from torch.testing import assert_close

from linodenet.forecasting.utils import (
    BatchedCombinedArgs,
    BatchedDenseArgs,
    BatchedTripletArgs,
    CombinedArg,
    DenseArg,
    TripletArg,
)


def _assert_query_mask_equal(
    actual: Tensor,
    expected: Tensor,
    /,
    *,
    query_times: Tensor,
) -> None:
    assert torch.equal(actual.any(dim=-1), query_times.isfinite())
    assert torch.equal(actual, expected)


def _assert_dense_equal(actual: DenseArg, expected: DenseArg, /) -> None:
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
    assert torch.equal(actual.context_mask, expected.context_mask)
    assert_close(
        actual.query_times, expected.query_times, atol=0.0, rtol=0.0, equal_nan=True
    )

    _assert_query_mask_equal(
        actual.query_mask,
        expected.query_mask,
        query_times=actual.query_times,
    )

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
        actual.context_times, expected.context_times, atol=0.0, rtol=0.0, equal_nan=True
    )
    assert_close(
        actual.context_values,
        expected.context_values,
        atol=0.0,
        rtol=0.0,
        equal_nan=True,
    )
    assert torch.equal(actual.context_mask, expected.context_mask)
    assert_close(
        actual.query_times, expected.query_times, atol=0.0, rtol=0.0, equal_nan=True
    )

    _assert_query_mask_equal(
        actual.query_mask,
        expected.query_mask,
        query_times=actual.query_times,
    )

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


def _assert_combined_equal(actual: CombinedArg, expected: CombinedArg, /) -> None:
    assert_close(actual.times, expected.times, atol=0.0, rtol=0.0, equal_nan=True)
    assert_close(
        actual.context_values,
        expected.context_values,
        atol=0.0,
        rtol=0.0,
        equal_nan=True,
    )
    assert torch.equal(actual.context_mask, expected.context_mask)
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
    assert torch.equal(actual.query_mask, expected.query_mask)

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


def _assert_batched_combined_equal(
    actual: BatchedCombinedArgs,
    expected: BatchedCombinedArgs,
    /,
) -> None:
    assert_close(actual.times, expected.times, atol=0.0, rtol=0.0, equal_nan=True)
    assert_close(
        actual.context_values,
        expected.context_values,
        atol=0.0,
        rtol=0.0,
        equal_nan=True,
    )
    assert torch.equal(actual.context_mask, expected.context_mask)
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
    assert torch.equal(actual.query_mask, expected.query_mask)

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


def _combined_arg(
    *,
    times: Tensor,
    values: Tensor,
    context_mask: Tensor,
    query_mask: Tensor,
    query_values_available: bool = True,
    static_covariates: Tensor | None = None,
) -> CombinedArg:
    return CombinedArg(
        times=times,
        context_values=values.masked_fill(~context_mask, nan),
        context_mask=context_mask,
        query_mask=query_mask,
        query_values=(
            values.masked_fill(~query_mask, nan) if query_values_available else None
        ),
        static_covariates=static_covariates,
    )


def _batched_combined_args(
    *,
    times: Tensor,
    values: Tensor,
    context_mask: Tensor,
    query_mask: Tensor,
    query_values_available: bool = True,
    static_covariates: Tensor | None = None,
) -> BatchedCombinedArgs:
    return BatchedCombinedArgs(
        times=times,
        context_values=values.masked_fill(~context_mask, nan),
        context_mask=context_mask,
        query_mask=query_mask,
        query_values=(
            values.masked_fill(~query_mask, nan) if query_values_available else None
        ),
        static_covariates=static_covariates,
    )


def _make_random_batched_triplet(
    batch_shape: tuple[int, ...],
    /,
) -> BatchedTripletArgs:
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
    query_values = torch.full((num_samples, num_query), nan)

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
                query_values[sample, index] = float(
                    1000 + sample * 100 + step * 10 + channel
                )
                index += 1

    static_covariates = torch.randn((num_samples, 2), generator=generator)
    return BatchedTripletArgs(
        context_times=context_times.reshape(*batch_shape, num_context),
        context_channels=context_channels.reshape(*batch_shape, num_context),
        context_values=context_values.reshape(*batch_shape, num_context),
        query_times=query_times.reshape(*batch_shape, num_query),
        query_channels=query_channels.reshape(*batch_shape, num_query),
        query_values=query_values.reshape(*batch_shape, num_query),
        static_covariates=static_covariates.reshape(*batch_shape, 2),
    )


def _make_random_batched_dense(
    batch_shape: tuple[int, ...],
    /,
    *,
    seed: int = 3141,
    num_context_steps: int = 4,
    num_query_steps: int = 3,
    context_dim: int = 3,
    query_dim: int = 4,
) -> BatchedDenseArgs:
    num_samples = 1
    for size in batch_shape:
        num_samples *= size

    generator = torch.Generator().manual_seed(
        seed + sum((k + 1) * size for k, size in enumerate(batch_shape))
    )
    active_context_dim = max(context_dim - 1, 1)
    active_query_dim = max(query_dim - 1, 1)

    context_times = torch.full((num_samples, num_context_steps), nan)
    context_values = torch.full((num_samples, num_context_steps, context_dim), nan)
    context_mask = torch.zeros(
        (num_samples, num_context_steps, context_dim),
        dtype=torch.bool,
    )
    query_times = torch.full((num_samples, num_query_steps), nan)
    query_mask = torch.zeros(
        (num_samples, num_query_steps, query_dim),
        dtype=torch.bool,
    )
    query_values = torch.full((num_samples, num_query_steps, query_dim), nan)

    for sample in range(num_samples):
        context_steps = (
            num_context_steps
            if sample == 0
            else int(torch.randint(1, num_context_steps + 1, (), generator=generator))
        )
        context_times[sample, :context_steps] = torch.arange(
            1,
            context_steps + 1,
            dtype=context_times.dtype,
        )
        for step in range(context_steps):
            if sample == 0:
                mask = torch.zeros(context_dim, dtype=torch.bool)
                mask[:active_context_dim] = True
            else:
                mask = torch.zeros(context_dim, dtype=torch.bool)
                mask[:active_context_dim] = (
                    torch.rand(active_context_dim, generator=generator) < 0.7
                )
                if not mask.any():
                    mask[
                        int(torch.randint(active_context_dim, (), generator=generator))
                    ] = True

            for channel in mask.nonzero(as_tuple=True)[0]:
                context_mask[sample, step, channel] = True
                context_values[sample, step, channel] = float(
                    sample * 100 + step * 10 + channel
                )

        query_steps = (
            num_query_steps
            if sample == 0
            else int(torch.randint(1, num_query_steps + 1, (), generator=generator))
        )
        query_times[sample, :query_steps] = torch.arange(
            10,
            10 + query_steps,
            dtype=query_times.dtype,
        )
        for step in range(query_steps):
            if sample == 0:
                mask = torch.zeros(query_dim, dtype=torch.bool)
                mask[:active_query_dim] = True
            else:
                mask = torch.zeros(query_dim, dtype=torch.bool)
                mask[:active_query_dim] = (
                    torch.rand(active_query_dim, generator=generator) < 0.7
                )
                if not mask.any():
                    mask[
                        int(torch.randint(active_query_dim, (), generator=generator))
                    ] = True

            query_mask[sample, step, mask] = True
            for channel in mask.nonzero(as_tuple=True)[0]:
                query_values[sample, step, channel] = float(
                    1000 + sample * 100 + step * 10 + channel
                )

    static_covariates = torch.randn((num_samples, 2), generator=generator)
    return BatchedDenseArgs(
        context_times=context_times.reshape(*batch_shape, num_context_steps),
        context_values=context_values.reshape(
            *batch_shape,
            num_context_steps,
            context_dim,
        ),
        context_mask=context_mask.reshape(
            *batch_shape,
            num_context_steps,
            context_dim,
        ),
        query_times=query_times.reshape(*batch_shape, num_query_steps),
        query_mask=query_mask.reshape(*batch_shape, num_query_steps, query_dim),
        query_values=query_values.reshape(*batch_shape, num_query_steps, query_dim),
        static_covariates=static_covariates.reshape(*batch_shape, 2),
    )


class TestDense:
    def test_query_mask_clears_masked_values(self) -> None:
        arg = DenseArg(
            context_times=torch.tensor([1.0]),
            context_values=torch.tensor([[10.0, 11.0]]),
            context_mask=torch.tensor([[True, True]]),
            query_times=torch.tensor([2.0]),
            query_mask=torch.tensor([[True, False]]),
            query_values=torch.tensor([[20.0, 21.0]]),
        )
        assert_close(arg.query_values, torch.tensor([[20.0, nan]]), equal_nan=True)

        batched = BatchedDenseArgs(
            context_times=torch.tensor([[1.0]]),
            context_values=torch.tensor([[[10.0, 11.0]]]),
            context_mask=torch.tensor([[[True, True]]]),
            query_times=torch.tensor([[2.0, nan]]),
            query_mask=torch.tensor([[[True, False], [False, False]]]),
            query_values=torch.tensor([[[20.0, 21.0], [nan, nan]]]),
        )
        assert_close(
            batched.query_values,
            torch.tensor([[[20.0, nan], [nan, nan]]]),
            equal_nan=True,
        )

    def test_context_mask_clears_masked_values(self) -> None:
        arg = DenseArg(
            context_times=torch.tensor([1.0]),
            context_values=torch.tensor([[10.0, 11.0]]),
            context_mask=torch.tensor([[True, False]]),
            query_times=torch.tensor([2.0]),
            query_mask=torch.tensor([[True, False]]),
        )

        assert_close(arg.context_values, torch.tensor([[10.0, nan]]), equal_nan=True)

    def test_to_triplet_unbatched(self) -> None:
        original = DenseArg(
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
            context_mask=torch.tensor([
                    [ True, False],
                    [False,  True],
                    [ True,  True],
            ]),
            query_times=torch.tensor([4.0, 5.0]),
            query_mask=torch.tensor([[True, True], [True, True]]),
            query_values=torch.tensor([[40.0, 41.0], [50.0, 51.0]]),
            static_covariates=torch.tensor([7.0]),
        )  # fmt: skip

        triplet = original.to_triplet()
        actual = triplet.to_dense(
            context_dim=original.context_values.shape[-1],
            query_dim=2,
        )

        _assert_dense_equal(actual, original)

        triplet_result = actual.to_triplet()
        _assert_triplet_equal(triplet_result, triplet)

    def test_to_combined_unbatched(self) -> None:
        original = DenseArg(
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
            query_values=torch.tensor([
                [20.0,  nan],
                [40.0, 41.0],
            ]),
            static_covariates=torch.tensor([5.0, 6.0]),
        )  # fmt: skip

        actual = original.to_combined()
        expected = _combined_arg(
            times=torch.tensor([1.0, 2.0, 3.0, 4.0]),
            values=torch.tensor([
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
            static_covariates=torch.tensor([5.0, 6.0]),
        )  # fmt: skip

        _assert_combined_equal(actual, expected)

    def test_to_combined_roundtrip_unbatched(self) -> None:
        original = DenseArg(
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
            query_values=torch.tensor([
                [20.0,  nan],
                [40.0, 41.0],
            ]),
            static_covariates=torch.tensor([5.0, 6.0]),
        )  # fmt: skip

        actual = original.to_combined().to_dense()

        _assert_dense_equal(actual, original)

    def test_batched_roundtrip(self) -> None:
        args = [
            DenseArg(
                context_times=torch.tensor([1.0, 2.0]),
                context_values=torch.tensor([[1.0, nan], [2.0, 3.0]]),
                context_mask=torch.tensor([[True, False], [True, True]]),
                query_times=torch.tensor([5.0]),
                query_mask=torch.tensor([[True, False]]),
                query_values=torch.tensor([[9.0, nan]]),
                static_covariates=torch.tensor([1.0, 2.0]),
            ),
            DenseArg(
                context_times=torch.tensor([3.0]),
                context_values=torch.tensor([[4.0, 5.0]]),
                context_mask=torch.tensor([[True, True]]),
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
            context_mask=context_values.isfinite(),
            query_times=query_times,
            query_mask=query_times.isfinite()
            .unsqueeze(-1)
            .expand(*query_times.shape, 2),
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

    def test_to_triplet_roundtrip_batched_duplicates(self) -> None:
        original = BatchedDenseArgs(
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
            query_values=torch.tensor([
                [[50.0,  nan,  nan],
                 [ nan, 61.0,  nan],
                 [70.0,  nan, 72.0]],
                [[80.0,  nan,  nan],
                 [ nan, 91.0, 92.0],
                 [ nan,  nan,  nan]],
            ]),
        )  # fmt: skip

        triplet = original.to_triplet()
        actual = triplet.to_dense(context_dim=3, query_dim=3)
        expected = BatchedDenseArgs(
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
            query_values=torch.tensor([
                [[50.0,  nan,  nan],
                 [ nan, 61.0,  nan],
                 [70.0,  nan, 72.0]],
                [[80.0,  nan,  nan],
                 [ nan, 91.0, 92.0],
                 [ nan,  nan,  nan]],
            ]),
        )  # fmt: skip

        _assert_batched_dense_equal(actual, expected)
        _assert_batched_triplet_equal(actual.to_triplet(), triplet)

    @pytest.mark.parametrize("batch_shape", [(), (3,), (2, 3), (1, 2, 3)])
    def test_to_triplet_roundtrip_batched_random(
        self,
        batch_shape: tuple[int, ...],
    ) -> None:
        original = _make_random_batched_dense(batch_shape, query_dim=3)

        actual = original.to_triplet().to_dense(
            context_dim=original.context_values.shape[-1],
            query_dim=3,
        )

        _assert_batched_dense_equal(actual, original)

    def test_to_combined_roundtrip_batched(self) -> None:
        original = BatchedDenseArgs(
            context_times=torch.tensor([
                [1.0, 3.0],
                [0.0, 6.0],
            ]),
            context_values=torch.tensor([
                [[10.0,  nan, 12.0],
                 [ nan, 30.0, 32.0]],
                [[ nan,  1.0,  2.0],
                 [60.0,  nan, 62.0]],
            ]),
            context_mask=torch.tensor([
                [[ True, False,  True],
                 [False,  True,  True]],
                [[False,  True,  True],
                 [ True, False,  True]],
            ]),
            query_times=torch.tensor([
                [2.0, 4.0],
                [5.0, 7.0],
            ]),
            query_mask=torch.tensor([
                [[ True, False,  True],
                 [ True,  True,  True]],
                [[False,  True,  True],
                 [ True, False,  True]],
            ]),
            query_values=torch.tensor([
                [[20.0,  nan, 22.0],
                 [40.0, 41.0, 42.0]],
                [[ nan, 51.0, 52.0],
                 [70.0,  nan, 72.0]],
            ]),
            static_covariates=torch.tensor([
                [5.0, 6.0],
                [7.0, 8.0],
            ]),
        )  # fmt: skip

        actual = original.to_combined().to_dense()

        _assert_batched_dense_equal(actual, original)


class TestCombined:
    def test_rejects_non_increasing_query_times(self) -> None:
        with pytest.raises(AssertionError):
            _combined_arg(
                times=torch.tensor([1.0, 1.0]),
                values=torch.tensor([
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
            )  # fmt: skip

        with pytest.raises(AssertionError):
            _batched_combined_args(
                times=torch.tensor([[1.0, 1.0]]),
                values=torch.tensor([[
                    [1.0, nan],
                    [2.0, 3.0],
                ]]),
                context_mask=torch.tensor([[
                    [ True, False],
                    [False, False],
                ]]),
                query_mask=torch.tensor([[
                    [False,  True],
                    [ True, False],
                ]]),
            )  # fmt: skip

    def test_rejects_mixed_query_value_availability(self) -> None:
        with pytest.raises(AssertionError):
            _combined_arg(
                times=torch.tensor([1.0, 2.0]),
                values=torch.tensor([
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
            )  # fmt: skip

        with pytest.raises(AssertionError):
            _batched_combined_args(
                times=torch.tensor([[1.0, 2.0]]),
                values=torch.tensor([[
                    [1.0, nan],
                    [2.0, 3.0],
                ]]),
                context_mask=torch.tensor([[
                    [ True, False],
                    [False, False],
                ]]),
                query_mask=torch.tensor([[
                    [False,  True],
                    [ True, False],
                ]]),
            )  # fmt: skip

    def test_batched_roundtrip(self) -> None:
        args = [
            _combined_arg(
                times=torch.tensor([1.0, 2.0, 3.0, 4.0]),
                values=torch.tensor([
                    [10.0,  nan, 12.0],
                    [20.0,  nan, 22.0],
                    [ nan, 30.0, 32.0],
                    [40.0, 41.0, 42.0],
                ]),
                context_mask=torch.tensor([
                    [ True, False,  True],
                    [False, False, False],
                    [False,  True,  True],
                    [False, False, False],
                ]),
                query_mask=torch.tensor([
                    [False, False, False],
                    [ True, False,  True],
                    [False, False, False],
                    [ True,  True,  True],
                ]),
                static_covariates=torch.tensor([5.0, 6.0]),
            ),
            _combined_arg(
                times=torch.tensor([0.0, 5.0]),
                values=torch.tensor([
                    [ nan,  1.0,  2.0],
                    [ nan, 51.0, 52.0],
                ]),
                context_mask=torch.tensor([
                    [False,  True,  True],
                    [False, False, False],
                ]),
                query_mask=torch.tensor([
                    [False, False, False],
                    [False,  True,  True],
                ]),
                static_covariates=torch.tensor([7.0, 8.0]),
            ),
        ]  # fmt: skip

        actual = BatchedCombinedArgs.from_unbatched(args)
        expected = _batched_combined_args(
            times=torch.tensor([
                [1.0, 2.0, 3.0, 4.0],
                [0.0, 5.0, nan, nan],
            ]),
            values=torch.tensor([
                [[10.0,  nan, 12.0],
                 [20.0,  nan, 22.0],
                 [ nan, 30.0, 32.0],
                 [40.0, 41.0, 42.0]],
                [[ nan,  1.0,  2.0],
                 [ nan, 51.0, 52.0],
                 [ nan,  nan,  nan],
                 [ nan,  nan,  nan]],
            ]),
            context_mask=torch.tensor([
                [[ True, False,  True],
                 [False, False, False],
                 [False,  True,  True],
                 [False, False, False]],
                [[False,  True,  True],
                 [False, False, False],
                 [False, False, False],
                 [False, False, False]],
            ]),
            query_mask=torch.tensor([
                [[False, False, False],
                 [ True, False,  True],
                 [False, False, False],
                 [ True,  True,  True]],
                [[False, False, False],
                 [False,  True,  True],
                 [False, False, False],
                 [False, False, False]],
            ]),
            static_covariates=torch.tensor([
                [5.0, 6.0],
                [7.0, 8.0],
            ]),
        )  # fmt: skip

        _assert_batched_combined_equal(actual, expected)

        unbatched = actual.unbatch()
        assert isinstance(unbatched, list)
        assert len(unbatched) == len(args)
        for actual, expected in zip(unbatched, args, strict=True):
            _assert_combined_equal(actual, expected)

    def test_to_dense_unbatched(self) -> None:
        original = _combined_arg(
            times=torch.tensor([1.0, 2.0, 3.0, 4.0]),
            values=torch.tensor([
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
            static_covariates=torch.tensor([5.0, 6.0]),
        )  # fmt: skip

        actual = original.to_dense()
        expected = DenseArg(
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
            query_values=torch.tensor([
                [20.0,  nan],
                [40.0, 41.0],
            ]),
            static_covariates=torch.tensor([5.0, 6.0]),
        )  # fmt: skip

        _assert_dense_equal(actual, expected)

    def test_to_dense_unbatched_without_query_values(self) -> None:
        original = _combined_arg(
            times=torch.tensor([1.0, 2.0, 3.0, 4.0]),
            values=torch.tensor([
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
            query_values_available=False,
            static_covariates=torch.tensor([5.0, 6.0]),
        )  # fmt: skip

        actual = original.to_dense()
        expected = DenseArg(
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
            static_covariates=torch.tensor([5.0, 6.0]),
        )  # fmt: skip

        _assert_dense_equal(actual, expected)

    def test_to_dense_batched_without_query_values(self) -> None:
        original = _batched_combined_args(
            times=torch.tensor([
                [1.0, 2.0, 3.0, 4.0],
                [0.0, 5.0, nan, nan],
            ]),
            values=torch.tensor([
                [[10.0,  nan, 12.0],
                 [20.0,  nan, 22.0],
                 [ nan, 30.0, 32.0],
                 [40.0, 41.0, 42.0]],
                [[ nan,  1.0,  2.0],
                 [ nan, 51.0, 52.0],
                 [ nan,  nan,  nan],
                 [ nan,  nan,  nan]],
            ]),
            context_mask=torch.tensor([
                [[ True, False,  True],
                 [False, False, False],
                 [False,  True,  True],
                 [False, False, False]],
                [[False,  True,  True],
                 [False, False, False],
                 [False, False, False],
                 [False, False, False]],
            ]),
            query_mask=torch.tensor([
                [[False, False, False],
                 [ True, False,  True],
                 [False, False, False],
                 [ True,  True,  True]],
                [[False, False, False],
                 [False,  True,  True],
                 [False, False, False],
                 [False, False, False]],
            ]),
            query_values_available=False,
            static_covariates=torch.tensor([
                [5.0, 6.0],
                [7.0, 8.0],
            ]),
        )  # fmt: skip

        actual = original.to_dense()
        expected = BatchedDenseArgs(
            context_times=torch.tensor([
                [1.0, 3.0],
                [0.0, nan],
            ]),
            context_values=torch.tensor([
                [[10.0,  nan, 12.0],
                 [ nan, 30.0, 32.0]],
                [[ nan,  1.0,  2.0],
                 [ nan,  nan,  nan]],
            ]),
            context_mask=torch.tensor([
                [[ True, False,  True],
                 [False,  True,  True]],
                [[False,  True,  True],
                 [False, False, False]],
            ]),
            query_times=torch.tensor([
                [2.0, 4.0],
                [5.0, nan],
            ]),
            query_mask=torch.tensor([
                [[ True, False,  True],
                 [ True,  True,  True]],
                [[False,  True,  True],
                 [False, False, False]],
            ]),
            static_covariates=torch.tensor([
                [5.0, 6.0],
                [7.0, 8.0],
            ]),
        )  # fmt: skip

        _assert_batched_dense_equal(actual, expected)

    def test_to_dense_roundtrip_unbatched(self) -> None:
        original = _combined_arg(
            times=torch.tensor([1.0, 2.0, 3.0, 4.0]),
            values=torch.tensor([
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
            static_covariates=torch.tensor([5.0, 6.0]),
        )  # fmt: skip

        actual = original.to_dense().to_combined()

        _assert_combined_equal(actual, original)

    def test_to_triplet_roundtrip_unbatched(self) -> None:
        original = _combined_arg(
            times=torch.tensor([1.0, 2.0, 3.0, 4.0]),
            values=torch.tensor([
                [10.0,  nan, 12.0],
                [20.0,  nan, 22.0],
                [ nan, 30.0, 32.0],
                [40.0, 41.0, 42.0],
            ]),
            context_mask=torch.tensor([
                [ True, False,  True],
                [False, False, False],
                [False,  True,  True],
                [False, False, False],
            ]),
            query_mask=torch.tensor([
                [False, False, False],
                [ True, False,  True],
                [False, False, False],
                [ True,  True,  True],
            ]),
            static_covariates=torch.tensor([5.0, 6.0]),
        )  # fmt: skip

        actual = original.to_triplet().to_combined()

        _assert_combined_equal(actual, original)

    def test_to_dense_roundtrip_batched(self) -> None:
        original = _batched_combined_args(
            times=torch.tensor([
                [1.0, 2.0, 3.0, 4.0],
                [0.0, 5.0, nan, nan],
            ]),
            values=torch.tensor([
                [[10.0,  nan, 12.0],
                 [20.0,  nan, 22.0],
                 [ nan, 30.0, 32.0],
                 [40.0, 41.0, 42.0]],
                [[ nan,  1.0,  2.0],
                 [ nan, 51.0, 52.0],
                 [ nan,  nan,  nan],
                 [ nan,  nan,  nan]],
            ]),
            context_mask=torch.tensor([
                [[ True, False,  True],
                 [False, False, False],
                 [False,  True,  True],
                 [False, False, False]],
                [[False,  True,  True],
                 [False, False, False],
                 [False, False, False],
                 [False, False, False]],
            ]),
            query_mask=torch.tensor([
                [[False, False, False],
                 [ True, False,  True],
                 [False, False, False],
                 [ True,  True,  True]],
                [[False, False, False],
                 [False,  True,  True],
                 [False, False, False],
                 [False, False, False]],
            ]),
            static_covariates=torch.tensor([
                [5.0, 6.0],
                [7.0, 8.0],
            ]),
        )  # fmt: skip

        actual = original.to_dense().to_combined()

        _assert_batched_combined_equal(actual, original)

    def test_to_triplet_roundtrip_batched(self) -> None:
        original = _batched_combined_args(
            times=torch.tensor([
                [1.0, 2.0, 3.0, 4.0],
                [0.0, 5.0, nan, nan],
            ]),
            values=torch.tensor([
                [[10.0,  nan, 12.0],
                 [20.0,  nan, 22.0],
                 [ nan, 30.0, 32.0],
                 [40.0, 41.0, 42.0]],
                [[ nan,  1.0,  2.0],
                 [ nan, 51.0, 52.0],
                 [ nan,  nan,  nan],
                 [ nan,  nan,  nan]],
            ]),
            context_mask=torch.tensor([
                [[ True, False,  True],
                 [False, False, False],
                 [False,  True,  True],
                 [False, False, False]],
                [[False,  True,  True],
                 [False, False, False],
                 [False, False, False],
                 [False, False, False]],
            ]),
            query_mask=torch.tensor([
                [[False, False, False],
                 [ True, False,  True],
                 [False, False, False],
                 [ True,  True,  True]],
                [[False, False, False],
                 [False,  True,  True],
                 [False, False, False],
                 [False, False, False]],
            ]),
            static_covariates=torch.tensor([
                [5.0, 6.0],
                [7.0, 8.0],
            ]),
        )  # fmt: skip

        actual = original.to_triplet().to_combined()

        _assert_batched_combined_equal(actual, original)


class TestTriplet:
    def test_rejects_duplicate_queries(self) -> None:
        with pytest.raises(AssertionError):
            TripletArg(
                context_times=torch.tensor([1.0]),
                context_channels=torch.tensor([0]),
                context_values=torch.tensor([10.0]),
                query_times=torch.tensor([2.0, 2.0]),
                query_channels=torch.tensor([1, 1]),
            )

        with pytest.raises(AssertionError):
            BatchedTripletArgs(
                context_times=torch.tensor([[1.0]]),
                context_channels=torch.tensor([[0]]),
                context_values=torch.tensor([[10.0]]),
                query_times=torch.tensor([[2.0, 2.0]]),
                query_channels=torch.tensor([[1, 1]]),
            )

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
            context_mask=torch.tensor([
                [ True, False,  True],
                [False,  True,  True],
            ]),
            query_times=torch.tensor([3.0, 4.0]),
            query_mask=torch.tensor([[True, False], [True, True]]),
            query_values=torch.tensor([[30.0, nan], [40.0, 41.0]]),
            static_covariates=torch.tensor([5.0, 6.0]),
        )  # fmt: skip

        _assert_dense_equal(actual, expected)

    def test_to_dense_unbatched_with_dims(self) -> None:
        original = TripletArg(
            context_times=torch.tensor([1.0, 2.0]),
            context_channels=torch.tensor([0, 1]),
            context_values=torch.tensor([10.0, 20.0]),
            query_times=torch.tensor([3.0]),
            query_channels=torch.tensor([1]),
            query_values=torch.tensor([30.0]),
        )

        actual = original.to_dense(context_dim=3, query_dim=4)
        expected = DenseArg(
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
            query_values=torch.tensor([[nan, 30.0, nan, nan]]),
        )  # fmt: skip

        _assert_dense_equal(actual, expected)

    def test_to_combined_roundtrip_unbatched(self) -> None:
        original = TripletArg(
            context_times=torch.tensor([1.0, 1.0, 3.0, 3.0]),
            context_channels=torch.tensor([0, 2, 1, 2]),
            context_values=torch.tensor([10.0, 12.0, 30.0, 32.0]),
            query_times=torch.tensor([2.0, 2.0, 4.0, 4.0, 4.0]),
            query_channels=torch.tensor([0, 2, 0, 1, 2]),
            query_values=torch.tensor([20.0, 22.0, 40.0, 41.0, 42.0]),
            static_covariates=torch.tensor([5.0, 6.0]),
        )

        actual = original.to_combined().to_triplet()

        _assert_triplet_equal(actual, original)

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

    def test_to_dense_batched(self) -> None:
        original = BatchedTripletArgs(
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
            query_values=torch.tensor([
                [30.0, 40.0,  nan],
                [51.0, 50.0, 61.0],
            ]),
            static_covariates=torch.tensor([
                [7.0, 8.0],
                [9.0, 10.0],
            ]),
        )  # fmt: skip

        actual = original.to_dense(context_dim=2, query_dim=2)
        expected = BatchedDenseArgs(
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
            query_values=torch.tensor([
                [[30.0,  nan], [ nan, 40.0]],
                [[50.0, 51.0], [ nan, 61.0]],
            ]),
            static_covariates=torch.tensor([
                [7.0, 8.0],
                [9.0, 10.0],
            ]),
        )  # fmt: skip

        _assert_batched_dense_equal(actual, expected)

    def test_to_dense_batched_without_values(self) -> None:
        original = BatchedTripletArgs(
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

        actual = original.to_dense(context_dim=2, query_dim=2)
        expected = BatchedDenseArgs(
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

        _assert_batched_dense_equal(actual, expected)

    def test_to_combined_roundtrip_batched(self) -> None:
        original = BatchedTripletArgs(
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
                [ 1.0,  2.0,  nan,  nan],
            ]),
            query_times=torch.tensor([
                [2.0, 2.0, 4.0, 4.0, 4.0],
                [5.0, 5.0, nan, nan, nan],
            ]),
            query_channels=torch.tensor([
                [0, 2, 0, 1, 2],
                [1, 2, -1, -1, -1],
            ]),
            query_values=torch.tensor([
                [20.0, 22.0, 40.0, 41.0, 42.0],
                [51.0, 52.0,  nan,  nan,  nan],
            ]),
            static_covariates=torch.tensor([
                [5.0, 6.0],
                [7.0, 8.0],
            ]),
        )  # fmt: skip

        actual = original.to_combined().to_triplet()

        _assert_batched_triplet_equal(actual, original)

    @pytest.mark.parametrize("batch_shape", [(), (3,), (2, 3), (1, 2, 3)])
    def test_to_dense_roundtrip_batched_random(
        self,
        batch_shape: tuple[int, ...],
    ) -> None:
        original = _make_random_batched_triplet(batch_shape)

        actual = original.to_dense(context_dim=3, query_dim=4).to_triplet()

        _assert_batched_triplet_equal(actual, original)
