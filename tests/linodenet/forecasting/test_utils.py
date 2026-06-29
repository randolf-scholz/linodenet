r"""Tests for forecasting utility containers."""

from typing import NamedTuple

import pytest
import torch
from torch import Tensor, nan
from torch.testing import assert_close

from linodenet.forecasting.utils import (
    EventBatch,
    JointTimeData,
    SplitTimeData,
    TripletBatch,
    TripletTimeData,
)

from .base import make_forecasting_request


def _assert_query_mask_equal(
    actual: Tensor,
    expected: Tensor,
    /,
    *,
    query_times: Tensor,
) -> None:
    assert torch.equal(actual.any(dim=-1), query_times.isfinite())
    assert torch.equal(actual, expected)


def _assert_dense_equal(actual: SplitTimeData, expected: SplitTimeData, /) -> None:
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

    if actual.target_values is None or expected.target_values is None:
        assert actual.target_values is expected.target_values
    else:
        assert_close(
            actual.target_values,
            expected.target_values,
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
    actual: SplitTimeData,
    expected: SplitTimeData,
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

    if actual.target_values is None or expected.target_values is None:
        assert actual.target_values is expected.target_values
    else:
        assert_close(
            actual.target_values,
            expected.target_values,
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


def _assert_triplet_equal(
    actual: TripletTimeData, expected: TripletTimeData, /
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
    assert actual.query_channels is not None
    assert expected.query_channels is not None
    assert torch.equal(actual.query_channels, expected.query_channels)

    if actual.target_values is None or expected.target_values is None:
        assert actual.target_values is expected.target_values
    else:
        assert_close(
            actual.target_values,
            expected.target_values,
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


def _assert_combined_equal(actual: JointTimeData, expected: JointTimeData, /) -> None:
    assert_close(
        actual.timestamps, expected.timestamps, atol=0.0, rtol=0.0, equal_nan=True
    )
    assert_close(
        actual.context_values,
        expected.context_values,
        atol=0.0,
        rtol=0.0,
        equal_nan=True,
    )
    assert torch.equal(actual.context_mask, expected.context_mask)
    if actual.target_values is None or expected.target_values is None:
        assert actual.target_values is expected.target_values
    else:
        assert_close(
            actual.target_values,
            expected.target_values,
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
    actual: JointTimeData,
    expected: JointTimeData,
    /,
) -> None:
    assert_close(
        actual.timestamps, expected.timestamps, atol=0.0, rtol=0.0, equal_nan=True
    )
    assert_close(
        actual.context_values,
        expected.context_values,
        atol=0.0,
        rtol=0.0,
        equal_nan=True,
    )
    assert torch.equal(actual.context_mask, expected.context_mask)
    if actual.target_values is None or expected.target_values is None:
        assert actual.target_values is expected.target_values
    else:
        assert_close(
            actual.target_values,
            expected.target_values,
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
    actual: TripletTimeData,
    expected: TripletTimeData,
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

    if actual.target_values is None or expected.target_values is None:
        assert actual.target_values is expected.target_values
    else:
        assert_close(
            actual.target_values,
            expected.target_values,
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
    target_values_available: bool = True,
    static_covariates: Tensor | None = None,
) -> JointTimeData:
    return JointTimeData(
        timestamps=times,
        context_values=values.masked_fill(~context_mask, nan),
        context_mask=context_mask,
        query_mask=query_mask,
        target_values=(
            values.masked_fill(~query_mask, nan) if target_values_available else None
        ),
        static_covariates=static_covariates,
    )


def _batched_combined_args(
    *,
    times: Tensor,
    values: Tensor,
    context_mask: Tensor,
    query_mask: Tensor,
    target_values_available: bool = True,
    static_covariates: Tensor | None = None,
) -> JointTimeData:
    return JointTimeData(
        timestamps=times,
        context_values=values.masked_fill(~context_mask, nan),
        context_mask=context_mask,
        query_mask=query_mask,
        target_values=(
            values.masked_fill(~query_mask, nan) if target_values_available else None
        ),
        static_covariates=static_covariates,
    )


class _CanonicalTimeFormats(NamedTuple):
    split: SplitTimeData
    joint: JointTimeData
    triplet: TripletTimeData


class _CanonicalTimeData(NamedTuple):
    samples: tuple[_CanonicalTimeFormats, _CanonicalTimeFormats]
    batched: _CanonicalTimeFormats


def _canonical_time_data(
    *,
    target_values_available: bool = True,
) -> _CanonicalTimeData:
    first_target_values = (
        torch.tensor([[20.0, nan, 22.0], [40.0, 41.0, 42.0]])
        if target_values_available
        else None
    )
    second_target_values = (
        torch.tensor([[nan, 51.0, 52.0]]) if target_values_available else None
    )
    batched_target_values = (
        torch.tensor([
            [[20.0, nan, 22.0], [40.0, 41.0, 42.0]],
            [[nan, 51.0, 52.0], [nan, nan, nan]],
        ])
        if target_values_available
        else None
    )  # fmt: skip
    first = _CanonicalTimeFormats(
        split=SplitTimeData(
            context_times=torch.tensor([1.0, 3.0]),
            context_values=torch.tensor([[10.0, nan, 12.0], [nan, 30.0, 32.0]]),
            context_mask=torch.tensor([[True, False, True], [False, True, True]]),
            query_times=torch.tensor([2.0, 4.0]),
            query_mask=torch.tensor([[True, False, True], [True, True, True]]),
            target_values=first_target_values,
            static_covariates=torch.tensor([5.0, 6.0]),
        ),  # fmt: skip
        joint=_combined_arg(
            times=torch.tensor([1.0, 2.0, 3.0, 4.0]),
            values=torch.tensor(
                [
                    [10.0, nan, 12.0],
                    [20.0, nan, 22.0],
                    [nan, 30.0, 32.0],
                    [40.0, 41.0, 42.0],
                ]
            ),
            context_mask=torch.tensor(
                [
                    [True, False, True],
                    [False, False, False],
                    [False, True, True],
                    [False, False, False],
                ]
            ),
            query_mask=torch.tensor(
                [
                    [False, False, False],
                    [True, False, True],
                    [False, False, False],
                    [True, True, True],
                ]
            ),
            target_values_available=target_values_available,
            static_covariates=torch.tensor([5.0, 6.0]),
        ),  # fmt: skip
        triplet=TripletTimeData(
            context_times=torch.tensor([1.0, 1.0, 3.0, 3.0]),
            context_channels=torch.tensor([0, 2, 1, 2]),
            context_values=torch.tensor([10.0, 12.0, 30.0, 32.0]),
            query_times=torch.tensor([2.0, 2.0, 4.0, 4.0, 4.0]),
            query_channels=torch.tensor([0, 2, 0, 1, 2]),
            target_values=(
                torch.tensor([20.0, 22.0, 40.0, 41.0, 42.0])
                if target_values_available
                else None
            ),
            static_covariates=torch.tensor([5.0, 6.0]),
        ),
    )
    second = _CanonicalTimeFormats(
        split=SplitTimeData(
            context_times=torch.tensor([0.0]),
            context_values=torch.tensor([[nan, 1.0, 2.0]]),
            context_mask=torch.tensor([[False, True, True]]),
            query_times=torch.tensor([5.0]),
            query_mask=torch.tensor([[False, True, True]]),
            target_values=second_target_values,
            static_covariates=torch.tensor([7.0, 8.0]),
        ),
        joint=_combined_arg(
            times=torch.tensor([0.0, 5.0]),
            values=torch.tensor([[nan, 1.0, 2.0], [nan, 51.0, 52.0]]),
            context_mask=torch.tensor([[False, True, True], [False, False, False]]),
            query_mask=torch.tensor([[False, False, False], [False, True, True]]),
            target_values_available=target_values_available,
            static_covariates=torch.tensor([7.0, 8.0]),
        ),  # fmt: skip
        triplet=TripletTimeData(
            context_times=torch.tensor([0.0, 0.0]),
            context_channels=torch.tensor([1, 2]),
            context_values=torch.tensor([1.0, 2.0]),
            query_times=torch.tensor([5.0, 5.0]),
            query_channels=torch.tensor([1, 2]),
            target_values=(
                torch.tensor([51.0, 52.0]) if target_values_available else None
            ),
            static_covariates=torch.tensor([7.0, 8.0]),
        ),
    )
    batched = _CanonicalTimeFormats(
        split=SplitTimeData(
            context_times=torch.tensor([[1.0, 3.0], [0.0, nan]]),
            context_values=torch.tensor(
                [
                    [[10.0, nan, 12.0], [nan, 30.0, 32.0]],
                    [[nan, 1.0, 2.0], [nan, nan, nan]],
                ]
            ),
            context_mask=torch.tensor(
                [
                    [[True, False, True], [False, True, True]],
                    [[False, True, True], [False, False, False]],
                ]
            ),
            query_times=torch.tensor([[2.0, 4.0], [5.0, nan]]),
            query_mask=torch.tensor(
                [
                    [[True, False, True], [True, True, True]],
                    [[False, True, True], [False, False, False]],
                ]
            ),
            target_values=batched_target_values,
            static_covariates=torch.tensor([[5.0, 6.0], [7.0, 8.0]]),
        ),  # fmt: skip
        joint=_batched_combined_args(
            times=torch.tensor([[1.0, 2.0, 3.0, 4.0], [0.0, 5.0, nan, nan]]),
            values=torch.tensor(
                [
                    [
                        [10.0, nan, 12.0],
                        [20.0, nan, 22.0],
                        [nan, 30.0, 32.0],
                        [40.0, 41.0, 42.0],
                    ],
                    [
                        [nan, 1.0, 2.0],
                        [nan, 51.0, 52.0],
                        [nan, nan, nan],
                        [nan, nan, nan],
                    ],
                ]
            ),
            context_mask=torch.tensor(
                [
                    [
                        [True, False, True],
                        [False, False, False],
                        [False, True, True],
                        [False, False, False],
                    ],
                    [
                        [False, True, True],
                        [False, False, False],
                        [False, False, False],
                        [False, False, False],
                    ],
                ]
            ),
            query_mask=torch.tensor(
                [
                    [
                        [False, False, False],
                        [True, False, True],
                        [False, False, False],
                        [True, True, True],
                    ],
                    [
                        [False, False, False],
                        [False, True, True],
                        [False, False, False],
                        [False, False, False],
                    ],
                ]
            ),
            target_values_available=target_values_available,
            static_covariates=torch.tensor([[5.0, 6.0], [7.0, 8.0]]),
        ),  # fmt: skip
        triplet=TripletTimeData(
            context_times=torch.tensor([[1.0, 1.0, 3.0, 3.0], [0.0, 0.0, nan, nan]]),
            context_channels=torch.tensor([[0, 2, 1, 2], [1, 2, -1, -1]]),
            context_values=torch.tensor(
                [[10.0, 12.0, 30.0, 32.0], [1.0, 2.0, nan, nan]]
            ),
            query_times=torch.tensor(
                [[2.0, 2.0, 4.0, 4.0, 4.0], [5.0, 5.0, nan, nan, nan]]
            ),
            query_channels=torch.tensor([[0, 2, 0, 1, 2], [1, 2, -1, -1, -1]]),
            target_values=(
                torch.tensor(
                    [
                        [20.0, 22.0, 40.0, 41.0, 42.0],
                        [51.0, 52.0, nan, nan, nan],
                    ]
                )
                if target_values_available
                else None
            ),
            static_covariates=torch.tensor([[5.0, 6.0], [7.0, 8.0]]),
        ),  # fmt: skip
    )
    return _CanonicalTimeData(samples=(first, second), batched=batched)


def _make_random_batched_triplet(
    batch_shape: tuple[int, ...],
    /,
) -> TripletTimeData:
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


class TestSplitTimeData:
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

        _assert_triplet_equal(actual, expected)

    def test_to_triplet_roundtrip_unbatched(self) -> None:
        original = SplitTimeData(
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
            target_values=torch.tensor([[40.0, 41.0], [50.0, 51.0]]),
            static_covariates=torch.tensor([7.0]),
        )  # fmt: skip

        triplet = original.to_triplets()
        actual = triplet.to_split_time(
            context_dim=original.context_values.shape[-1],
            query_dim=2,
        )

        _assert_dense_equal(actual, original)

        triplet_result = actual.to_triplets()
        _assert_triplet_equal(triplet_result, triplet)

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

        actual = original.to_joint_time().to_split_time()

        _assert_dense_equal(actual, original)

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

        _assert_dense_equal(actual, original)

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

        _assert_batched_dense_equal(actual, expected)

        unbatched = actual.unbatch()

        assert isinstance(unbatched, list)
        assert len(unbatched) == len(args)
        for actual, expected in zip(unbatched, args, strict=True):
            _assert_dense_equal(actual, expected)

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

        _assert_batched_triplet_equal(actual, expected)

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

        expected = TripletTimeData(
            context_times=torch.tensor([[[1.0, nan], [2.0, 2.0]]]),
            context_channels=torch.tensor([[[0, -1], [0, 1]]]),
            context_values=torch.tensor([[[1.0, nan], [2.0, 3.0]]]),
            query_times=torch.tensor([[[4.0, 4.0, nan, nan], [5.0, 5.0, 6.0, 6.0]]]),
            query_channels=torch.tensor([[[0, 1, -1, -1], [0, 1, 0, 1]]]),
        )

        _assert_batched_triplet_equal(actual, expected)

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

        _assert_batched_dense_equal(actual, expected)
        _assert_batched_triplet_equal(actual.to_triplets(), triplet)

    @pytest.mark.parametrize("batch_shape", [(), (3,), (2, 3), (1, 2, 3)])
    def test_to_triplet_roundtrip_batched_random(
        self,
        batch_shape: tuple[int, ...],
    ) -> None:
        original = make_forecasting_request(
            seed=3141,
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

        _assert_batched_dense_equal(actual, original)

    def test_to_combined_roundtrip_batched(self) -> None:
        original = SplitTimeData(
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
            target_values=torch.tensor([
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

        actual = original.to_joint_time().to_split_time()

        _assert_batched_dense_equal(actual, original)

    @pytest.mark.parametrize("batch_shape", [(), (3,), (2, 3)])
    def test_to_combined_roundtrip_batched_distinct_dims(
        self,
        batch_shape: tuple[int, ...],
    ) -> None:
        original = make_forecasting_request(
            seed=3141,
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

        _assert_batched_dense_equal(actual, original)


class TestJointTimeData:
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
        canonical = _canonical_time_data()
        args = [sample.joint for sample in canonical.samples]

        actual = JointTimeData.from_unbatched(args)
        expected = canonical.batched.joint

        _assert_batched_combined_equal(actual, expected)

        unbatched = actual.unbatch()
        assert isinstance(unbatched, list)
        assert len(unbatched) == len(args)
        for actual, expected in zip(unbatched, args, strict=True):
            _assert_combined_equal(actual, expected)

    def test_to_dense_unbatched(self) -> None:
        canonical = _canonical_time_data()
        original = canonical.samples[0].joint

        actual = original.to_split_time()
        expected = canonical.samples[0].split

        _assert_dense_equal(actual, expected)

    def test_to_dense_unbatched_without_target_values(self) -> None:
        canonical = _canonical_time_data(target_values_available=False)
        original = canonical.samples[0].joint

        actual = original.to_split_time()
        expected = canonical.samples[0].split

        _assert_dense_equal(actual, expected)

    def test_to_dense_batched_without_target_values(self) -> None:
        canonical = _canonical_time_data(target_values_available=False)
        original = canonical.batched.joint

        actual = original.to_split_time()
        expected = canonical.batched.split

        _assert_batched_dense_equal(actual, expected)

    def test_to_dense_roundtrip_unbatched(self) -> None:
        original = _canonical_time_data().samples[0].joint

        actual = original.to_split_time().to_joint_time()

        _assert_combined_equal(actual, original)

    def test_to_triplet_roundtrip_unbatched(self) -> None:
        original = _canonical_time_data().samples[0].joint

        actual = original.to_triplets().to_joint_time()

        _assert_combined_equal(actual, original)

    def test_to_dense_roundtrip_batched(self) -> None:
        original = _canonical_time_data().batched.joint

        actual = original.to_split_time().to_joint_time()

        _assert_batched_combined_equal(actual, original)

    def test_to_triplet_roundtrip_batched(self) -> None:
        original = _canonical_time_data().batched.joint

        actual = original.to_triplets().to_joint_time()

        _assert_batched_combined_equal(actual, original)


class TestTripletTimeData:
    def test_rejects_duplicate_queries(self) -> None:
        with pytest.raises(AssertionError):
            TripletTimeData(
                context_times=torch.tensor([1.0]),
                context_channels=torch.tensor([0]),
                context_values=torch.tensor([10.0]),
                query_times=torch.tensor([2.0, 2.0]),
                query_channels=torch.tensor([1, 1]),
            )

        with pytest.raises(AssertionError):
            TripletTimeData(
                context_times=torch.tensor([[1.0]]),
                context_channels=torch.tensor([[0]]),
                context_values=torch.tensor([[10.0]]),
                query_times=torch.tensor([[2.0, 2.0]]),
                query_channels=torch.tensor([[1, 1]]),
            )

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

        _assert_dense_equal(actual, expected)

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

        _assert_dense_equal(actual, expected)

    def test_to_combined_roundtrip_unbatched(self) -> None:
        original = TripletTimeData(
            context_times=torch.tensor([1.0, 1.0, 3.0, 3.0]),
            context_channels=torch.tensor([0, 2, 1, 2]),
            context_values=torch.tensor([10.0, 12.0, 30.0, 32.0]),
            query_times=torch.tensor([2.0, 2.0, 4.0, 4.0, 4.0]),
            query_channels=torch.tensor([0, 2, 0, 1, 2]),
            target_values=torch.tensor([20.0, 22.0, 40.0, 41.0, 42.0]),
            static_covariates=torch.tensor([5.0, 6.0]),
        )

        actual = original.to_joint_time().to_triplets()

        _assert_triplet_equal(actual, original)

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

        _assert_batched_triplet_equal(actual, expected)

        unbatched = actual.unbatch()
        assert isinstance(unbatched, list)
        assert len(unbatched) == len(args)
        for actual, expected in zip(unbatched, args, strict=True):
            _assert_triplet_equal(actual, expected)

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

        _assert_batched_triplet_equal(actual, expected)

        unbatched = actual.unbatch()
        assert isinstance(unbatched, list)
        assert len(unbatched) == len(args)
        for actual, expected in zip(unbatched, args, strict=True):
            _assert_triplet_equal(actual, expected)

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

        _assert_batched_dense_equal(actual, expected)

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

        _assert_batched_dense_equal(actual, expected)

    def test_to_combined_roundtrip_batched(self) -> None:
        original = TripletTimeData(
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
            target_values=torch.tensor([
                [20.0, 22.0, 40.0, 41.0, 42.0],
                [51.0, 52.0,  nan,  nan,  nan],
            ]),
            static_covariates=torch.tensor([
                [5.0, 6.0],
                [7.0, 8.0],
            ]),
        )  # fmt: skip

        actual = original.to_joint_time().to_triplets()

        _assert_batched_triplet_equal(actual, original)

    @pytest.mark.parametrize("batch_shape", [(), (8,), (1, 2, 3)])
    def test_to_dense_roundtrip_batched_random(
        self,
        batch_shape: tuple[int, ...],
    ) -> None:
        original = _make_random_batched_triplet(batch_shape)

        actual = original.to_split_time(context_dim=3, query_dim=4).to_triplets()

        _assert_batched_triplet_equal(actual, original)


class TestEventBatch:
    def test_query_indices_unbatched(self) -> None:
        req = SplitTimeData(
            context_times=torch.tensor([1.0, 3.0]),
            context_values=torch.tensor([[10.0, nan], [nan, 30.0]]),
            context_mask=torch.tensor([[True, False], [False, True]]),
            query_times=torch.tensor([2.0, 4.0]),
            query_mask=torch.tensor([[True, False], [True, True]]),
            target_values=torch.tensor([[20.0, nan], [40.0, 41.0]]),
        )
        event = EventBatch.from_request(
            context_times=req.context_times,
            context_values=req.context_values,
            context_mask=req.context_mask,
            query_times=req.query_times,
            query_mask=req.query_mask,
            target_values=req.target_values,
        )
        assert event.target_values is not None
        assert req.target_values is not None
        assert_close(
            event.target_values[event.query_indices],
            req.target_values,
            atol=0.0,
            rtol=0.0,
            equal_nan=True,
        )

    @pytest.mark.parametrize("batch_shape", [(), (3,), (2, 3)])
    def test_query_indices_random(self, batch_shape: tuple[int, ...]) -> None:
        req = make_forecasting_request(
            seed=3141,
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

    @pytest.mark.parametrize("batch_shape", [(), (3,), (2, 3)])
    def test_query_indices_batch_last(self, batch_shape: tuple[int, ...]) -> None:
        req = make_forecasting_request(
            seed=3141,
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


class TestTripletBatch:
    def test_from_request_unbatched(self) -> None:
        req = SplitTimeData(
            context_times=torch.tensor([1.0, 3.0]),
            context_values=torch.tensor([[10.0, nan], [nan, 30.0]]),
            context_mask=torch.tensor([[True, False], [False, True]]),
            query_times=torch.tensor([2.0, 4.0]),
            query_mask=torch.tensor([[True, False], [True, True]]),
            target_values=torch.tensor([[20.0, nan], [40.0, 41.0]]),
            static_covariates=torch.tensor([5.0, 6.0]),
        )

        actual = TripletBatch.from_request(
            context_times=req.context_times,
            context_values=req.context_values,
            context_mask=req.context_mask,
            query_times=req.query_times,
            query_mask=req.query_mask,
            target_values=req.target_values,
            static_covariates=req.static_covariates,
        )
        expected = req.to_triplets()

        _assert_triplet_equal(actual, expected)

    @pytest.mark.parametrize("batch_shape", [(), (3,), (2, 3)])
    def test_from_request_random(self, batch_shape: tuple[int, ...]) -> None:
        req = make_forecasting_request(
            seed=3141,
            batch_shape=batch_shape,
            min_steps=1,
            max_steps=4,
            context_shape=(3,),
            output_shape=(4,),
            input_missingness=True,
            target_missingness=True,
        )

        actual = TripletBatch.from_request(
            context_times=req.context_times,
            context_values=req.context_values,
            context_mask=req.context_mask,
            query_times=req.query_times,
            query_mask=req.query_mask,
            target_values=req.target_values,
            static_covariates=req.static_covariates,
        )
        expected = req.to_triplets()

        _assert_batched_triplet_equal(actual, expected)

    @pytest.mark.parametrize("batch_shape", [(), (3,), (2, 3)])
    def test_from_request_batch_last(self, batch_shape: tuple[int, ...]) -> None:
        req = make_forecasting_request(
            seed=3141,
            batch_shape=batch_shape,
            min_steps=1,
            max_steps=4,
            context_shape=(3,),
            output_shape=(4,),
            input_missingness=True,
            target_missingness=True,
        )

        ctx_times = req.context_times.movedim(-1, 0)
        ctx_values = req.context_values.movedim(-2, 0)
        ctx_mask = req.context_mask.movedim(-2, 0)
        qry_times = req.query_times.movedim(-1, 0)
        qry_mask = req.query_mask.movedim(-2, 0)
        tgt_values = (
            req.target_values.movedim(-2, 0) if req.target_values is not None else None
        )

        actual = TripletBatch.from_request(
            context_times=ctx_times,
            context_values=ctx_values,
            context_mask=ctx_mask,
            query_times=qry_times,
            query_mask=qry_mask,
            target_values=tgt_values,
            static_covariates=req.static_covariates,
            batch_first=False,
        )
        expected = req.to_triplets()

        _assert_batched_triplet_equal(
            TripletTimeData(
                context_times=actual.context_times.movedim(0, -1),
                context_channels=actual.context_channels.movedim(0, -1),
                context_values=actual.context_values.movedim(0, -1),
                query_times=actual.query_times.movedim(0, -1),
                query_channels=actual.query_channels.movedim(0, -1),
                target_values=(
                    actual.target_values.movedim(0, -1)
                    if actual.target_values is not None
                    else None
                ),
                static_covariates=actual.static_covariates,
            ),
            expected,
        )
