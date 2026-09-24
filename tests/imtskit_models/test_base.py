r"""Tests for shared forecasting test helpers."""

import pytest
import torch
from torch.testing import assert_close

from .base import make_continuous_time_request


@pytest.mark.parametrize(
    "batch_shape",
    [(), (4,), (2, 1)],
    ids=["batch_shape=()", "batch_shape=(4,)", "batch_shape=(2,1)"],
)
@pytest.mark.parametrize(
    "input_missingness",
    [False, True],
    ids=["input_missingness=False", "input_missingness=True"],
)
def test_make_forecasting_request_is_deterministic_and_valid(
    batch_shape: tuple[int, ...],
    input_missingness: bool,
) -> None:
    kwargs = {
        "rng": 123,
        "batch_shape": batch_shape,
        "min_steps": 2,
        "max_steps": 5,
        "context_shape": (3,),
        "output_shape": (2,),
        "input_missingness": input_missingness,
    }

    actual = make_continuous_time_request(**kwargs)
    repeat = make_continuous_time_request(**kwargs)

    assert_close(actual.context_times, repeat.context_times, equal_nan=True)
    assert_close(actual.context_values, repeat.context_values, equal_nan=True)
    assert torch.equal(actual.context_mask, repeat.context_mask)
    assert_close(actual.query_times, repeat.query_times, equal_nan=True)
    assert torch.equal(actual.query_mask, repeat.query_mask)
    assert actual.target_values is not None
    assert repeat.target_values is not None
    assert_close(actual.target_values, repeat.target_values, equal_nan=True)

    assert actual.context_times.shape == (*batch_shape, kwargs["max_steps"])
    assert actual.context_values.shape == (
        *batch_shape,
        kwargs["max_steps"],
        *kwargs["context_shape"],
    )
    assert actual.query_times.shape == (*batch_shape, kwargs["max_steps"])
    assert actual.query_mask.shape == (
        *batch_shape,
        kwargs["max_steps"],
        *kwargs["output_shape"],
    )
    assert actual.target_values.shape == (
        *batch_shape,
        kwargs["max_steps"],
        *kwargs["output_shape"],
    )

    for request in actual.unbatch():
        assert request.target_values is not None
        assert torch.equal(request.context_values.isfinite(), request.context_mask)
        assert torch.equal(request.target_values.isfinite(), request.query_mask)
        assert request.context_times.diff().gt(0.0).all()
        assert request.query_times.diff().gt(0.0).all()
        assert request.query_times.gt(request.context_times[-1]).all()
        assert (
            request.context_mask.reshape(request.context_mask.shape[0], -1)
            .any(dim=-1)
            .all()
        )
        assert (
            request.query_mask.reshape(request.query_mask.shape[0], -1)
            .any(dim=-1)
            .all()
        )

        if not input_missingness:
            assert request.context_mask.all()
        assert request.query_mask.all()
