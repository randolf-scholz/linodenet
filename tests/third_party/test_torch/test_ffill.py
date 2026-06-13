r"""Benchmarks for forward-fill implementations in PyTorch.

References:
    - https://stackoverflow.com/questions/77202743/how-to-efficiently-implement-forward-fill-in-pytorch
"""

from collections.abc import Callable

import pytest
import torch
from pytest_benchmark.fixture import BenchmarkFixture
from torch import Tensor
from torch.testing import assert_close

type FFill = Callable[[Tensor], Tensor]


def _flatten_sequence_axis(x: Tensor, /) -> tuple[Tensor, torch.Size]:
    r"""Flatten ``(..., L, D)`` into ``(N, L)`` independent sequences."""
    shape = x.shape
    flat = x.movedim(-2, -1).reshape(-1, shape[-2])
    return flat, shape


def _unflatten_sequence_axis(x: Tensor, /, *, shape: torch.Size) -> Tensor:
    r"""Restore ``(N, L)`` flattened independent sequences to ``(..., L, D)``."""
    return x.reshape(*shape[:-2], shape[-1], shape[-2]).movedim(-1, -2)


def ffill_loop(x: Tensor, /) -> Tensor:
    r"""Forward-fill by scanning along the sequence axis."""
    y = x.clone()
    last = y[..., 0, :]
    for index in range(y.shape[-2]):
        current = y[..., index, :]
        last = torch.where(current.isnan(), last, current)
        y[..., index, :] = last
    return y


def ffill_range_cummax(x: Tensor, /) -> Tensor:
    r"""Forward-fill via range indices followed by ``cummax`` and ``gather``."""
    flat, shape = _flatten_sequence_axis(x)
    length = flat.shape[-1]
    row = torch.arange(flat.shape[0], device=x.device)
    index = torch.arange(length, device=x.device).expand_as(flat)
    index = torch.where(flat.isnan(), 0, index).cummax(dim=-1).values
    filled = flat[row[:, None], index]
    return _unflatten_sequence_axis(filled, shape=shape)


def ffill_bool_cummax(x: Tensor, /) -> Tensor:
    r"""Forward-fill using the indices returned by boolean ``cummax``."""
    index = x.isfinite().cummax(dim=-2).indices
    return x.gather(-2, index)


def ffill_tril(x: Tensor, /) -> Tensor:
    r"""Forward-fill using an explicit lower-triangular dependency matrix."""
    flat, shape = _flatten_sequence_axis(x)
    length = flat.shape[-1]
    row = torch.arange(flat.shape[0], device=x.device)
    index = torch.arange(length, device=x.device)
    lower = torch.ones(length, length, dtype=torch.bool, device=x.device).tril()
    index = torch.where(
        flat.isfinite().unsqueeze(-2) & lower,
        index.reshape(1, 1, length),
        0,
    ).amax(dim=-1)
    filled = flat[row[:, None], index]
    return _unflatten_sequence_axis(filled, shape=shape)


def ffill_best(x: Tensor, /) -> Tensor:
    r"""Forward-fill directly on ``(..., L, D)`` with ``cummax`` and ``gather``."""
    index = torch.arange(x.shape[-2], device=x.device)
    index = index.reshape((1,) * (x.ndim - 2) + (-1, 1))
    index = torch.where(x.isfinite(), index, 0).cummax(dim=-2).values
    return x.gather(-2, index)


FFILL_METHODS: dict[str, FFill] = {
    "loop": ffill_loop,
    "range_cummax": ffill_range_cummax,
    "bool_cummax": ffill_bool_cummax,
    "tril": ffill_tril,
    "best": ffill_best,
}


@pytest.fixture(params=[(32,)])
def batch_shape(request: pytest.FixtureRequest) -> tuple[int, ...]:
    r"""Batch shape used for forward-fill benchmarks."""
    return request.param


@pytest.fixture(params=[64])
def num_dim(request: pytest.FixtureRequest) -> int:
    r"""Number of feature dimensions."""
    return int(request.param)


@pytest.fixture(params=[128])
def sequence_length(request: pytest.FixtureRequest) -> int:
    r"""Sequence length used for forward-fill benchmarks."""
    return int(request.param)


@pytest.fixture(params=["cpu", "cuda"])
def device(request: pytest.FixtureRequest) -> torch.device:
    r"""Device used for forward-fill benchmarks."""
    device = torch.device(str(request.param))
    if device.type == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available.")
    return device


def make_data(
    *,
    batch_shape: tuple[int, ...],
    sequence_length: int,
    num_dim: int,
    device: torch.device,
    seed: int = 0,
) -> Tensor:
    r"""Create random ``(..., L, D)`` data with NaN missing values."""
    generator = torch.Generator().manual_seed(seed)
    shape = (*batch_shape, sequence_length, num_dim)
    x = torch.randn(shape, generator=generator)
    missing = torch.rand(shape, generator=generator) < 0.35
    return torch.where(missing, torch.nan, x).to(device=device)


@pytest.mark.parametrize("method", FFILL_METHODS)
def test_forward_fill_matches_reference(method: str) -> None:
    r"""All forward-fill implementations match the scanning reference."""
    x = torch.tensor(
        [
            [
                [float("nan"), 1.0],
                [2.0, float("nan")],
                [float("nan"), float("nan")],
                [3.0, 4.0],
                [float("nan"), float("nan")],
            ],
            [
                [1.0, float("nan")],
                [float("nan"), 2.0],
                [float("nan"), 3.0],
                [4.0, float("nan")],
                [float("nan"), float("nan")],
            ],
        ]
    )

    actual = FFILL_METHODS[method](x)
    expected = ffill_loop(x)

    assert_close(actual, expected, equal_nan=True)


@pytest.mark.benchmark(warmup=True, disable_gc=True, group="ffill")
@pytest.mark.parametrize("method", FFILL_METHODS)
def test_benchmark_forward_fill(
    benchmark: BenchmarkFixture,
    method: str,
    batch_shape: tuple[int, ...],
    sequence_length: int,
    num_dim: int,
    device: torch.device,
) -> None:
    r"""Benchmark compiled forward-fill implementations."""
    x = make_data(
        batch_shape=batch_shape,
        sequence_length=sequence_length,
        num_dim=num_dim,
        device=device,
    )
    expected = ffill_loop(x)

    compiled = torch.compile(
        FFILL_METHODS[method],
        fullgraph=True,
        backend="cudagraphs",
    )

    def fn() -> Tensor:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        y = compiled(x)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        return y

    # warmup
    for _ in range(20):
        fn()

    # correctness check
    actual = fn()
    assert_close(actual, expected, equal_nan=True)

    benchmark.group = f"ffill/{device.type}"
    num_iterations = 100 if device.type == "cuda" else 20
    benchmark.pedantic(fn, rounds=3, iterations=num_iterations, warmup_rounds=0)
