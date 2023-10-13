#!/usr/bin/env python
"""Benchmark spectral norm / singular triplet implementations."""

__all__ = [
    "test_spectral_norm_forward",
    "test_spectral_norm_backward",
    "test_singular_triplet_forward",
    "test_singular_triplet_backward",
    "test_singular_triplet_backward_full",
]


from collections.abc import Callable

import pytest
import torch
import torch.utils.cpp_extension
from pytest_benchmark.fixture import BenchmarkFixture
from torch import nn

from linodenet.lib import (
    singular_triplet,
    singular_triplet_native,
    spectral_norm,
    spectral_norm_native,
)

torch.manual_seed(0)

DEVICES = ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]
NORM_ONLY = {True: "norm_only", False: "triplet"}
SHAPES = [
    # m > n
    (8, 4),
    (32, 16),
    (128, 64),
    (512, 256),
    # m < n
    (4, 8),
    (16, 32),
    (64, 128),
    (256, 512),
    # m == n
    (8, 8),
    (32, 32),
    (128, 128),
    (512, 512),
]

SPECTRAL_NORMS = {
    spectral_norm_native: "native",
    # jit.script(spectral_norm_native): "native+jit",
    # torch.compile(spectral_norm_native): "native+compile",
    spectral_norm: "custom",
    # jit.script(spectral_norm): "custom+jit",
    # torch.ops.custom.spectral_norm: "unwrapped",
    # jit.script(torch.ops.custom.spectral_norm): "unwrapped+jit",
    # torch.compile(spectral_norm): "custom+compile",
}

SINGULAR_TRIPLETS = {
    singular_triplet_native: "native",
    singular_triplet: "custom",
}


def get_param(shape: tuple[int, int], device: str | torch.device) -> nn.Parameter:
    """Get a random parameter of shape (m, n)."""
    n = shape[-1]
    A = torch.randn(shape, device=device) / torch.sqrt(torch.tensor(n))
    return nn.Parameter(A)


@pytest.mark.parametrize("shape", [(256, 256)], ids=lambda x: f"{x[0]}x{x[1]}")
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("impl", SPECTRAL_NORMS, ids=SPECTRAL_NORMS.get)
@pytest.mark.benchmark(group="spectral_norm_backward")
def test_spectral_norm_forward(
    benchmark: BenchmarkFixture, impl: Callable, device: str, shape: tuple[int, int]
) -> None:
    """Test the spectral norm implementation."""
    torch.manual_seed(0)
    A_original = get_param(shape, device)

    # get reference gradient
    A_native = nn.Parameter(A_original.clone().detach())
    s_native = spectral_norm_native(A_native)

    # get custom gradient
    A_custom = nn.Parameter(A_original.clone().detach())
    s_custom = impl(A_custom)

    # check correctness
    residual = s_custom - s_native
    assert (
        residual.norm() < 1e-06 + 1e-4 * s_native.norm()
    ), "Large error in spectral norm value!"

    def setup():  # get args and kwargs for benchmark
        param = get_param(shape, device)
        return (param,), {}

    with torch.no_grad():
        benchmark.pedantic(impl, setup=setup, rounds=512, warmup_rounds=128)


@pytest.mark.parametrize("shape", [(256, 256)], ids=lambda x: f"{x[0]}x{x[1]}")
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("impl", SPECTRAL_NORMS, ids=SPECTRAL_NORMS.get)
@pytest.mark.benchmark(group="spectral_norm_forward")
def test_spectral_norm_backward(
    benchmark: BenchmarkFixture, impl: Callable, device: str, shape: tuple[int, int]
) -> None:
    """Test the spectral norm implementation."""
    torch.manual_seed(0)
    A_original = get_param(shape, device)
    g_s = torch.randn((), device=device)

    def backward(s):
        loss = g_s * s
        loss.backward()

    # get reference gradient
    A_native = nn.Parameter(A_original.clone().detach())
    s_native = spectral_norm_native(A_native)
    backward(s_native)
    assert A_native.grad is not None
    g_native = A_native.grad.clone().detach()

    # get custom gradient
    A_custom = nn.Parameter(A_original.clone().detach())
    s_custom = impl(A_custom)
    backward(s_custom)
    assert A_custom.grad is not None
    g_custom = A_custom.grad.clone().detach()

    # check correctness
    residual = g_custom - g_native
    assert (
        residual.norm() < 1e-06 + 1e-2 * g_native.norm()
    ), "Large error in spectral norm gradient!"

    # perform benchmark
    def setup():  # get args and kwargs for benchmark
        param = get_param(shape, device)
        output = impl(param)
        return (output,), {}

    benchmark.pedantic(backward, setup=setup, rounds=512, warmup_rounds=128)


@pytest.mark.parametrize("shape", [(256, 256)], ids=lambda x: f"{x[0]}x{x[1]}")
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("impl", SINGULAR_TRIPLETS, ids=SINGULAR_TRIPLETS.get)
@pytest.mark.benchmark(group="singular_triplet_backward")
def test_singular_triplet_forward(
    benchmark: BenchmarkFixture, impl: Callable, device: str, shape: tuple[int, int]
) -> None:
    """Test the spectral norm implementation."""
    torch.manual_seed(0)
    A_original = get_param(shape, device)

    # get reference gradient
    A_native = nn.Parameter(A_original.clone().detach())
    s_native, u_native, v_native = singular_triplet_native(A_native)
    dyadic_native = torch.outer(u_native, v_native)

    # get custom gradient
    A_custom = nn.Parameter(A_original.clone().detach())
    s_custom, u_custom, v_custom = impl(A_custom)
    dyadic_custom = torch.outer(u_custom, v_custom)

    # check correctness
    assert (
        s_custom - s_native
    ).norm() < 1e-06 + 1e-4 * s_native.norm(), "Large error in singular value!"
    assert (
        dyadic_custom - dyadic_native
    ).norm() < 1e-06 + 1e-4 * dyadic_native.norm(), "Large error in singular vectors!"

    def setup():  # get args and kwargs for benchmark
        param = get_param(shape, device)
        return (param,), {}

    # perform benchmark
    with torch.no_grad():
        benchmark.pedantic(impl, setup=setup, rounds=512, warmup_rounds=128)


@pytest.mark.parametrize("shape", [(256, 256)], ids=lambda x: f"{x[0]}x{x[1]}")
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("impl", SINGULAR_TRIPLETS, ids=SINGULAR_TRIPLETS.get)
@pytest.mark.benchmark(group="singular_triplet_forward")
def test_singular_triplet_backward(
    benchmark: BenchmarkFixture, impl: Callable, device: str, shape: tuple[int, int]
) -> None:
    """Test simplified backward when only singular value used."""
    torch.manual_seed(0)
    A_original = get_param(shape, device)
    g_s = torch.randn((), device=device)

    def backward(s):
        loss = g_s * s
        loss.backward()

    # get reference gradient
    A_native = nn.Parameter(A_original.clone().detach())
    s_native, _, _ = singular_triplet_native(A_native)
    backward(s_native)
    assert A_native.grad is not None
    g_native = A_native.grad.clone().detach()

    # get custom gradient
    A_custom = nn.Parameter(A_original.clone().detach())
    s_custom, _, _ = impl(A_custom)
    backward(s_custom)
    assert A_custom.grad is not None
    g_custom = A_custom.grad.clone().detach()

    # check correctness
    assert (
        g_custom - g_native
    ).norm() < 1e-06 + 1e-2 * g_native.norm(), "Large error in spectral norm gradient!"

    def setup():  # get args and kwargs for benchmark
        param = get_param(shape, device)
        s, _, _ = impl(param)
        return (s,), {}

    # perform benchmark
    benchmark.pedantic(backward, setup=setup, rounds=512, warmup_rounds=128)


@pytest.mark.parametrize("shape", [(256, 256)], ids=lambda x: f"{x[0]}x{x[1]}")
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("impl", SINGULAR_TRIPLETS, ids=SINGULAR_TRIPLETS.get)
@pytest.mark.benchmark(group="singular_triplet_forward")
def test_singular_triplet_backward_full(
    benchmark: BenchmarkFixture, impl: Callable, device: str, shape: tuple[int, int]
) -> None:
    """Test full backward when singular triplet used."""
    torch.manual_seed(0)
    m, n = shape
    A_original = get_param(shape, device)
    g_s = torch.randn((), device=device)
    g_u = torch.randn(m, device=device)
    g_v = torch.randn(n, device=device)

    def backward(s, u, v):
        loss = g_s * s + g_u.dot(u) + g_v.dot(v)
        loss.backward()

    # get reference gradient
    A_native = nn.Parameter(A_original.clone().detach())
    s_native, u_native, v_native = singular_triplet_native(A_native)
    backward(s_native, u_native, v_native)
    assert A_native.grad is not None
    g_native = A_native.grad.clone().detach()

    # get custom gradient
    A_custom = nn.Parameter(A_original.clone().detach())
    s_custom, u_custom, v_custom = impl(A_custom)
    backward(s_custom, u_custom, v_custom)
    assert A_custom.grad is not None
    g_custom = A_custom.grad.clone().detach()

    # check correctness
    residual = g_custom - g_native
    assert (
        residual.norm() < 1e-06 + 1e-2 * g_native.norm()
    ), "Large error in spectral norm gradient!"

    def setup():  # get args and kwargs for benchmark
        param = get_param(shape, device)
        s, u, v = impl(param)
        return (s, u, v), {}

    # perform benchmark
    benchmark.pedantic(backward, setup=setup, rounds=512, warmup_rounds=128)


if __name__ == "__main__":
    pytest.main()
