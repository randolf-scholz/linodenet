#!/usr/bin/env python
"""Benchmark spectral norm / singular triplet implementations."""

__all__ = [
    "test_spectral_norm",
    "test_spectral_norm_backward",
    "test_spectral_norm_forward",
    "test_singular_triplet_forward",
    "test_singular_triplet_backward",
    "test_singular_triplet_full_backward",
]


from collections.abc import Callable

import pytest
import torch
import torch.utils.cpp_extension
from pytest import mark
from pytest_benchmark.fixture import BenchmarkFixture
from torch import nn

from linodenet.lib import (
    singular_triplet,
    singular_triplet_native,
    singular_triplet_riemann,
    spectral_norm,
    spectral_norm_native,
    spectral_norm_riemann,
)

torch.manual_seed(0)

DEVICES = ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]
NORM_ONLY = {True: "norm_only", False: "triplet"}
# SHAPES = [
#     # m > n
#     (8, 4),
#     (32, 16),
#     (128, 64),
#     (512, 256),
#     # m < n
#     (4, 8),
#     (16, 32),
#     (64, 128),
#     (256, 512),
#     # m == n
#     (8, 8),
#     (32, 32),
#     (128, 128),
#     (512, 512),
# ]

SPECTRAL_NORMS = {
    spectral_norm: "custom",
    spectral_norm_native: "native",
    spectral_norm_riemann: "riemann",
    # spectral_norm_debug: "debug",
    # jit.script(spectral_norm_native): "native+jit",
    # torch.compile(spectral_norm_native): "native+compile",
    # jit.script(spectral_norm): "custom+jit",
    # torch.ops.custom.spectral_norm: "unwrapped",
    # jit.script(torch.ops.custom.spectral_norm): "unwrapped+jit",
    # torch.compile(spectral_norm): "custom+compile",
}

SINGULAR_TRIPLETS = {
    singular_triplet: "custom",
    singular_triplet_native: "native",
    singular_triplet_riemann: "riemann",
}
ATOL = 1e-5
RTOL = 1e-3
SHAPES = [
    (64, 64),
    (256, 256),
    (512, 512),
]
ROUNDS = {
    16: 1024,
    32: 512,
    64: 512,
    128: 256,
    256: 128,
    512: 32,
    1024: 16,
}


def get_param(
    shape: tuple[int, int],
    *,
    device: str | torch.device,
    generator: torch.Generator,
) -> nn.Parameter:
    """Get a random parameter of shape (m, n)."""
    n = shape[-1]
    A = torch.randn(shape, device=device, generator=generator) / torch.sqrt(
        torch.tensor(n)
    )
    return nn.Parameter(A)


@mark.parametrize("shape", SHAPES, ids=lambda x: f"{x[0]}x{x[1]}")
@mark.parametrize("device", DEVICES)
@mark.parametrize("impl", SPECTRAL_NORMS, ids=SPECTRAL_NORMS.get)
@mark.benchmark(group="spectral_norm_forward")
def test_spectral_norm_forward(
    benchmark: BenchmarkFixture, impl: Callable, device: str, shape: tuple[int, int]
) -> None:
    """Test the spectral norm forward pass."""
    generator = torch.Generator(device=device)
    generator.manual_seed(0)
    A_original = get_param(shape, device=device, generator=generator)

    # get reference gradient
    A_native = nn.Parameter(A_original.clone().detach())
    s_native = spectral_norm_native(A_native)

    # get custom gradient
    A_custom = nn.Parameter(A_original.clone().detach())
    s_custom = impl(A_custom)

    # check correctness
    residual = s_custom - s_native
    assert (
        residual.norm() < ATOL + RTOL * s_native.norm()
    ), "Large error in spectral norm value!"

    def setup():  # get args and kwargs for benchmark
        param = get_param(shape, device=device, generator=generator)
        return (param,), {}

    with torch.no_grad():
        benchmark.pedantic(
            impl,
            setup=setup,
            rounds=ROUNDS[shape[0]],
            warmup_rounds=ROUNDS[shape[0]] // 4,
        )


@mark.parametrize("shape", SHAPES, ids=lambda x: f"{x[0]}x{x[1]}")
@mark.parametrize("device", DEVICES)
@mark.parametrize("impl", SPECTRAL_NORMS, ids=SPECTRAL_NORMS.get)
@mark.benchmark(group="spectral_norm_backward")
def test_spectral_norm_backward(
    benchmark: BenchmarkFixture, impl: Callable, device: str, shape: tuple[int, int]
) -> None:
    """Test the spectral norm backward pass."""
    generator = torch.Generator(device=device)
    generator.manual_seed(0)
    A_original = get_param(shape, device=device, generator=generator)
    g_s = torch.randn((), device=device, generator=generator)

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
        residual.norm() < ATOL + RTOL * g_native.norm()
    ), "Large error in spectral norm gradient!"

    # perform benchmark
    def setup():  # get args and kwargs for benchmark
        param = get_param(shape, device=device, generator=generator)
        output = impl(param)
        return (output,), {}

    benchmark.pedantic(
        backward,
        setup=setup,
        rounds=ROUNDS[shape[0]],
        warmup_rounds=ROUNDS[shape[0]] // 4,
    )


@mark.parametrize("shape", SHAPES, ids=lambda x: f"{x[0]}x{x[1]}")
@mark.parametrize("device", DEVICES)
@mark.parametrize("impl", SPECTRAL_NORMS, ids=SPECTRAL_NORMS.get)
@mark.benchmark(group="spectral_norm")
def test_spectral_norm(
    benchmark: BenchmarkFixture, impl: Callable, device: str, shape: tuple[int, int]
) -> None:
    """Test the spectral norm forward+backward."""
    generator = torch.Generator(device=device)
    generator.manual_seed(0)
    A_original = get_param(shape, device=device, generator=generator)
    g_s = torch.randn((), device=device, generator=generator)

    def backward(sigma, /):
        loss = g_s * sigma
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
        residual.norm() < ATOL + RTOL * g_native.norm()
    ), "Large error in spectral norm gradient!"

    def func():
        param = get_param(shape, device=device, generator=generator)
        sigma = impl(param)
        loss = g_s * sigma
        loss.backward()

    benchmark.pedantic(
        func,
        rounds=ROUNDS[shape[0]],
        warmup_rounds=ROUNDS[shape[0]] // 4,
    )


@mark.parametrize("shape", [(256, 256)], ids=lambda x: f"{x[0]}x{x[1]}")
@mark.parametrize("device", DEVICES)
@mark.parametrize("impl", SINGULAR_TRIPLETS, ids=SINGULAR_TRIPLETS.get)
@mark.benchmark(
    group="singular_triplet_forward",
)
def test_singular_triplet_forward(
    benchmark: BenchmarkFixture, impl: Callable, device: str, shape: tuple[int, int]
) -> None:
    """Test the spectral norm implementation."""
    generator = torch.Generator(device=device)
    generator.manual_seed(0)
    A_original = get_param(shape, device=device, generator=generator)

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
    ).norm() < ATOL + RTOL * s_native.norm(), "Large error in singular value!"
    assert (
        dyadic_custom - dyadic_native
    ).norm() < ATOL + RTOL * dyadic_native.norm(), "Large error in singular vectors!"

    def setup():  # get args and kwargs for benchmark
        param = get_param(shape, device=device, generator=generator)
        return (param,), {}

    # perform benchmark
    with torch.no_grad():
        with torch.no_grad():
            benchmark.pedantic(
                impl,
                setup=setup,
                rounds=ROUNDS[shape[0]],
                warmup_rounds=ROUNDS[shape[0]] // 4,
            )


@mark.parametrize("shape", [(256, 256)], ids=lambda x: f"{x[0]}x{x[1]}")
@mark.parametrize("device", DEVICES)
@mark.parametrize("impl", SINGULAR_TRIPLETS, ids=SINGULAR_TRIPLETS.get)
@mark.benchmark(group="singular_triplet_backward")
def test_singular_triplet_backward(
    benchmark: BenchmarkFixture, impl: Callable, device: str, shape: tuple[int, int]
) -> None:
    """Test simplified backward when only singular value used."""
    torch.manual_seed(0)
    generator = torch.Generator(device=device)
    generator.manual_seed(0)
    A_original = get_param(shape, device=device, generator=generator)
    g_s = torch.randn((), device=device, generator=generator)

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
    ).norm() < ATOL + RTOL * g_native.norm(), "Large error in spectral norm gradient!"

    def setup():  # get args and kwargs for benchmark
        param = get_param(shape, device=device, generator=generator)
        s, _, _ = impl(param)
        return (s,), {}

    # perform benchmark
    benchmark.pedantic(backward, setup=setup, rounds=512, warmup_rounds=128)


@mark.skip(reason="Errors too large for custom impl.")
@mark.parametrize("shape", [(256, 256)], ids=lambda x: f"{x[0]}x{x[1]}")
@mark.parametrize("device", DEVICES)
@mark.parametrize("impl", SINGULAR_TRIPLETS, ids=SINGULAR_TRIPLETS.get)
@mark.benchmark(group="singular_triplet_full_backward")
def test_singular_triplet_full_backward(
    benchmark: BenchmarkFixture, impl: Callable, device: str, shape: tuple[int, int]
) -> None:
    """Test full backward when singular triplet used."""
    torch.manual_seed(0)
    m, n = shape
    generator = torch.Generator(device=device)
    generator.manual_seed(0)
    A_original = get_param(shape, device=device, generator=generator)
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
        residual.norm() < ATOL + RTOL * g_native.norm()
    ), "Large error in spectral norm gradient!"

    def setup():  # get args and kwargs for benchmark
        param = get_param(shape, device=device, generator=generator)
        s, u, v = impl(param)
        return (s, u, v), {}

    # perform benchmark
    benchmark.pedantic(backward, setup=setup, rounds=512, warmup_rounds=128)


if __name__ == "__main__":
    pytest.main()
