#!/usr/bin/env python
"""Test the spectral norm implementation.""" ""

import pytest
import torch
import torch.utils.cpp_extension
from torch import nn

from linodenet.lib import (
    singular_triplet,
    singular_triplet_native,
    spectral_norm,
    spectral_norm_native,
)
from linodenet.utils import timer

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


@pytest.mark.flaky(reruns=3)
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("shape", SHAPES, ids=lambda x: f"{x[0]}x{x[1]}")
def test_spectral_norm(device: str, shape: tuple[int, int]) -> None:
    """Test the spectral norm implementation."""
    m, n = shape
    A0 = torch.randn(m, n, device=device)
    cond = torch.linalg.cond(A0)

    # Native Forward
    A_native = nn.Parameter(A0.clone())
    with timer() as time:
        s_native = spectral_norm_native(A_native)
    time_val_native = time.elapsed

    # Native Backward
    with timer() as time:
        s_native.backward()
        assert A_native.grad is not None
        g_native = A_native.grad.clone().detach()
    time_grad_native = time.elapsed

    # Custom Forward
    A_custom = nn.Parameter(A0.clone())
    with timer() as time:
        s_custom = spectral_norm(A_custom)
    time_val_custom = time.elapsed

    # Custom Backward
    with timer() as time:
        s_custom.backward()
        assert A_custom.grad is not None
        g_custom = A_custom.grad.clone().detach()
    time_grad_custom = time.elapsed

    err_value = torch.norm(s_custom - s_native) / torch.norm(s_native)
    err_grads = torch.norm(g_custom - g_native) / torch.norm(g_native)
    print(f"{shape=}  {device=}  {cond=}")
    print(f"ERRORS: value: {err_value:.4%}, grad: {err_grads:.4%}")
    print(f"TIME (native): value: {time_val_native:.4f}, grad: {time_grad_native:.4f}")
    print(f"TIME (custom): value: {time_val_custom:.4f}, grad: {time_grad_custom:.4f}")

    assert s_custom > 0, "Singular value is non-positive."
    assert err_value < 1e-4, "Large error in spectral norm value"
    assert err_grads < 1e-2, "Large error in spectral norm gradient"
    assert time_grad_custom < 1.2 * time_grad_native, "Custom backward is too slow"


@pytest.mark.flaky(reruns=3)
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("shape", SHAPES, ids=lambda x: f"{x[0]}x{x[1]}")
@pytest.mark.parametrize("norm_only", NORM_ONLY, ids=NORM_ONLY.get)
def test_singular_triplet(device: str, shape: tuple[int, int], norm_only: bool) -> None:
    """Test the singular triplet implementation."""
    m, n = shape
    A0 = torch.randn(m, n, device=device)
    xi = torch.randn(1, device=device)
    phi = torch.randn(m, device=device) * (1 - norm_only)
    psi = torch.randn(n, device=device) * (1 - norm_only)
    cond = torch.linalg.cond(A0)

    # Native Forward
    A_native = nn.Parameter(A0.clone())
    with timer() as time:
        s_native, u_native, v_native = singular_triplet_native(A_native)
    time_val_native = time.elapsed

    # Native Backward
    with timer() as time:
        r_native = xi * s_native + phi.dot(u_native) + psi.dot(v_native)
        r_native.backward()
        assert A_native.grad is not None
        g_native = A_native.grad.clone().detach()
    time_grad_native = time.elapsed

    # Custom Forward
    A_custom = nn.Parameter(A0.clone())
    with timer() as time:
        s_custom, u_custom, v_custom = singular_triplet(A_custom)
    time_val_custom = time.elapsed

    # Adjust signs so they match
    if (u_custom - u_native).norm() > (u_custom + u_native).norm():
        u_custom = -1 * u_custom
        v_custom = -1 * v_custom

    # Custom Backward
    with timer() as time:
        # NOTE: We flip signs if appropriate to match the native implementation
        r_custom = xi * s_custom + phi.dot(u_custom) + psi.dot(v_custom)
        r_custom.backward()
        assert A_custom.grad is not None
        g_custom = A_custom.grad.clone().detach()
    time_grad_custom = time.elapsed

    err_s = (s_custom - s_native).norm() / s_native.norm()
    err_u = (u_custom - u_native).norm() / u_native.norm()
    err_v = (v_custom - v_native).norm() / v_native.norm()
    err_grads = (g_custom - g_native).norm() / g_native.norm()

    print(f"{shape=}  {device=}  {cond=}")
    print(f"ERRORS: value: {err_s:.4%}/{err_u:.4%}/{err_v:.4%}, grad: {err_grads:.4%}")
    print(f"TIME (native): value: {time_val_native:.4f}, grad: {time_grad_native:.4f}")
    print(f"TIME (custom): value: {time_val_custom:.4f}, grad: {time_grad_custom:.4f}")
    print((g_native - xi * torch.outer(u_native, v_native)).norm() / g_native.norm())
    print((g_custom - xi * torch.outer(u_custom, v_custom)).norm() / g_custom.norm())

    assert s_custom > 0, "Singular value is non-positive."
    assert err_s < 1e-4, "Large error in spectral norm value"
    assert err_u < 1e-3, "Large error in left singular vector"
    assert err_v < 1e-3, "Large error in right singular vector"
    assert err_grads < 1e-2, "Large error in spectral norm gradient"

    if norm_only:
        assert time_grad_custom < 1.2 * time_grad_native, "Custom backward is too slow"


def _main() -> None:
    """Run the main function."""
    test_spectral_norm("cpu", (128, 128))
    test_singular_triplet("cpu", (128, 128))


if __name__ == "__main__":
    _main()
