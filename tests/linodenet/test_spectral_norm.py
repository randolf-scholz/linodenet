#!/usr/bin/env python
"""Test the spectral norm implementation.""" ""

import pytest
import torch
import torch.utils.cpp_extension
from torch import nn

from linodenet.utils import timer
from linodenet.utils.custom_operators import (
    singular_triplet,
    singular_triplet_native,
    spectral_norm,
    spectral_norm_native,
)

DEVICES = ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]
SHAPES = [
    (8, 8),
    (64, 64),
    (128, 128),
    (256, 256),
    (512, 512),
]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("shape", SHAPES)
def test_spectral_norm(device: str, shape: tuple[int, int]) -> None:
    """Test the spectral norm implementation."""
    m, n = shape
    A0 = torch.randn(m, n, device=device)
    cond = torch.linalg.cond(A0)

    # Native Forward
    A = nn.Parameter(A0.clone())
    with timer() as time:
        s_native = spectral_norm_native(A)
    time_val_native = time.elapsed

    # Native Backward
    with timer() as time:
        s_native.backward()
        g_native = A.grad.clone().detach()
    time_grad_native = time.elapsed

    # Custom Forward
    A = nn.Parameter(A0.clone())
    with timer() as time:
        s_custom = spectral_norm(A, maxiter=10_000)
    time_val_custom = time.elapsed

    # Custom Backward
    with timer() as time:
        s_custom.backward()
        g_custom = A.grad.clone().detach()
    time_grad_custom = time.elapsed

    err_value = torch.norm(s_custom - s_native) / torch.norm(s_native)
    err_grads = torch.norm(g_custom - g_native) / torch.norm(g_native)
    print(f"{shape=}  {device=}  {cond=}")
    print(f"ERRORS: value: {err_value:.4%}, grad: {err_grads:.4%}")
    print(f"TIME (native): value: {time_val_native:.4f}, grad: {time_grad_native:.4f}")
    print(f"TIME (custom): value: {time_val_custom:.4f}, grad: {time_grad_custom:.4f}")

    assert err_value < 1e-4, "Large error in spectral norm value"
    assert err_grads < 1e-2, "Large error in spectral norm gradient"
    assert time_grad_custom < 1.2 * time_grad_native, "Custom backward is too slow"


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("shape", SHAPES)
def test_singular_triplet(device: str, shape: tuple[int, int]) -> None:
    """Test the singular triplet implementation."""
    m, n = shape
    A0 = torch.randn(m, n, device=device)
    phi = torch.randn(m, device=device)
    psi = torch.randn(n, device=device)
    cond = torch.linalg.cond(A0)

    # Native Forward
    A = nn.Parameter(A0.clone())
    with timer() as time:
        s_native, u_native, v_native = singular_triplet_native(A)
    time_val_native = time.elapsed

    # Native Backward
    with timer() as time:
        r_native = s_native + phi.dot(u_native) + psi.dot(v_native)
        r_native.backward()
        g_native = A.grad.clone().detach()
    time_grad_native = time.elapsed

    # Custom Forward
    A = nn.Parameter(A0.clone())
    with timer() as time:
        s_custom, u_custom, v_custom = singular_triplet(A)
    time_val_custom = time.elapsed

    # Custom Backward
    with timer() as time:
        r_custom = s_custom + phi.dot(u_custom) + psi.dot(v_custom)
        r_custom.backward()
        g_custom = A.grad.clone().detach()
    time_grad_custom = time.elapsed

    err_value = torch.norm(s_custom - s_native) / torch.norm(s_native)
    err_grads = torch.norm(g_custom - g_native) / torch.norm(g_native)
    print(f"{shape=}  {device=}  {cond=}")
    print(f"ERRORS: value: {err_value:.4%}, grad: {err_grads:.4%}")
    print(f"TIME (native): value: {time_val_native:.4f}, grad: {time_grad_native:.4f}")
    print(f"TIME (custom): value: {time_val_custom:.4f}, grad: {time_grad_custom:.4f}")

    assert err_value < 1e-4, "Large error in spectral norm value"
    assert err_grads < 1e-2, "Large error in spectral norm gradient"
    assert time_grad_custom < 1.2 * time_grad_native, "Custom backward is too slow"


def _main() -> None:
    """Run the main function."""
    test_spectral_norm("cpu", (128, 128))
    test_singular_triplet("cpu", (128, 128))


if __name__ == "__main__":
    _main()
