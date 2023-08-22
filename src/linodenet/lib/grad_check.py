#!/usr/bin/env python
"""Test gradients of custom operators."""


__all__ = ["compute_spectral_norm_impl"]

import torch
from torch import nn

from linodenet.lib import spectral_norm, spectral_norm_native


def compute_spectral_norm_impl(impl, shape: tuple[int, int]) -> None:
    """Test the spectral norm implementation."""
    m, n = shape
    A0 = torch.randn(m, n)

    # Native Forward
    A_native = nn.Parameter(A0.clone())
    s_native = spectral_norm_native(A_native)
    s_native.backward()
    g_native = A_native.grad.clone().detach()

    # Custom Forward
    A_custom = nn.Parameter(A0.clone())
    s_custom = impl(A_custom, maxiter=2000)

    err_value = torch.norm(s_custom - s_native) / torch.norm(s_native)

    # Custom Backward
    s_custom.backward()
    assert A_custom.grad is not None
    g_custom = A_custom.grad.clone().detach()
    err_grads = torch.norm(g_custom - g_native) / torch.norm(g_native)

    return err_value, err_grads


if __name__ == "__main__":
    # main program
    err_vals = []
    err_grad = []
    torch.manual_seed(0)
    for _ in range(100):
        m, n = 512, 512
        err_value, err_grads = compute_spectral_norm_impl(spectral_norm, (m, n))
        err_vals.append(err_value.item())
        err_grad.append(err_grads.item())
    avgerr_vals = sum(err_vals) / len(err_vals)
    avgerr_grad = sum(err_grad) / len(err_grad)
    print(f"Average Error:: {avgerr_vals:.3e}, grad: {avgerr_grad:.3e}")
    assert avgerr_vals < 1e-5, f"Value Average error too large! {avgerr_vals:.3e}"
    assert avgerr_grad < 1e-3, f"Grads Average error too large! {avgerr_grad:.3e}"
    print("All tests passed.")
