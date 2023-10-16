#!/usr/bin/env python
"""Test gradients of custom operators."""


__all__ = ["compute_spectral_norm_impl"]

import torch
from pytest import mark
from torch import nn

from linodenet.lib import (
    singular_triplet,
    singular_triplet_native,
    spectral_norm,
    spectral_norm_native,
)


def inner(x, y):
    """Compute the inner product."""
    return torch.einsum("..., ... ->", x, y)


def compute_spectral_norm_impl(impl, shape, **kwargs):
    """Test the spectral norm implementation."""
    m, n = shape
    A0 = torch.randn(m, n)

    # outer gradients
    g_s = torch.randn(())

    # Native Forward
    A_native = nn.Parameter(A0.clone())
    s_native = spectral_norm_native(A_native)

    # Native Backward
    r_native = inner(g_s, s_native)
    r_native.backward()

    # Native Gradient
    assert A_native.grad is not None
    g_native = A_native.grad.clone().detach()

    # Custom Forward
    A_custom = nn.Parameter(A0.clone())
    s_custom = impl(A_custom, **kwargs)

    # Custom Backward
    r_custom = inner(g_s, s_custom)
    r_custom.backward()

    # Custom Gradient
    assert A_custom.grad is not None
    g_custom = A_custom.grad.clone().detach()

    # Compute the errors
    err_value = (s_custom - s_native).norm() / s_native.norm()
    err_grads = (g_custom - g_native).norm() / g_native.norm()

    return err_value, err_grads


def compute_singular_triplet_impl(impl, shape, **kwargs):
    """Test the spectral norm implementation."""
    m, n = shape
    A0 = torch.randn(m, n)

    # outer gradients
    g_s = torch.randn(())
    g_u = torch.randn(m)
    g_v = torch.randn(n)

    # Native Forward
    A_native = nn.Parameter(A0.clone())
    s_native, u_native, v_native = singular_triplet_native(A_native)

    # Native Backward
    r_native = inner(g_s, s_native) + inner(g_u, u_native) + inner(g_v, v_native)
    r_native.backward()

    # Native Gradient
    assert A_native.grad is not None
    g_native = A_native.grad.clone().detach()

    # Custom Forward
    A_custom = nn.Parameter(A0.clone())
    s_custom, u_custom, v_custom = impl(A_custom, **kwargs)

    # Custom Backward
    r_custom = inner(g_s, s_custom) + inner(g_u, u_custom) + inner(g_v, v_custom)
    r_custom.backward()

    # Custom Gradient
    assert A_custom.grad is not None
    g_custom = A_custom.grad.clone().detach()

    # Compute the errors
    err_value = (s_custom - s_native).norm() / s_native.norm()
    err_grads = (g_custom - g_native).norm() / g_native.norm()

    return err_value, err_grads


@mark.xfail(reason="Matrices badly conditioned.")
def test_singular_triplet(value_tol: float = 1e-5, grads_tol: float = 1e-3) -> None:
    """Test the singular triplet."""
    err_vals = []
    err_grad = []
    torch.manual_seed(0)
    for _ in range(100):
        m, n = 32, 32
        err_value, err_grads = compute_singular_triplet_impl(singular_triplet, (m, n))
        err_vals.append(err_value.item())
        err_grad.append(err_grads.item())
    avgerr_vals = sum(err_vals) / len(err_vals)
    avgerr_grad = sum(err_grad) / len(err_grad)
    print(f"Average Error:: {avgerr_vals:.3e}, grad: {avgerr_grad:.3e}")
    assert (
        avgerr_vals < value_tol
    ), f"Value error too large! {avgerr_vals:.3e} > {value_tol=}"
    assert (
        avgerr_grad < grads_tol
    ), f"Grads error too large! {avgerr_grad:.3e} > {grads_tol=}"
    print("All tests passed.")


@mark.xfail(reason="Matrices badly conditioned.")
def test_spectral_norm(value_tol: float = 1e-5, grads_tol: float = 1e-3) -> None:
    """Test the spectral norm."""
    err_vals = []
    err_grad = []
    torch.manual_seed(0)
    for _ in range(100):
        m, n = 16, 16
        err_value, err_grads = compute_spectral_norm_impl(spectral_norm, (m, n))
        err_vals.append(err_value.item())
        err_grad.append(err_grads.item())
    avgerr_vals = sum(err_vals) / len(err_vals)
    avgerr_grad = sum(err_grad) / len(err_grad)
    print(f"Average Error:: {avgerr_vals:.3e}, grad: {avgerr_grad:.3e}")
    assert (
        avgerr_vals < value_tol
    ), f"Value error too large! {avgerr_vals:.3e} > {value_tol=}"
    assert (
        avgerr_grad < grads_tol
    ), f"Grads error too large! {avgerr_grad:.3e} > {grads_tol=}"
    print("All tests passed.")


if __name__ == "__main__":
    # main program
    test_spectral_norm()
    test_singular_triplet()
