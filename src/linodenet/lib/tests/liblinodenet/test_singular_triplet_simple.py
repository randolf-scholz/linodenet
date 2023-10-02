#!/usr/bin/env python
"""Some simple tests for the singular triplet method."""

__all__ = ["test_rank_one_matrix", "test_grad_rank_one", "test_diagonal_matrix"]

from collections.abc import Callable

import numpy as np
import scipy
import torch
from numpy.random import default_rng
from pytest import mark

import linodenet


def random_rank_one_matrix(m: int, n: int) -> np.ndarray:
    rng = default_rng(seed=0)
    ustar = rng.standard_normal(m)
    vstar = rng.standard_normal(n)
    return np.outer(ustar, vstar)


SVD_METHODS = {
    "numpy_svd": np.linalg.svd,
    "scipy_svd": scipy.linalg.svd,
    "torch_svd": torch.linalg.svd,
    "linodenet_svd": linodenet.lib.singular_triplet,
}


SHAPES = [
    # square matrices
    (1, 1),
    (2, 2),
    (16, 16),
    (64, 64),
    # rectangular matrices
    (1, 16),
    (16, 1),
    (2, 16),
    (16, 2),
    (64, 16),
    (16, 64),
]


@mark.parametrize("method", SVD_METHODS.values(), ids=SVD_METHODS.keys())
@mark.parametrize("shape", SHAPES, ids=str)
def test_rank_one_matrix(shape: tuple[int, int], method: Callable) -> None:
    """Checks that the singular triplet method works for rank one matrices."""
    m, n = shape
    matrix = random_rank_one_matrix(m, n)

    match method:
        case scipy.linalg.svd:
            A = matrix
            U, S, Vh = scipy.linalg.svd(A)
            # cols of U = LSV, rows of Vh: RSV
            u, s, v = U[:, 0], S[0], Vh[0, :]
            assert np.allclose(s * np.outer(u, v), A)
        case np.linalg.svd:
            A = matrix
            U, S, Vh = np.linalg.svd(A)
            # cols of U = LSV, rows of Vh: RSV
            u, s, v = U[:, 0], S[0], Vh[0, :]
            assert np.allclose(s * np.outer(u, v), A)
        case torch.linalg.svd:
            A = torch.from_numpy(matrix)
            U, S, Vh = torch.linalg.svd(A)
            # cols of U = LSV, rows of Vh: RSV
            u, s, v = U[:, 0], S[0], Vh[0, :]
            assert torch.allclose(s * torch.outer(u, v), A)
        case linodenet.lib.singular_triplet:
            A = torch.from_numpy(matrix)
            s, u, v = linodenet.lib.singular_triplet(A)
            assert torch.allclose(s * np.outer(u, v), A)
        case _:
            raise ValueError(f"Unknown method: {method}")


@mark.parametrize("shape", SHAPES, ids=str)
@mark.parametrize(
    "impl",
    [
        linodenet.lib.spectral_norm,
        linodenet.lib.spectral_norm_native,
        linodenet.lib.singular_triplet,
        linodenet.lib.singular_triplet_native,
    ],
)
def test_grad_rank_one(
    shape: tuple[int, int], impl: Callable, atol: float = 1e-8, rtol: float = 1e-5
) -> None:
    """Test the accuracy of the gradient for rank one matrices.

    The analytical gradient is ∂‖A‖₂/∂A = uvᵀ, where u and v are the singular vectors.
    In particular, for rank one matrices A=uvᵀ, the gradient is ∂‖A‖₂/∂A = uvᵀ/(‖u‖⋅‖v‖).
    """
    # generate random rank one matrix
    torch.manual_seed(0)
    m, n = shape
    sigma_star = 1000 * torch.rand(()) + 1
    u_star = torch.randn(m)
    u_star = u_star / u_star.norm()
    v_star = torch.randn(n)
    v_star = v_star / v_star.norm()
    A = sigma_star * torch.outer(u_star, v_star)
    A = torch.tensor(A, requires_grad=True)

    # analytical result
    analytical_value = sigma_star
    analytical_grad = torch.outer(u_star, v_star)

    # forward pass
    outputs = impl(A)
    sigma = outputs[0] if isinstance(outputs, tuple) else outputs

    # check forward pass
    assert torch.allclose(sigma, analytical_value, atol=atol, rtol=rtol)

    # backward pass
    sigma.backward()

    # check backward pass
    assert A.grad is not None
    assert torch.allclose(A.grad, analytical_grad, atol=atol, rtol=rtol)


@mark.parametrize("dim", [1, 2, 4, 16, 64])
@mark.parametrize(
    "impl",
    [
        linodenet.lib.spectral_norm,
        linodenet.lib.spectral_norm_native,
        linodenet.lib.singular_triplet,
        linodenet.lib.singular_triplet_native,
    ],
)
def test_diagonal_matrix(
    dim: int, impl: Callable, atol: float = 1e-8, rtol: float = 1e-5
) -> None:
    """Checks that the singular triplet method works for diagonal matrices."""
    torch.manual_seed(0)
    diag = 100 * torch.randn(dim)
    A = torch.diag(diag)
    A = torch.tensor(A, requires_grad=True)

    # analytical result
    idx_star = torch.argmax(diag.abs())
    unit_vector = torch.eye(dim)
    sigma_star = diag[idx_star].abs()
    sign_star = torch.sign(diag[idx_star])
    u_star = unit_vector[idx_star]
    v_star = unit_vector[idx_star]

    # analytical result
    analytical_value = sigma_star
    analytical_grad = sign_star * torch.outer(u_star, v_star)

    # forward pass
    outputs = impl(A)
    sigma = outputs[0] if isinstance(outputs, tuple) else outputs

    # check forward pass
    assert (sigma - analytical_value).norm() < atol + rtol * analytical_value.norm()
    assert torch.allclose(sigma, analytical_value, atol=atol, rtol=rtol)

    # backward pass
    sigma.backward()

    # check backward pass
    assert A.grad is not None
    assert (A.grad - analytical_grad).norm() < atol + rtol * analytical_grad.norm()
    assert torch.allclose(A.grad, analytical_grad, atol=atol, rtol=rtol)


if __name__ == "__main__":
    pass
