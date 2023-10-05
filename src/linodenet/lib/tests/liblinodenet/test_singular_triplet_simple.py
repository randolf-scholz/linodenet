#!/usr/bin/env python
"""Some simple tests for the singular triplet method.

Remark:
    - 64-bit floats have a mantissa of 52 bits, and are precise for ≈ 15 decimal digits.
    - 32-bit floats have a mantissa of 23 bits, and are precise for ≈ 6 decimal digits.
    - 16-bit floats have a mantissa of 10 bits, and are precise for ≈ 3 decimal digits.

Therefore, we set the absolute tolerance to 1e-6 and the relative tolerance to 1e-6.
For example tensorflow.debugging.assert_near uses 10⋅eps as the tolerance.

References:
     numpy.finfo
"""

__all__ = ["test_svd_algs_rank_one", "test_rank_one", "test_diagonal"]

from collections.abc import Callable

import numpy as np
import scipy
import torch
from numpy.random import default_rng
from pytest import mark
from scipy.stats import ortho_group

import linodenet


def snorm(x: torch.Tensor) -> torch.Tensor:
    """Scaled norm of a tensor."""
    return x.pow(2).mean().sqrt()


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

ATOL = 1e-5
RTOL = 1e-5


@mark.parametrize("method", SVD_METHODS.values(), ids=SVD_METHODS.keys())
@mark.parametrize("shape", SHAPES, ids=str)
def test_svd_algs_rank_one(shape: tuple[int, int], method: Callable) -> None:
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
def test_rank_one(
    shape: tuple[int, int], impl: Callable, atol: float = ATOL, rtol: float = RTOL
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
    A = A.clone().detach().requires_grad_(True)

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
def test_diagonal(
    dim: int, impl: Callable, atol: float = ATOL, rtol: float = RTOL
) -> None:
    """Checks that the singular triplet method works for diagonal matrices.

    NOTE: builtin SVD seems to have auto-detection for diagonal matrices...
    """
    torch.manual_seed(0)
    diag = 100 * torch.randn(dim)
    A = torch.diag(diag)
    A = A.clone().detach().requires_grad_(True)

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
    assert snorm(A.grad - analytical_grad) < atol + rtol * snorm(analytical_grad)
    assert torch.allclose(A.grad, analytical_grad, atol=atol, rtol=rtol)


@mark.parametrize(
    "shape",
    [
        # square matrices
        (2, 2),
        (4, 4),
        (16, 16),
        (64, 64),
        (256, 256),
        # rectangular matrices
        (2, 16),
        (16, 2),
        (64, 16),
        (16, 64),
        (256, 64),
        (64, 256),
    ],
    ids=str,
)
@mark.parametrize(
    "impl",
    [
        linodenet.lib.spectral_norm,
        linodenet.lib.spectral_norm_native,
        linodenet.lib.singular_triplet,
        linodenet.lib.singular_triplet_native,
    ],
)
def test_analytical(
    shape: tuple[int, int], impl: Callable, atol: float = ATOL, rtol: float = RTOL
) -> None:
    """We test the analytical result for random matrices.

    We randomly sample U, S and V.
    """
    torch.manual_seed(0)
    M, N = shape
    K = min(M, N)
    S = 100 * torch.rand(K)
    np.random.seed(0)
    U = ortho_group.rvs(M)
    V = ortho_group.rvs(N)
    # take the first K vectors
    U = torch.tensor(U[:, :K], dtype=torch.float)
    Vh = torch.tensor(V[:, :K], dtype=torch.float).T
    A = torch.einsum("ij,j,jk->ik", U, S, Vh)
    A = A.clone().detach().requires_grad_(True)

    # analytical result
    sigma_star = S.max()
    u_star = U[:, S.argmax()]
    v_star = Vh[S.argmax(), :]

    # analytical result
    analytical_value = sigma_star
    analytical_grad = torch.outer(u_star, v_star)

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
    assert snorm(A.grad - analytical_grad) < atol + rtol * snorm(analytical_grad)
    assert torch.allclose(
        A.grad, analytical_grad, atol=atol, rtol=rtol
    ), f"Max element-wise error: {(A.grad - analytical_grad).abs().max()}"


if __name__ == "__main__":
    pass
