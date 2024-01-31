"""Some simple tests for the singular triplet method.

Remark:
    - 64-bit floats have a mantissa of 52 bits, and are precise for ≈ 15 decimal digits.
    - 32-bit floats have a mantissa of 23 bits, and are precise for ≈ 6 decimal digits.
    - 16-bit floats have a mantissa of 10 bits, and are precise for ≈ 3 decimal digits.

Therefore, we set the absolute tolerance to 1e-6 and the relative tolerance to 1e-6.
For example tensorflow.debugging.assert_near uses 10⋅eps as the tolerance.

References:
     - `numpy.finfo`
"""

__all__ = [
    "test_analytical",
    "test_diagonal",
    "test_orthogonal",
    "test_rank_one",
    "test_svd_rank_one",
]

from collections.abc import Callable

import numpy as np
import scipy
import torch
from numpy.random import default_rng
from pytest import mark
from scipy.stats import ortho_group

import linodenet
from linodenet.lib import (  # singular_triplet,; singular_triplet_native,; singular_triplet_riemann,;
    spectral_norm,
    spectral_norm_native,
    spectral_norm_riemann,
)

RANK_ONE_SHAPES = [
    (1, 1),
    (1, 2),
    (1, 4),
    (1, 16),
    (1, 64),
    (1, 256),
    (2, 1),
    (4, 1),
    (16, 1),
    (64, 1),
    (256, 1),
]
SVD_METHODS = {
    "numpy_svd": np.linalg.svd,
    "scipy_svd": scipy.linalg.svd,
    "torch_svd": torch.linalg.svd,
    "linodenet_svd": linodenet.lib.singular_triplet,
}
IMPL = {
    spectral_norm: "custom",
    spectral_norm_native: "native",
    spectral_norm_riemann: "riemann",
    # singular_triplet: "custom",
    # singular_triplet_native: "native",
    # singular_triplet_riemann: "riemann",
}
SHAPES = [
    # square matrices
    (2, 2),
    (4, 4),
    (16, 16),
    (64, 64),
    (256, 256),
    # rectangular matrices
    (16, 64),
    (256, 64),
]
SEEDS = [1000, 1001, 1002, 1003, 1004]
DIMS = [2, 4, 16, 64, 256]
ATOL = 1e-3
RTOL = 1e-6


def snorm(x: torch.Tensor) -> torch.Tensor:
    """Scaled norm of a tensor."""
    return x.pow(2).mean().sqrt()


def random_rank_one_matrix(m: int, n: int) -> np.ndarray:
    rng = default_rng(seed=0)
    ustar = rng.standard_normal(m)
    vstar = rng.standard_normal(n)
    return np.outer(ustar, vstar)


# noinspection PyTupleAssignmentBalance
@mark.parametrize("seed", SEEDS, ids=lambda x: f"seed={x}")
@mark.parametrize("shape", SHAPES, ids=str)
@mark.parametrize("method", SVD_METHODS.values(), ids=SVD_METHODS.keys())
def test_svd_rank_one(
    method: Callable,
    shape: tuple[int, int],
    seed: int,
    atol: float = ATOL,
    rtol: float = RTOL,
) -> None:
    """Checks that the singular triplet method works for rank one matrices."""
    torch.manual_seed(seed)
    m, n = shape
    matrix = random_rank_one_matrix(m, n)

    match method:
        case scipy.linalg.svd:
            A = matrix
            U, S, Vh = scipy.linalg.svd(A)
            # cols of U = LSV, rows of Vh: RSV
            u, s, v = U[:, 0], S[0], Vh[0, :]
            assert np.allclose(s * np.outer(u, v), A, atol=atol, rtol=rtol)
        case np.linalg.svd:
            A = matrix
            U, S, Vh = np.linalg.svd(A)
            # cols of U = LSV, rows of Vh: RSV
            u, s, v = U[:, 0], S[0], Vh[0, :]
            assert np.allclose(s * np.outer(u, v), A, atol=atol, rtol=rtol)
        case torch.linalg.svd:
            B = torch.from_numpy(matrix)
            U, S, Vh = torch.linalg.svd(B)
            # cols of U = LSV, rows of Vh: RSV
            u, s, v = U[:, 0], S[0], Vh[0, :]
            assert torch.allclose(s * torch.outer(u, v), B, atol=atol, rtol=rtol)
        case linodenet.lib.singular_triplet:
            B = torch.from_numpy(matrix)
            s, u, v = linodenet.lib.singular_triplet(B)
            assert torch.allclose(s * np.outer(u, v), B, atol=atol, rtol=rtol)
        case _:
            raise ValueError(f"Unknown method: {method}")


@mark.parametrize("seed", SEEDS, ids=lambda x: f"seed={x}")
@mark.parametrize("shape", SHAPES, ids=str)
@mark.parametrize("impl", IMPL, ids=IMPL.get)
def test_rank_one(
    impl: Callable,
    shape: tuple[int, int],
    seed: int,
    atol: float = ATOL,
    rtol: float = RTOL,
) -> None:
    """Test the accuracy of the gradient for rank one matrices.

    The analytical gradient is ∂‖A‖₂/∂A = uvᵀ, where u and v are the singular vectors.
    In particular, for rank one matrices A=uvᵀ, the gradient is ∂‖A‖₂/∂A = uvᵀ/(‖u‖⋅‖v‖).
    """
    # generate random rank one matrix
    torch.manual_seed(seed)
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
    assert snorm(A.grad - analytical_grad) < (atol + rtol * snorm(analytical_grad)), (
        f"Max element-wise error: {(A.grad - analytical_grad).abs().max():.3e}"
        f"  ‖A‖₂={analytical_value:.3e}"
        f"  κ(A)={torch.linalg.cond(A):.3e}"
    )
    assert torch.allclose(A.grad, analytical_grad, atol=atol, rtol=rtol), (
        f"Max element-wise error: {(A.grad - analytical_grad).abs().max():.3e}"
        f"  ‖A‖₂={analytical_value:.3e}"
        f"  κ(A)={torch.linalg.cond(A):.3e}"
    )


@mark.parametrize("seed", SEEDS, ids=lambda x: f"seed={x}")
@mark.parametrize("dim", DIMS, ids=lambda x: f"dim={x}")
@mark.parametrize("impl", IMPL, ids=IMPL.get)
def test_diagonal(
    impl: Callable,
    dim: int,
    seed: int,
    atol: float = ATOL,
    rtol: float = RTOL,
) -> None:
    """Checks that the singular triplet method works for diagonal matrices.

    NOTE: builtin SVD seems to have auto-detection for diagonal matrices...
    """
    torch.manual_seed(seed)
    S = 10 * (torch.randn(dim) + ATOL)
    A = torch.diag(S)
    A = A.clone().detach().requires_grad_(True)

    # analytical result
    idx_star = S.abs().argmax()
    unit_vector = torch.eye(dim)
    sigma_star = S[idx_star].abs()
    sign_star = torch.sign(S[idx_star])
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
    assert snorm(A.grad - analytical_grad) < (atol + rtol * snorm(analytical_grad)), (
        f"Max element-wise error: {(A.grad - analytical_grad).abs().max():.3e}"
        f"  ‖A‖₂={analytical_value:.3e}"
        f"  κ(A)={torch.linalg.cond(A):.3e}"
        f"  δ(A)={S.abs().sort().values.diff()[-1]:.3e}"
    )
    assert torch.allclose(A.grad, analytical_grad, atol=atol, rtol=rtol), (
        f"Max element-wise error: {(A.grad - analytical_grad).abs().max():.3e}"
        f"  ‖A‖₂={analytical_value:.3e}"
        f"  κ(A)={torch.linalg.cond(A):.3e}"
        f"  δ(A)={S.abs().sort().values.diff()[-1]:.3e}"
    )


@mark.parametrize("seed", SEEDS, ids=lambda x: f"seed={x}")
@mark.parametrize("shape", SHAPES, ids=str)
@mark.parametrize("impl", IMPL, ids=IMPL.get)
def test_analytical(
    impl: Callable,
    shape: tuple[int, int],
    seed: int,
    atol: float = ATOL,
    rtol: float = RTOL,
) -> None:
    """We test the analytical result for random matrices.

    We randomly sample U, S and V.
    """
    torch.manual_seed(seed)
    rng = default_rng(seed=seed)
    M, N = shape
    K = min(M, N)
    S = 10 * (torch.rand(K) + ATOL)
    U = ortho_group.rvs(M, random_state=rng)
    V = ortho_group.rvs(N, random_state=rng)
    # take the first K vectors
    U = torch.tensor(U[:, :K], dtype=torch.float)
    Vh = torch.tensor(V[:, :K], dtype=torch.float).T
    A = torch.einsum("ij,j,jk->ik", U, S, Vh)
    A = A.clone().detach().requires_grad_(True)

    # analytical result
    idx_star = S.abs().argmax()
    sigma_star = S[idx_star]
    u_star = U[:, idx_star]
    v_star = Vh[idx_star, :]

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
    assert snorm(A.grad - analytical_grad) < (atol + rtol * snorm(analytical_grad)), (
        f"Max element-wise error: {(A.grad - analytical_grad).abs().max():.3e}"
        f"  ‖A‖₂={analytical_value:.3e}"
        f"  κ(A)={torch.linalg.cond(A):.3e}"
        f"  δ(A)={S.abs().sort().values.diff()[-1]:.3e}"
    )
    assert torch.allclose(A.grad, analytical_grad, atol=atol, rtol=rtol), (
        f"Max element-wise error: {(A.grad - analytical_grad).abs().max():.3e}"
        f"  ‖A‖₂={analytical_value:.3e}"
        f"  κ(A)={torch.linalg.cond(A):.3e}"
        f"  δ(A)={S.abs().sort().values.diff()[-1]:.3e}"
    )


@mark.skip(reason="Algorithms are unstable for repeated singular values.")
@mark.parametrize("seed", SEEDS, ids=lambda x: f"seed={x}")
@mark.parametrize("dim", DIMS, ids=lambda x: f"dim={x}")
@mark.parametrize("impl", IMPL, ids=IMPL.get)
def test_orthogonal(
    impl: Callable,
    dim: int,
    seed: int,
    atol: float = ATOL,
    rtol: float = RTOL,
) -> None:
    """Tests algorithm against orthogonal matrix.

    Note:
        For repeated singular values, the gradient is no longer well-defined,
        but (I think), any sum  $∑_{j≤i} uᵢvᵢᵀ$ is a subgradient.
        In particular, if A is already orthogonal, then $∂‖A‖₂/∂A = A$.
        Is, in some sense, the largest subgradient.
    """
    torch.manual_seed(seed)
    rng = default_rng(seed=seed)
    S = torch.ones(dim, dtype=torch.float)
    U = ortho_group.rvs(dim, random_state=rng)
    A = torch.from_numpy(U).to(dtype=torch.float).requires_grad_(True)

    # analytical result
    analytical_value = S[0]
    analytical_grad = A.clone().detach()

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
    assert snorm(A.grad - analytical_grad) < (atol + rtol * snorm(analytical_grad)), (
        f"Max element-wise error: {(A.grad - analytical_grad).abs().max():.3e}"
        f"  ‖A‖₂={analytical_value:.3e}"
        f"  κ(A)={torch.linalg.cond(A):.3e}"
        f"  δ(A)={S.abs().sort().values.diff()[-1]:.3e}"
    )
    assert torch.allclose(A.grad, analytical_grad, atol=atol, rtol=rtol), (
        f"Max element-wise error: {(A.grad - analytical_grad).abs().max():.3e}"
        f"  ‖A‖₂={analytical_value:.3e}"
        f"  κ(A)={torch.linalg.cond(A):.3e}"
        f"  δ(A)={S.abs().sort().values.diff()[-1]:.3e}"
    )
