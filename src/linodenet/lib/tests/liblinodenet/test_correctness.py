r"""Some simple tests for the singular triplet method.

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
import pytest
import scipy
import torch
from numpy.random import default_rng
from scipy.stats import ortho_group

import linodenet
from linodenet.lib import (  # singular_triplet,; singular_triplet_native,; singular_triplet_riemann,;
    SingularTriplet,
    SpectralNorm,
    spectral_norm,
    spectral_norm_native,
    spectral_norm_riemann,
)

RANK_ONE_SHAPES: list[tuple[int, int]] = [
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
SVD_METHODS: dict[str, Callable] = {
    "numpy_svd": np.linalg.svd,
    "scipy_svd": scipy.linalg.svd,
    "torch_svd": torch.linalg.svd,
    "linodenet_svd": linodenet.lib.singular_triplet,
}
SPECTRAL_NORMS: dict[str, SpectralNorm | SingularTriplet] = {
    "custom": spectral_norm,
    "native": spectral_norm_native,
    "riemann": spectral_norm_riemann,
}
SHAPES: list[tuple[int, int]] = [
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
RTOL = 1e-5


def snorm(x: torch.Tensor) -> torch.Tensor:
    r"""Scaled norm of a tensor."""
    return x.pow(2).mean().sqrt()


def random_rank_one_matrix(m: int, n: int) -> np.ndarray:
    rng = default_rng(seed=0)
    ustar = rng.standard_normal(m)
    vstar = rng.standard_normal(n)
    return np.outer(ustar, vstar)


# noinspection PyTupleAssignmentBalance
@pytest.mark.parametrize("seed", SEEDS, ids=lambda seed: f"{seed=}")
@pytest.mark.parametrize(
    ("atol", "rtol"), [pytest.param(ATOL, RTOL, id=f"atol={ATOL},rtol={RTOL}")]
)
@pytest.mark.parametrize("shape", SHAPES, ids=lambda shape: f"{shape=}")
@pytest.mark.parametrize("method", SVD_METHODS)
def test_svd_rank_one(
    *,
    method: str,
    shape: tuple[int, int],
    seed: int,
    atol: float,
    rtol: float,
) -> None:
    r"""Checks that the singular triplet method works for rank one matrices."""
    torch.manual_seed(seed)
    m, n = shape
    matrix = random_rank_one_matrix(m, n)

    match SVD_METHODS[method]:
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
            assert torch.allclose(s * torch.outer(u, v), B, atol=atol, rtol=rtol)
        case _:
            raise ValueError(f"Unknown method: {method}")


@pytest.mark.parametrize("seed", SEEDS, ids=lambda seed: f"{seed=}")
@pytest.mark.parametrize(
    ("atol", "rtol"), [pytest.param(ATOL, RTOL, id=f"atol={ATOL},rtol={RTOL}")]
)
@pytest.mark.parametrize("shape", SHAPES, ids=lambda shape: f"{shape=}")
@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("method", SPECTRAL_NORMS)
def test_rank_one(
    *,
    method: str,
    shape: tuple[int, int],
    seed: int,
    device: str | torch.device,
    atol: float,
    rtol: float,
) -> None:
    r"""Test the accuracy of the gradient for rank one matrices.

    The analytical gradient is ∂‖A‖₂/∂A = uvᵀ, where u and v are the singular vectors.
    In particular, for rank one matrices A=uvᵀ, the gradient is ∂‖A‖₂/∂A = uvᵀ/(‖u‖⋅‖v‖).
    """
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    device = torch.device(device)
    impl = SPECTRAL_NORMS[method]
    torch.manual_seed(seed)

    # generate random rank one matrix
    m, n = shape
    sigma_star = 1000 * torch.rand((), device=device) + 1
    u_star = torch.randn(m, device=device)
    u_star = u_star / u_star.norm()
    v_star = torch.randn(n, device=device)
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


@pytest.mark.parametrize("seed", SEEDS, ids=lambda seed: f"{seed=}")
@pytest.mark.parametrize(
    ("atol", "rtol"), [pytest.param(ATOL, RTOL, id=f"atol={ATOL},rtol={RTOL}")]
)
@pytest.mark.parametrize("dim", DIMS, ids=lambda dim: f"{dim=}")
@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("method", SPECTRAL_NORMS)
def test_diagonal(
    *,
    method: str,
    device: str | torch.device,
    dim: int,
    seed: int,
    atol: float,
    rtol: float,
) -> None:
    r"""Checks that the singular triplet method works for diagonal matrices.

    NOTE: builtin SVD seems to have auto-detection for diagonal matrices...
    """
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    device = torch.device(device)
    impl = SPECTRAL_NORMS[method]
    torch.manual_seed(seed)

    # generate a random diagonal matrix
    S = 10 * (torch.randn(dim, device=device) + ATOL)
    A = torch.diag(S)
    A = A.clone().detach().requires_grad_(True)

    # analytical result
    idx_star = S.abs().argmax()
    unit_vector = torch.eye(dim, device=device)
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


@pytest.mark.parametrize("seed", SEEDS, ids=lambda seed: f"{seed=}")
@pytest.mark.parametrize(
    ("atol", "rtol"), [pytest.param(ATOL, RTOL, id=f"atol={ATOL},rtol={RTOL}")]
)
@pytest.mark.parametrize("shape", SHAPES, ids=lambda shape: f"{shape=}")
@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("method", SPECTRAL_NORMS)
def test_analytical(
    *,
    method: str,
    device: str | torch.device,
    shape: tuple[int, int],
    seed: int,
    atol: float,
    rtol: float,
) -> None:
    r"""We test the analytical result for random matrices.

    We randomly sample U, S and V.
    """
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    device = torch.device(device)
    impl = SPECTRAL_NORMS[method]
    torch.manual_seed(seed)
    rng = default_rng(seed=seed)

    # randomly generate a matrix with known SVD
    M, N = shape
    K = min(M, N)
    S = 10 * (torch.rand(K, device=device) + ATOL)
    _U = ortho_group.rvs(M, random_state=rng)
    _V = ortho_group.rvs(N, random_state=rng)
    # take the first K vectors
    U = torch.tensor(_U[:, :K], dtype=torch.float, device=device)
    Vh = torch.tensor(_V[:, :K].T, dtype=torch.float, device=device)
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


@pytest.mark.xfail(reason="Algorithms are unstable for repeated singular values.")
@pytest.mark.parametrize("seed", SEEDS, ids=lambda seed: f"{seed=}")
@pytest.mark.parametrize(
    ("atol", "rtol"), [pytest.param(ATOL, RTOL, id=f"atol={ATOL},rtol={RTOL}")]
)
@pytest.mark.parametrize("dim", DIMS, ids=lambda dim: f"{dim=}")
@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("method", SPECTRAL_NORMS)
def test_orthogonal(
    *,
    method: str,
    device: str | torch.device,
    dim: int,
    seed: int,
    atol: float,
    rtol: float,
) -> None:
    r"""Tests algorithm against orthogonal matrix.

    Note:
        For repeated singular values, the gradient is no longer well-defined,
        but (I think), any sum  $∑_{j≤i} uᵢvᵢᵀ$ is a subgradient.
        In particular, if A is already orthogonal, then $∂‖A‖₂/∂A = A$.
        Is, in some sense, the largest subgradient.
    """
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    device = torch.device(device)
    impl = SPECTRAL_NORMS[method]
    torch.manual_seed(seed)
    rng = default_rng(seed=seed)

    # sample random orthogonal matrix
    U = ortho_group.rvs(dim, random_state=rng)
    S = torch.ones(dim, dtype=torch.float, device=device)
    A = torch.from_numpy(U).to(dtype=torch.float, device=device).requires_grad_(True)

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
