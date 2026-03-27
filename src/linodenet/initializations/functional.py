r"""Initializations for the Linear ODE Networks.

All initializations are normalized such that if $x∼𝓝(0,1)$, then $Ax∼𝓝(0,1)$ as well.

Notes:
    - See `linodenet.initializations.functional` for functional implementations.
    - See `linodenet.initializations.modules` for all module-based initializations.
"""

__all__ = [
    # Deterministic Initializations
    "symplectic",
    # Initializations
    "diagonally_dominant",
    "gaussian",
    "low_rank",
    "orthogonal",
    "skew_symmetric",
    "special_orthogonal",
    "symmetric",
    "traceless",
]

from math import sqrt
from typing import Optional

import torch
from torch import Tensor


def _normalize_sample_shape(size: int | tuple[int, ...], /) -> tuple[int, ...]:
    r"""Normalize sample shape arguments."""
    return (size,) if isinstance(size, int) else tuple(size)


def _normalize_matrix_dim(dim: int | tuple[int, int], /) -> tuple[int, int]:
    r"""Normalize matrix dimensions."""
    shape = (dim, dim) if isinstance(dim, int) else tuple(dim)
    if len(shape) != 2:
        raise ValueError(f"Expected a matrix shape, got {shape}.")
    return shape


def _normalize_square_dim(dim: int | tuple[int, int], /) -> int:
    r"""Normalize square matrix dimensions."""
    m, n = _normalize_matrix_dim(dim)
    if m != n:
        raise ValueError(f"Expected a square matrix shape, got {(m, n)}.")
    return m


# region initializations ---------------------------------------------------------------
def gaussian(
    size: int | tuple[int, ...],
    dim: int | tuple[int, ...],
    *,
    loc: float = 0.0,
    scale: float = 1.0,
    dtype: Optional[torch.dtype] = None,
    device: Optional[str | torch.device] = None,
) -> Tensor:
    r"""Sample a random Gaussian tensor.

    `size` is interpreted as sample shape and `dim` as the event shape.
    The standard deviation is normalized by the last event axis when present.
    """
    batch = _normalize_sample_shape(size)
    event = (dim,) if isinstance(dim, int) else tuple(dim)
    shape = (*batch, *event)
    std = scale if len(event) == 0 else scale / sqrt(event[-1])
    mean = torch.full(shape, loc, dtype=dtype, device=device)
    return torch.normal(mean=mean, std=std)


def diagonally_dominant(
    size: int | tuple[int, ...],
    dim: int | tuple[int, int],
    *,
    dtype: Optional[torch.dtype] = None,
    device: Optional[str | torch.device] = None,
) -> Tensor:
    r"""Sample a random diagonally dominant matrix.

    We sample a random traceless matrix $B_{ij}∼𝓝(0,1/n)$, $B_{ii}=0$ and
    then consider a linear combination with the identity matrix $𝕀ₙ$.
    We choose the coefficients such that the resulting matrix is diagonally dominant.

    .. math:: A = 𝕀ₙ + B  \qq{with}  B_{ij}∼𝓝(0,1/n²)

    Normalized such that if $x∼𝓝(0,1)$, then $A⋅x∼𝓝(0,1)$.

    `size` is interpreted as sample shape and `dim` as the matrix dimension.
    """
    n = _normalize_square_dim(dim)
    B = traceless(size, dim=n, dtype=dtype, device=device)
    # calculate 1-norm of B
    eye = torch.eye(n, dtype=dtype, device=device)
    # for diagonal dominance, we need to multiply
    return eye + B / B.abs().sum(dim=(-2, -1), keepdim=True)


def symmetric(
    size: int | tuple[int, ...],
    dim: int | tuple[int, int],
    *,
    dtype: Optional[torch.dtype] = None,
    device: Optional[str | torch.device] = None,
) -> Tensor:
    r"""Sample a symmetric matrix, i.e. $Aᵀ = A$.

    Normalized such that if $x∼𝓝(0,1)$, then $A⋅x∼𝓝(0,1)$.

    There are two common ways to sample a symmetric matrix:

    1. Copy the upper triangular part to the lower triangular part.
    2. Use the projection formula $(A+Aᵀ)/2$ with additional normalization.
       The variance of off-diagonal elements is then $2σ²/4$, so we need to scale by $√2$.
       The variance of diagonal elements is $4σ²/4$, so we don't need to scale.
    """
    batch = _normalize_sample_shape(size)
    n = _normalize_square_dim(dim)
    shape = (*batch, n, n)
    mean = torch.zeros(shape, dtype=dtype, device=device)
    A = torch.normal(mean=mean, std=1 / sqrt(n))
    return A.triu() + A.triu(1).swapaxes(-2, -1)


def skew_symmetric(
    size: int | tuple[int, ...],
    dim: int | tuple[int, int],
    *,
    dtype: Optional[torch.dtype] = None,
    device: Optional[str | torch.device] = None,
) -> Tensor:
    r"""Sample a random skew-symmetric matrix, i.e. $Aᵀ = -A$.

    Normalized such that if $x∼𝓝(0,1)$, then $A⋅x∼𝓝(0,1)$.
    """
    batch = _normalize_sample_shape(size)
    n = _normalize_square_dim(dim)
    shape = (*batch, n, n)
    mean = torch.zeros(shape, dtype=dtype, device=device)
    A = torch.normal(mean=mean, std=1 / sqrt(n))
    return A.triu() - A.triu().swapaxes(-2, -1)


def orthogonal(
    size: int | tuple[int, ...],
    dim: int | tuple[int, int],
    *,
    dtype: Optional[torch.dtype] = None,
    device: Optional[str | torch.device] = None,
) -> Tensor:
    r"""Sample a random matrix with orthonormal columns.

    Normalized such that if $x∼𝓝(0,1)$, then $A⋅x∼𝓝(0,1)$.

    We sample a Gaussian matrix and take its QR factorization, then fix the signs
    using the diagonal of $R$ so the result matches the standard Haar sampler.
    """
    batch = _normalize_sample_shape(size)
    m, n = _normalize_matrix_dim(dim)
    if m < n:
        raise ValueError(f"Expected a tall matrix shape with m >= n, got {(m, n)}.")

    shape = (*batch, m, n)
    A = torch.randn(shape, dtype=dtype, device=device)
    # QR of a Gaussian matrix gives an orthogonal factor with the correct law
    # up to independent sign flips encoded in diag(R).
    Q, R = torch.linalg.qr(A)
    d = torch.diagonal(R, dim1=-2, dim2=-1)
    # Flip the columns of Q so diag(R) is positive, matching SciPy's construction.
    signs = torch.where(d == 0, torch.ones_like(d), d.sign())
    return Q * signs.unsqueeze(-2)


def special_orthogonal(
    size: int | tuple[int, ...],
    dim: int | tuple[int, int],
    *,
    dtype: Optional[torch.dtype] = None,
    device: Optional[str | torch.device] = None,
) -> Tensor:
    r"""Sample a random special orthogonal matrix, i.e. $Aᵀ = A⁻¹$ with $\det(A)=1$.

    Normalized such that if $x∼𝓝(0,1)$, then $A⋅x∼𝓝(0,1)$.

    We first sample from O(n) with the QR-based Haar sampler above, then flip the
    first col when the determinant is negative to project the sample into SO(n).
    """
    n = _normalize_square_dim(dim)
    Q = orthogonal(size, dim=n, dtype=dtype, device=device)
    if n == 0:
        return Q

    # Orthogonal matrices have determinant close to ±1. Flipping one column when
    # det(Q) is negative preserves orthogonality and forces the determinant to +1.
    dets = torch.linalg.det(Q).unsqueeze(-1)  # (..., 1)
    q = Q[..., 0]  # (..., d)
    q = torch.where(dets > 0, q, -q)
    return torch.cat([q.unsqueeze(-1), Q[..., 1:]], dim=-1)


def low_rank(
    size: int | tuple[int, ...],
    dim: int | tuple[int, int],
    *,
    rank: int,
    dtype: Optional[torch.dtype] = None,
    device: Optional[str | torch.device] = None,
) -> Tensor:
    r"""Sample a random low-rank m×n matrix, i.e. $A = UVᵀ$.

    Normalized such that if $x∼𝓝(0,1)$, then $A⋅x∼𝓝(0,1)$.
    """
    batch = _normalize_sample_shape(size)
    m, n = _normalize_matrix_dim(dim)

    if rank > min(m, n):
        raise ValueError("Rank must be smaller than min(m,n)")

    mean_u = torch.zeros((*batch, m, rank), dtype=dtype, device=device)
    mean_v = torch.zeros((*batch, rank, n), dtype=dtype, device=device)
    U = torch.normal(mean=mean_u, std=1 / sqrt(rank))
    V = torch.normal(mean=mean_v, std=1 / sqrt(n))
    return torch.einsum("...ij, ...jk -> ...ik", U, V)


def traceless(
    size: int | tuple[int, ...],
    dim: int | tuple[int, int],
    *,
    dtype: Optional[torch.dtype] = None,
    device: Optional[str | torch.device] = None,
) -> Tensor:
    r"""Sample a random traceless matrix, i.e. $\tr(A)=0$.

    Normalized such that if $x∼𝓝(0,1)$, then $A⋅x∼𝓝(0,1)$.
    """
    n = _normalize_square_dim(dim)
    A = gaussian(size, dim=(n, n), dtype=dtype, device=device)
    # FIXME: add normalization correction.
    eye = torch.eye(n, dtype=dtype, device=device)
    return A - torch.einsum("...ij, ij -> ...ij", A, eye)


# region canonical (deterministic) initializations -------------------------------------
def symplectic(
    size: int | tuple[int, ...],
    dim: int | tuple[int, int],
    *,
    device: Optional[str | torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> Tensor:
    r"""Return the canonical symplectic matrix of size $n=2k$.

    .. math:: 𝕊_n = \begin{bmatrix}0 & 𝕀_k \\ -𝕀_k & 0\end{bmatrix}

    Normalized such that if $x∼𝓝(0,1)$, then $A⋅x∼𝓝(0,1)$.
    """
    batch = _normalize_sample_shape(size)
    n = _normalize_square_dim(dim)
    if n % 2 != 0:
        raise ValueError("The dimension must be divisible by 2!")

    # create J matrix
    J1 = torch.tensor([[0, 1], [-1, 0]], device=device, dtype=dtype)
    eye = torch.eye(n // 2, device=device, dtype=dtype)
    J = torch.kron(J1, eye)

    # duplicate J for batch-size
    ones = torch.ones(batch, device=device, dtype=dtype)
    return torch.einsum("..., de -> ...de", ones, J)


# endregion canonical (deterministic) initializations ----------------------------------
# endregion initializations ------------------------------------------------------------
