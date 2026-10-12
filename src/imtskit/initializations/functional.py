r"""Functional matrix initializations."""

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
    dim: int | tuple[int, int],
    *,
    loc: float = 0.0,
    scale: float = 1.0,
    dtype: Optional[torch.dtype] = None,
    device: Optional[str | torch.device] = None,
) -> Tensor:
    r"""Sample a random Gaussian matrix.

    Entries are independent with mean `loc` and standard deviation $σ/√n$,
    where $σ$ is `scale` and $n$ is the number of columns.
    """
    batch = _normalize_sample_shape(size)
    m, n = _normalize_matrix_dim(dim)
    shape = (*batch, m, n)
    std = scale / sqrt(n)
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

    $$  A = 𝕀ₙ + B/s, \qquad s = ∑ᵢⱼ |Bᵢⱼ|  $$

    Sample $B$ with $Bᵢᵢ=0$ and independent off-diagonal entries
    $Bᵢⱼ∼𝓝(0,1/n)$. Normalize by the sum of absolute entries $s$:

    For $s>0$, diagonal entries equal one and each off-diagonal row sum is at
    most one, giving weak diagonal dominance. If $s=0$, division by zero
    produces non-finite entries; this always occurs for $n=1$.
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

    $A$ is constructed by sampling an upper triangular matrix $𝓝(0,1/n)$
    and copying the entries to their corresponding lower tringular entry.
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

    Entries strictly above the diagonal are independent samples from $𝓝(0,1/n)$.
    They are negated and copied below the diagonal; diagonal entries are zero.
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

    Sample an $m×n$ Gaussian matrix, take its QR factorization, and fix the
    column signs using the diagonal of $R$. The result is uniform over matrices
    with orthonormal columns and satisfies $AᵀA=𝕀ₙ$.

    Requires $m≥n$. For square matrices, this is the Haar distribution on $O(n)$.
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
    signs = torch.where(d == 0, 1.0, d.sign())
    return Q * signs.unsqueeze(-2)


def special_orthogonal(
    size: int | tuple[int, ...],
    dim: int | tuple[int, int],
    *,
    dtype: Optional[torch.dtype] = None,
    device: Optional[str | torch.device] = None,
) -> Tensor:
    r"""Sample a special orthogonal matrix with $Aᵀ=A⁻¹$ and $\det(A)=1$.

    Sample from the Haar distribution on $O(n)$ using `orthogonal`, then flip
    the first column when the determinant is negative. This gives the Haar
    distribution on $SO(n)$.

    For $n=0$, return the empty $0×0$ matrix without a column flip.
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
    r"""Sample a random low-rank $m×n$ matrix as $A=UV$.

    With $r≤\min(m,n)$ equal to `rank`, sample independent factors $U∈ℝ^{m×r}$ and $V∈ℝ^{r×n}$
    with entries distributed as $𝓝(0,1/r)$ and $𝓝(0,1/n)$, respectively.
    The product has rank at most $r$ and each entry has variance $1/n$.
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
    r"""Sample a random matrix with zero diagonal and hence $\tr(A)=0$.

    Sample a Gaussian matrix with entry variance $1/n$, then set its diagonal
    to zero. Off-diagonal entries remain independent samples from $𝓝(0,1/n)$.
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

    $$ 𝕊ₙ = \begin{bmatrix}0 & 𝕀ₖ \\ -𝕀ₖ & 0\end{bmatrix} $$

    The result satisfies $𝕊ₙᵀ𝕊ₙ=𝕀ₙ$ and $𝕊ₙ²=-𝕀ₙ$, and preserves the
    standard Gaussian distribution. Odd dimensions are rejected.
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
