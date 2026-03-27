r"""Initializations for the Linear ODE Networks.

All initializations are normalized such that if $x∼𝓝(0,1)$, then $Ax∼𝓝(0,1)$ as well.

Notes:
    - See `linodenet.initializations.functional` for functional implementations.
    - See `linodenet.initializations.modules` for all module-based initializations.
"""

__all__ = [
    # ABCs & Protocols
    "Initialization",
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

from collections.abc import Sequence
from math import sqrt
from typing import Optional, Protocol, runtime_checkable

import torch
from torch import Tensor


@runtime_checkable
class Initialization(Protocol):
    r"""Protocol for Initializations."""

    def __call__(
        self,
        size: int | tuple[int, ...],
        /,
        *,
        # TODO: Add `generator` argument to all initializations.
        # generator: Optional[Generator] = None,
        dtype: Optional[torch.dtype] = None,
        device: Optional[str | torch.device] = None,
    ) -> Tensor:
        r"""Create a random matrix of shape `n`."""
        ...


# region initializations ---------------------------------------------------------------
def gaussian(
    size: int | tuple[int, ...],
    loc: float = 0.0,
    scale: float = 1.0,
    *,
    dtype: Optional[torch.dtype] = None,
    device: Optional[str | torch.device] = None,
) -> Tensor:
    r"""Sample a random gaussian matrix, i.e. $A_{ij}∼𝓝(0,1/n)$.

    Normalized such that if $x∼𝓝(0,1)$, then $A⋅x∼𝓝(0,1)$ if $σ=1$.

    If n is `tuple`, the last axis is interpreted as dimension and the others as batch.
    """
    tup = (size,) if isinstance(size, int) else tuple(size)
    batch, dim = tup[:-1], tup[-1]
    shape = (*batch, dim, dim)
    mean = torch.full(shape, loc, dtype=dtype, device=device)
    return torch.normal(mean=mean, std=scale / sqrt(dim))


def diagonally_dominant(
    size: int | tuple[int, ...],
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

    If n is `tuple`, the last axis is interpreted as dimension and the others as batch.
    """
    dim = size[-1] if isinstance(size, tuple) else size
    B = traceless(size, dtype=dtype, device=device)
    # calculate 1-norm of B
    eye = torch.eye(dim, dtype=dtype, device=device)
    # for diagonal dominance, we need to multiply
    return eye + B / B.abs().sum(dim=(-2, -1), keepdim=True)


def symmetric(
    size: int | tuple[int, ...],
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
    # convert to tuple
    tup = (size,) if isinstance(size, int) else tuple(size)
    batch, dim = tup[:-1], tup[-1]
    shape = (*batch, dim, dim)
    mean = torch.zeros(shape, dtype=dtype, device=device)
    A = torch.normal(mean=mean, std=1 / sqrt(dim))
    return A.triu() + A.triu(1).swapaxes(-2, -1)


def skew_symmetric(
    size: int | tuple[int, ...],
    *,
    dtype: Optional[torch.dtype] = None,
    device: Optional[str | torch.device] = None,
) -> Tensor:
    r"""Sample a random skew-symmetric matrix, i.e. $Aᵀ = -A$.

    Normalized such that if $x∼𝓝(0,1)$, then $A⋅x∼𝓝(0,1)$.
    """
    # convert to tuple
    tup = (size,) if isinstance(size, int) else tuple(size)
    batch, dim = tup[:-1], tup[-1]
    shape = (*batch, dim, dim)
    mean = torch.zeros(shape, dtype=dtype, device=device)
    A = torch.normal(mean=mean, std=1 / sqrt(dim))
    return A.triu() - A.triu().swapaxes(-2, -1)


def orthogonal(
    size: int | tuple[int, ...],
    *,
    dtype: Optional[torch.dtype] = None,
    device: Optional[str | torch.device] = None,
) -> Tensor:
    r"""Sample a random orthogonal matrix, i.e. $Aᵀ = A$.

    Normalized such that if $x∼𝓝(0,1)$, then $A⋅x∼𝓝(0,1)$.

    We sample a Gaussian matrix and take its QR factorization, then fix the signs
    using the diagonal of $R$ so the result matches the standard Haar sampler.
    """
    # convert to tuple
    tup = (size,) if isinstance(size, int) else tuple(size)
    batch, dim = tup[:-1], tup[-1]

    shape = (*batch, dim, dim)
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
    *,
    dtype: Optional[torch.dtype] = None,
    device: Optional[str | torch.device] = None,
) -> Tensor:
    r"""Sample a random special orthogonal matrix, i.e. $Aᵀ = A⁻¹$ with $\det(A)=1$.

    Normalized such that if $x∼𝓝(0,1)$, then $A⋅x∼𝓝(0,1)$.

    We first sample from O(n) with the QR-based Haar sampler above, then flip the
    first col when the determinant is negative to project the sample into SO(n).
    """
    Q = orthogonal(size, dtype=dtype, device=device)
    dim = Q.shape[-1]
    if dim == 0:
        return Q

    # Orthogonal matrices have determinant close to ±1. Flipping one column when
    # det(Q) is negative preserves orthogonality and forces the determinant to +1.
    dets = torch.linalg.det(Q).unsqueeze(-1)  # (..., 1)
    q = Q[..., 0]  # (..., d)
    q = torch.where(dets > 0, q, -q)
    return torch.cat([q.unsqueeze(-1), Q[..., 1:]], dim=-1)


def low_rank(
    size: int | tuple[int, ...],
    rank: int = 1,
    *,
    dtype: Optional[torch.dtype] = None,
    device: Optional[str | torch.device] = None,
) -> Tensor:
    r"""Sample a random low-rank m×n matrix, i.e. $A = UVᵀ$.

    Normalized such that if $x∼𝓝(0,1)$, then $A⋅x∼𝓝(0,1)$.
    """
    if isinstance(size, int):
        shape: tuple[int, ...] = (size, size)
    elif isinstance(size, Sequence) and len(size) == 1:
        shape = (size[0], size[0])
    else:
        shape = size

    *batch, m, n = shape

    if rank > min(m, n):
        raise ValueError("Rank must be smaller than min(m,n)")

    mean_u = torch.zeros((*batch, m, rank), dtype=dtype, device=device)
    mean_v = torch.zeros((*batch, rank, n), dtype=dtype, device=device)
    U = torch.normal(mean=mean_u, std=1 / sqrt(rank))
    V = torch.normal(mean=mean_v, std=1 / sqrt(n))
    return torch.einsum("...ij, ...jk -> ...ik", U, V)


def traceless(
    size: int | tuple[int, ...],
    *,
    dtype: Optional[torch.dtype] = None,
    device: Optional[str | torch.device] = None,
) -> Tensor:
    r"""Sample a random traceless matrix, i.e. $\tr(A)=0$.

    Normalized such that if $x∼𝓝(0,1)$, then $A⋅x∼𝓝(0,1)$.
    """
    # convert to tuple
    tup = (size,) if isinstance(size, int) else tuple(size)
    A = gaussian(size, dtype=dtype, device=device)
    # FIXME: add normalization correction.
    eye = torch.eye(tup[-1], dtype=dtype, device=device)
    return A - torch.einsum("...ij, ij -> ...ij", A, eye)


# region canonical (deterministic) initializations -------------------------------------
def symplectic(
    size: int | tuple[int, ...],
    *,
    device: Optional[str | torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> Tensor:
    r"""Return the canonical symplectic matrix of size $n=2k$.

    .. math:: 𝕊_n = \begin{bmatrix}0 & 𝕀_k \\ -𝕀_k & 0\end{bmatrix}

    Normalized such that if $x∼𝓝(0,1)$, then $A⋅x∼𝓝(0,1)$.
    """
    # convert to tuple
    tup = (size,) if isinstance(size, int) else tuple(size)
    batch, dim = tup[:-1], tup[-1]
    if dim % 2 != 0:
        raise ValueError("The dimension must be divisible by 2!")

    # create J matrix
    J1 = torch.tensor([[0, 1], [-1, 0]], device=device, dtype=dtype)
    eye = torch.eye(dim // 2, device=device, dtype=dtype)
    J = torch.kron(J1, eye)

    # duplicate J for batch-size
    ones = torch.ones(batch, device=device, dtype=dtype)
    return torch.einsum("..., de -> ...de", ones, J)


# endregion canonical (deterministic) initializations ----------------------------------
# endregion initializations ------------------------------------------------------------
