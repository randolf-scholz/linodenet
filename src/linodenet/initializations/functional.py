r"""Initializations for the Linear ODE Networks.

All initializations are normalized such that if $x∼𝓝(0,1)$, then $Ax∼𝓝(0,1)$ as well.

Notes:
    Contains initializations in functional form.
"""

__all__ = [
    # ABCs & Protocols
    "Initialization",
    # Deterministic Initializations
    "canonical_skew_symmetric",
    "canonical_symplectic",
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
from math import prod, sqrt

import torch
from numpy.typing import NDArray
from scipy import stats
from torch import Tensor, device as Device, dtype as Dtype
from typing_extensions import Optional, Protocol, runtime_checkable


@runtime_checkable
class Initialization(Protocol):
    r"""Protocol for Initializations."""

    __name__: str
    r"""Name of the initialization."""

    def __call__(
        self,
        size: int | tuple[int, ...],
        /,
        *,
        # TODO: Add `generator` argument to all initializations.
        # generator: Optional[Generator] = None,
        dtype: Optional[Dtype] = None,
        device: Optional[str | Device] = None,
    ) -> Tensor:
        """Create a random matrix of shape `n`."""
        ...


# region initializations ---------------------------------------------------------------
def gaussian(
    size: int | tuple[int, ...],
    loc: float = 0.0,
    scale: float = 1.0,
    *,
    dtype: Optional[Dtype] = None,
    device: Optional[str | Device] = None,
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
    dtype: Optional[Dtype] = None,
    device: Optional[str | Device] = None,
) -> Tensor:
    r"""Sample a random diagonally dominant matrix.

    We sample a random traceless matrix $B_{ij}∼𝓝(0,1/n)$, $B_{ii}=0$ and
    then consider a linear combination with the identity matrix $𝕀ₙ$.
    We choose the coefficients such that the resulting matrix is diagonally dominant.

    .. math:: A = 𝕀ₙ + B  with  B_{ij}∼𝓝(0,1/n²)

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
    dtype: Optional[Dtype] = None,
    device: Optional[str | Device] = None,
) -> Tensor:
    r"""Sample a symmetric matrix, i.e. $A^⊤ = A$.

    Normalized such that if $x∼𝓝(0,1)$, then $A⋅x∼𝓝(0,1)$.

    There are two ways to generate:
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
    dtype: Optional[Dtype] = None,
    device: Optional[str | Device] = None,
) -> Tensor:
    r"""Sample a random skew-symmetric matrix, i.e. $A^⊤ = -A$.

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
    dtype: Optional[Dtype] = None,
    device: Optional[str | Device] = None,
) -> Tensor:
    r"""Sample a random orthogonal matrix, i.e. $A^⊤ = A$.

    Normalized such that if $x∼𝓝(0,1)$, then $A⋅x∼𝓝(0,1)$.
    """
    # convert to tuple
    tup = (size,) if isinstance(size, int) else tuple(size)
    batch, dim = tup[:-1], tup[-1]
    num = prod(batch)
    shape = (*batch, dim, dim)

    A: NDArray = stats.ortho_group.rvs(dim=dim, size=num).reshape(shape)
    if dtype is None:
        dtype = torch.float32
    return torch.from_numpy(A).to(dtype=dtype, device=device)


def special_orthogonal(
    size: int | tuple[int, ...],
    *,
    dtype: Optional[Dtype] = None,
    device: Optional[str | Device] = None,
) -> Tensor:
    r"""Sample a random special orthogonal matrix, i.e. $A^⊤ = A^{-1}$ with $\det(A)=1$.

    Normalized such that if $x∼𝓝(0,1)$, then $A⋅x∼𝓝(0,1)$.
    """
    # convert to tuple
    tup = (size,) if isinstance(size, int) else tuple(size)
    batch, dim = tup[:-1], tup[-1]
    num = prod(batch)
    shape = (*batch, dim, dim)

    A: NDArray = stats.special_ortho_group.rvs(dim=dim, size=num).reshape(shape)
    if dtype is None:
        dtype = torch.float32
    return torch.from_numpy(A).to(dtype=dtype, device=device)


def low_rank(
    size: int | tuple[int, ...],
    rank: int = 1,
    *,
    dtype: Optional[Dtype] = None,
    device: Optional[str | Device] = None,
) -> Tensor:
    r"""Sample a random low-rank m×n matrix, i.e. $A = UV^⊤$.

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
    dtype: Optional[Dtype] = None,
    device: Optional[str | Device] = None,
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
def canonical_skew_symmetric(
    size: int | tuple[int, ...],
    *,
    device: Optional[str | Device] = None,
    dtype: Optional[Dtype] = None,
) -> Tensor:
    r"""Return the canonical skew symmetric matrix of size $n=2k$.

    .. math:: 𝕁_n = 𝕀_n ⊗ \begin{bmatrix}0 & +1 \\ -1 & 0\end{bmatrix}

    Normalized such that if $x∼𝓝(0,1)$, then $A⋅x∼𝓝(0,1)$.
    """
    # convert to tuple
    tup = (size,) if isinstance(size, int) else tuple(size)
    batch, dim = tup[:-1], tup[-1]
    assert dim % 2 == 0, "The dimension must be divisible by 2!"

    # create J matrix
    J1 = torch.tensor([[0, 1], [-1, 0]], device=device, dtype=dtype)
    eye = torch.eye(dim // 2, device=device, dtype=dtype)
    J = torch.kron(J1, eye)

    # duplicate J for batch-size
    ones = torch.ones(batch, device=device, dtype=dtype)
    return torch.einsum("..., de -> ...de", ones, J)


def canonical_symplectic(
    size: int | tuple[int, ...],
    *,
    device: Optional[str | Device] = None,
    dtype: Optional[Dtype] = None,
) -> Tensor:
    r"""Return the canonical symplectic matrix of size $n=2k$.

    .. math:: 𝕊_n = \begin{bmatrix}0 & 𝕀_k \\ -𝕀_k & 0\end{bmatrix}

    Normalized such that if $x∼𝓝(0,1)$, then $A⋅x∼𝓝(0,1)$.

    Note:
        Alias for `canonical_skew_symmetric`.
    """
    return canonical_skew_symmetric(size, device=device, dtype=dtype)


# endregion canonical (deterministic) initializations ----------------------------------
# endregion initializations ------------------------------------------------------------
