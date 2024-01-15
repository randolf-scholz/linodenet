r"""Initializations for the Linear ODE Networks.

All initializations are normalized such that if $x∼𝓝(0,1)$, then $Ax∼𝓝(0,1)$ as well.

Notes:
    Contains initializations in functional form.
"""

__all__ = [
    # Protocol
    "Initialization",
    # Functions
    "diagonally_dominant",
    "gaussian",
    "low_rank",
    "orthogonal",
    "skew_symmetric",
    "special_orthogonal",
    "symmetric",
    "traceless",
    # Non-deterministic
    "canonical_skew_symmetric",
    "canonical_symplectic",
]

from collections.abc import Sequence
from math import prod, sqrt
from typing import Optional, Protocol, runtime_checkable

import torch
from numpy.typing import NDArray
from scipy import stats
from torch import Tensor

from linodenet.types import Device, Dtype, Shape


@runtime_checkable
class Initialization(Protocol):
    r"""Protocol for Initializations."""

    __name__: str
    r"""Name of the initialization."""

    def __call__(
        self, n: Shape, /, *, dtype: Dtype = None, device: Device = None
    ) -> Tensor:
        """Create a random matrix of shape `n`."""
        ...


# region initializations ---------------------------------------------------------------
def gaussian(
    n: Shape,
    /,
    *,
    loc: float = 0.0,
    scale: float = 1.0,
    dtype: Dtype = None,
    device: Device = None,
) -> Tensor:
    r"""Sample a random gaussian matrix, i.e. $A_{ij}∼𝓝(0,1/n)$.

    Normalized such that if $x∼𝓝(0,1)$, then $A⋅x∼𝓝(0,1)$ if $σ=1$.

    If n is `tuple`, the last axis is interpreted as dimension and the others as batch.
    """
    # convert to tuple
    tup = (n,) if isinstance(n, int) else tuple(n)
    size, dim = tup[:-1], tup[-1]
    shape = (*size, dim, dim)
    mean = torch.full(shape, loc, dtype=dtype, device=device)
    return torch.normal(mean=mean, std=scale / sqrt(dim))


def diagonally_dominant(n: Shape, dtype: Dtype = None, device: Device = None) -> Tensor:
    r"""Sample a random diagonally dominant matrix, i.e. $A = 𝕀_n + B$,with $B_{ij}∼𝓝(0,1/n²)$.

    Normalized such that if $x∼𝓝(0,1)$, then $A⋅x∼𝓝(0,1)$.

    If n is `tuple`, the last axis is interpreted as dimension and the others as batch.
    """
    # convert to tuple
    tup = (n,) if isinstance(n, int) else tuple(n)
    size, dim = tup[:-1], tup[-1]
    shape = (*size, dim, dim)
    eye = torch.eye(dim, dtype=dtype, device=device)
    mean = torch.zeros(shape, dtype=dtype, device=device)
    return eye + torch.normal(mean=mean, std=1 / dim)


def symmetric(n: Shape, dtype: Dtype = None, device: Device = None) -> Tensor:
    r"""Sample a symmetric matrix, i.e. $A^⊤ = A$.

    Normalized such that if $x∼𝓝(0,1)$, then $A⋅x∼𝓝(0,1)$.
    """
    # convert to tuple
    tup = (n,) if isinstance(n, int) else tuple(n)
    size, dim = tup[:-1], tup[-1]
    shape = (*size, dim, dim)
    mean = torch.zeros(shape, dtype=dtype, device=device)
    A = torch.normal(mean=mean, std=1 / sqrt(dim))
    return (A + A.swapaxes(-1, -2)) / sqrt(2)


def skew_symmetric(n: Shape, dtype: Dtype = None, device: Device = None) -> Tensor:
    r"""Sample a random skew-symmetric matrix, i.e. $A^⊤ = -A$.

    Normalized such that if $x∼𝓝(0,1)$, then $A⋅x∼𝓝(0,1)$.
    """
    # convert to tuple
    # convert to tuple
    tup = (n,) if isinstance(n, int) else tuple(n)
    size, dim = tup[:-1], tup[-1]
    shape = (*size, dim, dim)
    mean = torch.zeros(shape, dtype=dtype, device=device)
    A = torch.normal(mean=mean, std=1 / sqrt(dim))
    return (A - A.swapaxes(-1, -2)) / sqrt(2)


def orthogonal(n: Shape, dtype: Dtype = None, device: Device = None) -> Tensor:
    r"""Sample a random orthogonal matrix, i.e. $A^⊤ = A$.

    Normalized such that if $x∼𝓝(0,1)$, then $A⋅x∼𝓝(0,1)$.
    """
    # convert to tuple
    tup = (n,) if isinstance(n, int) else tuple(n)
    size, dim = tup[:-1], tup[-1]
    num = prod(size)
    shape = (*size, dim, dim)

    A: NDArray = stats.ortho_group.rvs(dim=dim, size=num).reshape(shape)
    if dtype is None:
        dtype = torch.float32
    return torch.from_numpy(A).to(dtype=dtype, device=device)


def special_orthogonal(n: Shape, dtype: Dtype = None, device: Device = None) -> Tensor:
    r"""Sample a random special orthogonal matrix, i.e. $A^⊤ = A^{-1}$ with $\det(A)=1$.

    Normalized such that if $x∼𝓝(0,1)$, then $A⋅x∼𝓝(0,1)$.
    """
    # convert to tuple
    tup = (n,) if isinstance(n, int) else tuple(n)
    size, dim = tup[:-1], tup[-1]
    num = prod(size)
    shape = (*size, dim, dim)

    A: NDArray = stats.special_ortho_group.rvs(dim=dim, size=num).reshape(shape)
    if dtype is None:
        dtype = torch.float32
    return torch.from_numpy(A).to(dtype=dtype, device=device)


def low_rank(
    size: Shape,
    *,
    rank: Optional[int] = None,
    dtype: Dtype = None,
    device: Device = None,
) -> Tensor:
    r"""Sample a random low-rank m×n matrix, i.e. $A = UV^⊤$."""
    if isinstance(size, int):
        shape: tuple[int, ...] = (size, size)
    elif isinstance(size, Sequence) and len(size) == 1:
        shape = (size[0], size[0])
    else:
        shape = size

    *batch, m, n = shape

    if isinstance(rank, int) and rank > min(m, n):
        raise ValueError("Rank must be smaller than min(m,n)")

    rank = max(1, min(m, n) // 2) if rank is None else rank
    mean_u = torch.zeros((*batch, m, rank), dtype=dtype, device=device)
    mean_v = torch.zeros((*batch, rank, n), dtype=dtype, device=device)
    U = torch.normal(mean=mean_u, std=1 / sqrt(rank))
    V = torch.normal(mean=mean_v, std=1 / sqrt(n))
    return torch.einsum("...ij, ...jk -> ...ik", U, V)


def traceless(n: Shape, dtype: Dtype = None, device: Device = None) -> Tensor:
    r"""Sample a random traceless matrix, i.e. $\tr(A)=0$."""
    # convert to tuple
    tup = (n,) if isinstance(n, int) else tuple(n)
    A = gaussian(n, dtype=dtype, device=device)
    eye = torch.eye(tup[-1], dtype=dtype, device=device)
    return A - torch.einsum("...ij, ij -> ...ij", A, eye)


# region canonical (non-deterministic) initializations ---------------------------------
def canonical_skew_symmetric(
    n: Shape, device: Device = None, dtype: Dtype = None
) -> Tensor:
    r"""Return the canonical skew symmetric matrix of size $n=2k$.

    .. math:: 𝕁_n = 𝕀_n ⊗ \begin{bmatrix}0 & +1 \\ -1 & 0\end{bmatrix}

    Normalized such that if $x∼𝓝(0,1)$, then $A⋅x∼𝓝(0,1)$.
    """
    # convert to tuple
    tup = (n,) if isinstance(n, int) else tuple(n)
    size, dim = tup[:-1], tup[-1]
    assert dim % 2 == 0, "The dimension must be divisible by 2!"

    # create J matrix
    J1 = torch.tensor([[0, 1], [-1, 0]], device=device, dtype=dtype)
    eye = torch.eye(dim // 2, device=device, dtype=dtype)
    J = torch.kron(J1, eye)

    # duplicate J for batch-size
    ones = torch.ones(size, device=device, dtype=dtype)
    return torch.einsum("..., de -> ...de", ones, J)


def canonical_symplectic(
    n: Shape, device: Device = None, dtype: Dtype = None
) -> Tensor:
    r"""Return the canonical symplectic matrix of size $n=2k$.

    .. math:: 𝕊_n = \begin{bmatrix}0 & 𝕀_k \\ -𝕀_k & 0\end{bmatrix}

    Normalized such that if $x∼𝓝(0,1)$, then $A⋅x∼𝓝(0,1)$.

    Note:
        Alias for `canonical_skew_symmetric`.
    """
    return canonical_skew_symmetric(n, device, dtype)


# endregion canonical (non-deterministic) initializations ------------------------------

# endregion initializations ------------------------------------------------------------
