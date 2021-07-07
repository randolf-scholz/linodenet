# TODO: remove scipy dependency once torch offers orthogonal matrix sampling
r"""Initializations for the Linear ODE Networks.

All initializations are normalized such that if $x~𝓝(0,1)$, then $Ax~𝓝(0,1)$ as well.
"""

from math import prod, sqrt
from typing import Union

from scipy import stats
import torch
from torch import Tensor


def gaussian(n: Union[int, tuple[int, ...]]) -> Tensor:
    r"""Sample a random gaussian matrix, i.e. $A_{ij}∼𝓝(0,1/n)$.

    Normalized such that if $x∼𝓝(0,1)$, then $A⋅x∼𝓝(0,1)$

    Parameters
    ----------
    n: int or tuple[int]
        If :class:`tuple`, the last axis is interpreted as dimension and the others as batch
    Returns
    -------
    Tensor
    """
    # convert to tuple
    tup = (n,) if isinstance(n, int) else tuple(n)
    dim, size = tup[-1], tup[:-1]
    shape = (*size, dim, dim)

    return torch.normal(mean=torch.zeros(shape), std=1 / sqrt(dim))


def diagonally_dominant(n: Union[int, tuple[int, ...]]) -> Tensor:
    r"""Sample a random diagonally dominant matrix, i.e. $A = I_n + B$,with $B_{ij}∼𝓝(0,1/n²)$.

    Normalized such that if $x∼𝓝(0,1)$, then $A⋅x∼𝓝(0,1)$

    Parameters
    ----------
    n: int or tuple[int]
        If :class:`tuple`, the last axis is interpreted as dimension and the others as batch
    Returns
    -------
    Tensor
    """
    # convert to tuple
    tup = (n,) if isinstance(n, int) else tuple(n)
    dim, size = tup[-1], tup[:-1]
    shape = (*size, dim, dim)

    return torch.eye(dim) + torch.normal(mean=torch.zeros(shape), std=1 / dim)


def symmetric(n: Union[int, tuple[int, ...]]) -> Tensor:
    r"""Sample a symmetric matrix, i.e. $A^⊤ = A$.

    Normalized such that if $x∼𝓝(0,1)$, then $A⋅x∼𝓝(0,1)$

    Parameters
    ----------
    n: int or tuple[int]

    Returns
    -------
    Tensor
    """
    # convert to tuple
    tup = (n,) if isinstance(n, int) else tuple(n)
    dim, size = tup[-1], tup[:-1]
    shape = (*size, dim, dim)

    A = torch.normal(mean=torch.zeros(shape), std=1 / sqrt(dim))
    return (A + A.swapaxes(-1, -2)) / sqrt(2)


def skew_symmetric(n: Union[int, tuple[int, ...]]) -> Tensor:
    r"""Sample a random skew-symmetric matrix, i.e. $A^⊤ = -A$.

    Normalized such that if $x∼𝓝(0,1)$, then $A⋅x∼𝓝(0,1)$

    Parameters
    ----------
    n: int or tuple[int]

    Returns
    -------
    Tensor
    """
    # convert to tuple
    # convert to tuple
    tup = (n,) if isinstance(n, int) else tuple(n)
    dim, size = tup[-1], tup[:-1]
    shape = (*size, dim, dim)

    A = torch.normal(mean=torch.zeros(shape), std=1 / sqrt(dim))
    return (A - A.swapaxes(-1, -2)) / sqrt(2)


def orthogonal(n: Union[int, tuple[int, ...]]) -> Tensor:
    r"""Sample a random orthogonal matrix, i.e. $A^⊤ = A$.

    Normalized such that if $x∼𝓝(0,1)$, then $A⋅x∼𝓝(0,1)$

    Parameters
    ----------
    n: int or tuple[int]

    Returns
    -------
    Tensor
    """
    # convert to tuple
    tup = (n,) if isinstance(n, int) else tuple(n)
    dim, size = tup[-1], tup[:-1]
    num = prod(size)
    shape = (*size, dim, dim)

    A = stats.ortho_group.rvs(dim=dim, size=num).reshape(shape)
    return torch.Tensor(A)


def special_orthogonal(n: Union[int, tuple[int, ...]]) -> Tensor:
    r"""Sample a random special orthogonal matrix, i.e. $A^⊤ = A^{-1}$ with $\det(A)=1$.

    Normalized such that if $x∼𝓝(0,1)$, then $A⋅x∼𝓝(0,1)$

    Parameters
    ----------
    n: int

    Returns
    -------
    Tensor
    """
    # convert to tuple
    tup = (n,) if isinstance(n, int) else tuple(n)
    dim, size = tup[-1], tup[:-1]
    num = prod(size)
    shape = (*size, dim, dim)

    A = stats.special_ortho_group.rvs(dim=dim, size=num).reshape(shape)
    return torch.Tensor(A)
