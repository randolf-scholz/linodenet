r"""
We create different initializations

all initializations are normalized such that if X~N(0,1), then Ax ~ N(0,1) as well

Note that since Var(X+y) = Var(X) + 2Cov(X, y) + Var(y)
"""

from math import sqrt, prod
from typing import Union

import torch
from scipy import stats
from torch import Tensor


def gaussian(n: Union[int, tuple[int, ...]]) -> Tensor:
    r"""Samples a random gaussian matrix, i.e. $A_{ij}\sim \mathcal N(0,\tfrac{1}{n})$, of size $n$.

    Normalized such that if $X\sim \mathcal N(0,1)$, then $A\cdot X\sim\mathcal N(0,1)$

    Parameters
    ----------
    n: int or tuple[int]
        If :class:`tuple`, the last axis is interpreted as dimension and the others as batch
    Returns
    -------
    :class:`~torch.Tensor`
    """
    # convert to tuple
    tup = (n,) if isinstance(n, int) else tuple(n)
    dim, size = tup[-1], tup[:-1]
    shape = (*size, dim, dim)

    return torch.normal(mean=torch.zeros(shape), std=1/sqrt(dim))


def diagonally_dominant(n: Union[int, tuple[int, ...]]) -> Tensor:
    r"""Samples a random diagonally dominant matrix, i.e. $A = I_n + B$,
    with $B_{ij}\sim \mathcal N(0, \tfrac{1}{n^2})$, of size $n$.

    Normalized such that if $X\sim \mathcal N(0,1)$, then $A\cdot X\sim\mathcal N(0,1)$

    Parameters
    ----------
    n: int or tuple[int]
        If :class:`tuple`, the last axis is interpreted as dimension and the others as batch
    Returns
    -------
    :class:`~torch.Tensor`
    """
    # convert to tuple
    tup = (n,) if isinstance(n, int) else tuple(n)
    dim, size = tup[-1], tup[:-1]
    shape = (*size, dim, dim)

    return torch.eye(dim) + torch.normal(mean=torch.zeros(shape), std=1/dim)


def symmetric(n: Union[int, tuple[int, ...]]) -> Tensor:
    r"""Samples a symmetric matrix, i.e. $A^T = A$, of size $n$.

    Normalized such that if $X\sim \mathcal N(0,1)$, then $A\cdot X\sim\mathcal N(0,1)$

    Parameters
    ----------
    n: int or tuple[int]

    Returns
    -------
    :class:`~torch.Tensor`
    """
    # convert to tuple
    tup = (n,) if isinstance(n, int) else tuple(n)
    dim, size = tup[-1], tup[:-1]
    shape = (*size, dim, dim)

    A = torch.normal(mean=torch.zeros(shape), std=1/sqrt(dim))
    return (A + A.swapaxes(-1, -2))/sqrt(2)


def skew_symmetric(n: Union[int, tuple[int, ...]]) -> Tensor:
    r"""Samples a random skew-symmetric matrix, i.e. $A^T = -A$, of size $n$.

    Normalized such that if $X\sim \mathcal N(0,1)$, then $A\cdot X\sim\mathcal N(0,1)$

    Parameters
    ----------
    n: int or tuple[int]

    Returns
    -------
    :class:`~torch.Tensor`
    """
    # convert to tuple
    # convert to tuple
    tup = (n,) if isinstance(n, int) else tuple(n)
    dim, size = tup[-1], tup[:-1]
    shape = (*size, dim, dim)

    A = torch.normal(mean=torch.zeros(shape), std=1/sqrt(dim))
    return (A - A.swapaxes(-1, -2))/sqrt(2)


def orthogonal(n: Union[int, tuple[int, ...]]) -> Tensor:
    r"""
    Samples a random orthogonal matrix, i.e. $A^T = A$, of size $n$.

    Normalized such that if $X\sim \mathcal N(0,1)$, then $A\cdot X\sim\mathcal N(0,1)$

    Parameters
    ----------
    n: int or tuple[int]

    Returns
    -------
    :class:`~torch.Tensor`
    """
    # convert to tuple
    tup = (n,) if isinstance(n, int) else tuple(n)
    dim, size = tup[-1], tup[:-1]
    num = prod(size)
    shape = (*size, dim, dim)

    A = stats.ortho_group.rvs(dim=dim, size=num).reshape(shape)
    return torch.Tensor(A)


def special_orthogonal(n: Union[int, tuple[int, ...]]) -> Tensor:
    r"""Samples a random special orthogonal matrix, i.e. $A^T = A^{-1}$ with $\det(A)=1$,
    of size $n$.

    Normalized such that if $X\sim \mathcal N(0,1)$, then $A\cdot X\sim\mathcal N(0,1)$

    Parameters
    ----------
    n: int

    Returns
    -------
    :class:`~torch.Tensor`
    """

    # convert to tuple
    tup = (n,) if isinstance(n, int) else tuple(n)
    dim, size = tup[-1], tup[:-1]
    num = prod(size)
    shape = (*size, dim, dim)

    A = stats.special_ortho_group.rvs(dim=dim, size=num).reshape(shape)
    return torch.Tensor(A)
