r"""
We create different initializations

all initializations are normalized such that if x~N(0,1), then Ax ~ N(0,1) as well

Note that since Var(x+y) = Var(x) + 2Cov(x, y) + Var(y)

"""

import torch
from torch import Tensor
from scipy import stats
from typing import Union
from math import sqrt, prod


def gaussian(n: Union[int, tuple[int]]) -> Tensor:
    r"""Samples a random gaussian matrix, i.e. $A_{ij}\sim \mathcal N(0, \tfrac{1}{n})$, of size $n$.

    Normalized such that if $x\sim \mathcal N(0,1)$, then $A\cdot x\sim\mathcal N(0,1)$

    Parameters
    ----------
    n: int or tuple[int]
        If :class:`tuple`, the last axis is interpreted as dimension and the others as batch
    Returns
    -------
    :class:`~torch.Tensor`
    """
    # convert to tuple
    TUPLE = (n,) if type(n) == int else tuple(n)
    DIM, SIZE = TUPLE[-1], TUPLE[:-1]
    SHAPE = (*SIZE, DIM, DIM)

    return torch.normal(mean=torch.zeros(SHAPE), std=1/sqrt(DIM))


def diagonally_dominant(n: Union[int, tuple[int]]) -> Tensor:
    r"""Samples a random diagonally dominant matrix, i.e. $A = I_n + B$,
    with $B_{ij}\sim \mathcal N(0, \tfrac{1}{n^2})$, of size $n$.

    Normalized such that if $x\sim \mathcal N(0,1)$, then $A\cdot x\sim\mathcal N(0,1)$

    Parameters
    ----------
    n: int or tuple[int]
        If :class:`tuple`, the last axis is interpreted as dimension and the others as batch
    Returns
    -------
    :class:`~torch.Tensor`
    """
    # convert to tuple
    TUPLE = (n,) if type(n) == int else tuple(n)
    DIM, SIZE = TUPLE[-1], TUPLE[:-1]
    SHAPE = (*SIZE, DIM, DIM)

    return torch.eye(DIM) + torch.normal(mean=torch.zeros(SHAPE), std=1/DIM)


def symmetric(n: Union[int, tuple[int]]) -> Tensor:
    r"""Samples a symmetric matrix, i.e. $A^T = A$, of size $n$.

    Normalized such that if $x\sim \mathcal N(0,1)$, then $A\cdot x\sim\mathcal N(0,1)$

    Parameters
    ----------
    n: int or tuple[int]

    Returns
    -------
    :class:`~torch.Tensor`
    """
    # convert to tuple
    TUPLE = (n,) if type(n) == int else tuple(n)
    DIM, SIZE = TUPLE[-1], TUPLE[:-1]
    SHAPE = (*SIZE, DIM, DIM)

    A = torch.normal(mean=torch.zeros(SHAPE), std=1/sqrt(DIM))
    return (A + A.swapaxes(-1, -2))/sqrt(2)


def skew_symmetric(n: Union[int, tuple[int]]) -> Tensor:
    r"""Samples a random skew-symmetric matrix, i.e. $A^T = -A$, of size $n$.

    Normalized such that if $x\sim \mathcal N(0,1)$, then $A\cdot x\sim\mathcal N(0,1)$

    Parameters
    ----------
    n: int or tuple[int]

    Returns
    -------
    :class:`~torch.Tensor`
    """
    # convert to tuple
    # convert to tuple
    TUPLE = (n,) if type(n) == int else tuple(n)
    DIM, SIZE = TUPLE[-1], TUPLE[:-1]
    SHAPE = (*SIZE, DIM, DIM)

    A = torch.normal(mean=torch.zeros(SHAPE), std=1/sqrt(DIM))
    return (A - A.swapaxes(-1, -2))/sqrt(2)


def orthogonal(n: Union[int, tuple[int]]) -> Tensor:
    r"""
    Samples a random orthogonal matrix, i.e. $A^T = A$, of size $n$.

    Normalized such that if $x\sim \mathcal N(0,1)$, then $A\cdot x\sim\mathcal N(0,1)$

    Parameters
    ----------
    n: int or tuple[int]

    Returns
    -------
    :class:`~torch.Tensor`
    """
    # convert to tuple
    TUPLE = (n,) if type(n) == int else tuple(n)
    DIM, SIZE = TUPLE[-1], TUPLE[:-1]
    NUM = prod(SIZE)
    SHAPE = (*SIZE, DIM, DIM)

    A = stats.ortho_group.rvs(dim=DIM, size=NUM).reshape(SHAPE)
    return torch.Tensor(A)


def special_orthogonal(n: Union[int, tuple[int]]) -> Tensor:
    r"""Samples a random special orthogonal matrix, i.e. $A^T = A^{-1}$ with $\det(A)=1$, of size $n$.

    Normalized such that if $x\sim \mathcal N(0,1)$, then $A\cdot x\sim\mathcal N(0,1)$

    Parameters
    ----------
    n: int

    Returns
    -------
    :class:`~torch.Tensor`
    """

    # convert to tuple
    TUPLE = (n,) if type(n) == int else tuple(n)
    DIM, SIZE = TUPLE[-1], TUPLE[:-1]
    NUM = prod(SIZE)
    SHAPE = (*SIZE, DIM, DIM)

    A = stats.special_ortho_group.rvs(dim=DIM, size=NUM).reshape(SHAPE)
    return torch.Tensor(A)
