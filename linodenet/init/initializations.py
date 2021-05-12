r"""
We create different initializations

all initializations are normalized such that if x~N(0,1), then Ax ~ N(0,1) as well

Note that since Var(x+y) = Var(x) + 2Cov(x, y) + Var(y)

"""

import torch
from torch import Tensor
import numpy as np
from scipy import stats


def gaussian(n: int) -> Tensor:
    r"""
    Samples a random gaussian matrix $A_{ij}\sim \mathcal N(0, \tfrac{1}{n})$ of size $n$.

    Parameters
    ----------
    n: int

    Returns
    -------
    :class:`torch.Tensor`
    """
    return torch.normal(mean=torch.zeros(n, n), std=1/np.sqrt(n))


def symmetric(n: int) -> Tensor:
    r"""
    Samples a random orthogonal matrix $A^T = A$ of size $n$.

    Parameters
    ----------
    n: int

    Returns
    -------
    :class:`torch.Tensor`
    """
    A = torch.normal(mean=torch.zeros(n, n), std=1/np.sqrt(n))
    return (A + A.T)/np.sqrt(2)


def skew_symmetric(n: int) -> Tensor:
    r"""
    Samples a random skew-symmetric matrix, i.e. $A^T = -A$, of size $n$.

    Parameters
    ----------
    n:  tensor

    Returns
    -------
    output : iResNet
        return value
    """
    A = torch.normal(mean=torch.zeros(n, n), std=1/np.sqrt(n))
    return (A - A.T)/np.sqrt(2)


def orthogonal(n: int) -> Tensor:
    r"""Summary line.

    Samples a random orthogonal matrix, i.e. $A^T = A^{-1}$, of size $n$.

    Args:
        n (int): dimension

    Returns:
        tensor: random matrix
    """
    A = stats.ortho_group.rvs(dim=n)
    return torch.Tensor(A)


def special_orthogonal(n: int) -> Tensor:
    r"""
    Samples a random special orthogonal matrix, i.e. $A^T = A^{-1}$ with $\det(A)=1$, of size $n$.

    Parameters
    ----------
    n: int

    Returns
    -------
    :class:`torch.Tensor`
    """
    A = stats.special_ortho_group.rvs(dim=n)
    return torch.Tensor(A)
