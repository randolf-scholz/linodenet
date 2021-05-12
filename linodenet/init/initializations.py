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

    Parameters
    ----------
    n: int

    Returns
    -------
    :class:`torch.Tensor`
    """
    A = torch.normal(mean=torch.zeros(n, n), std=1/np.sqrt(n))
    return (A - A.T)/np.sqrt(2)


def orthogonal(n: int) -> Tensor:
    r"""

    Parameters
    ----------
    n: int

    Returns
    -------
    :class:`torch.Tensor`
    """
    A = stats.ortho_group.rvs(dim=n)
    return torch.Tensor(A)


def special_orthogonal(n: int) -> Tensor:
    r"""

    Parameters
    ----------
    n: int

    Returns
    -------
    :class:`torch.Tensor`
    """
    A = stats.special_ortho_group.rvs(dim=n)
    return torch.Tensor(A)
