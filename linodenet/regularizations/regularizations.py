r"""Regularizations for LinODE kernel matrix.

Functional version.
"""

from __future__ import annotations

import logging
from typing import Final, Optional

import torch.linalg
from torch import Tensor, jit

from linodenet import projections

logger = logging.getLogger(__name__)
__all__: Final[list[str]] = [
    "logdetexp",
    "symmetric",
    "skew_symmetric",
    "diagonal",
    "orthogonal",
    "normal",
]


def logdetexp(x: Tensor, p: float = 1.0) -> Tensor:
    r"""Bias $\det(e^A)$ towards 1.

    By Jacobis formula

    .. math::

        \det(e^A) = e^{𝗍𝗋(A)} ⟺ \log(\det(e^A)) = 𝗍𝗋(A) ⟺ \log(\det(A)) = 𝗍𝗋(\log(A))

    In particular, we can regularize the LinODE model by adding a regularization term of the form

    .. math::

        |𝗍𝗋(A)|

    Signature
    ---------
    (..., n,n) ⟶ (...,)

    Parameters
    ----------
    x: Tensor
    p: float, default=1.0

    Returns
    -------
    Tensor
    """
    traces = torch.sum(torch.diagonal(x, dim1=-1, dim2=-2), dim=-1)
    return torch.abs(traces) ** p


@jit.script
def skew_symmetric(x: Tensor, p: Optional[float] = None) -> Tensor:
    r"""Bias the matrix towards being skew-symmetric.

    Signature
    ---------
    (..., n,n) ⟶ (...,)

    Parameters
    ----------
    x: Tensor
    p: Optional[float], If `None` uses Frobenius norm

    Returns
    -------
    Tensor
    """
    r = x - projections.skew_symmetric(x)
    if p is None:
        return torch.linalg.matrix_norm(r)
    return torch.linalg.matrix_norm(r, p)


@jit.script
def symmetric(x: Tensor, p: Optional[float] = None) -> Tensor:
    r"""Bias the matrix towards being symmetric.

    Signature
    ---------
    (..., n,n) ⟶ (...,)

    Parameters
    ----------
    x: Tensor
    p: Optional[float], If `None` uses Frobenius norm

    Returns
    -------
    Tensor
    """
    r = x - projections.symmetric(x)
    if p is None:
        return torch.linalg.matrix_norm(r)
    return torch.linalg.matrix_norm(r, p)


@jit.script
def orthogonal(x: Tensor, p: Optional[float] = None) -> Tensor:
    r"""Bias the matrix towards being orthogonal.

    Note that, given $n×n$ matrix $X$ with SVD $X=UΣVᵀ$ holds

    .. math::
          &(1) &  ‖ X - ΠX‖_F &= ‖ Σ - 𝕀 ‖_F
        \\&(1) &  ‖XᵀX - 𝕀‖_F &= ‖ΣᵀΣ - 𝕀‖_F
        \\&(1) &  ‖XXᵀ - X‖_F &= ‖ΣΣᵀ - 𝕀‖_F

    Signature
    ---------
    (..., n,n) ⟶ (...,)

    Parameters
    ----------
    x: Tensor
    p: Optional[float], If `None` uses Frobenius norm

    Returns
    -------
    Tensor
    """
    r = x - projections.orthogonal(x)
    if p is None:
        return torch.linalg.matrix_norm(r)
    return torch.linalg.matrix_norm(r, p)


@jit.script
def normal(x: Tensor, p: Optional[float] = None) -> Tensor:
    r"""Bias the matrix towards being normal.

    Signature
    ---------
    (..., n,n) ⟶ (...,)

    Parameters
    ----------
    x: Tensor
    p: Optional[float], If `None` uses Frobenius norm

    Returns
    -------
    Tensor
    """
    r = x - projections.normal(x)
    if p is None:
        return torch.linalg.matrix_norm(r)
    return torch.linalg.matrix_norm(r, p)


@jit.script
def diagonal(x: Tensor, p: Optional[float] = None) -> Tensor:
    r"""Bias the matrix towards being diagonal.

    Signature
    ---------
    (..., n,n) ⟶ (...,)

    Parameters
    ----------
    x: Tensor
    p: Optional[float], If `None` uses Frobenius norm

    Returns
    -------
    Tensor
    """
    r = x - projections.diagonal(x)
    if p is None:
        return torch.linalg.matrix_norm(r)
    return torch.linalg.matrix_norm(r, p)
