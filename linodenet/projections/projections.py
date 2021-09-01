r"""Projection Mappings."""
from __future__ import annotations

import logging
from typing import Final

import torch
from torch import Tensor, jit

logger = logging.getLogger(__name__)

__all__: Final[list[str]] = [
    "identity",
    "symmetric",
    "skew_symmetric",
    "orthogonal",
    "diagonal",
]


@jit.script
def identity(x: Tensor) -> Tensor:
    r"""Return x as-is.

    Parameters
    ----------
    x: Tensor

    Returns
    -------
    Tensor
    """
    return x


@jit.script
def symmetric(x: Tensor) -> Tensor:
    r"""Return the closest symmetric matrix to X.

    $$ ∥X-Y∥₂ s.t. Yᵀ = Y $$

    One can show analytically that Y = ½(X + Xᵀ) is the unique minimizer.

    Parameters
    ----------
    x: Tensor

    Returns
    -------
    Tensor
    """
    return (x + x.T) / 2


@jit.script
def skew_symmetric(x: Tensor) -> Tensor:
    r"""Return the closest skew-symmetric matrix to X.

    $$ \min ∥X-Y∥₂ s.t. Yᵀ = -Y $$

    One can show analytically that Y = ½(X - Xᵀ) is the unique minimizer.

    Parameters
    ----------
    x: Tensor

    Returns
    -------
    Tensor
    """
    return (x - x.T) / 2


@jit.script
def orthogonal(x: Tensor) -> Tensor:
    r"""Return the closest orthogonal matrix to X.

    $$ \min ∥X-Y∥₂ s.t. YᵀY = 𝕀 = YYᵀ $$

    One can show analytically that $Y = UVᵀ$ is the unique minimizer,
    where $X=UΣVᵀ$ is the SVD of $X$.

    References
    ----------
    - <https://math.stackexchange.com/q/2215359>_

    Parameters
    ----------
    x: Tensor

    Returns
    -------
    Tensor
    """
    U, _, V = torch.svd(x, some=False, compute_uv=True)
    return torch.einsum("...ij, ...kj -> ...ik", U, V)


@jit.script
def diagonal(x: Tensor) -> Tensor:
    r"""Return the closest diagonal matrix to X.

    $$ \min ∥X-Y∥₂ s.t. Y = 𝕀⊙Y $$

    One can show analytically that $Y = diag(X)$ is the unique minimizer.

    Parameters
    ----------
    x: Tensor

    Returns
    -------
    Tensor
    """
    return torch.where(torch.eye(x.shape[-1]), x, 0)
