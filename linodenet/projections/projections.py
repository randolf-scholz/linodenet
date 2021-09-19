r"""Projection Mappings."""
from __future__ import annotations

import logging
from typing import Final

import torch
from torch import Tensor, jit

LOGGER = logging.getLogger(__name__)

__all__: Final[list[str]] = [
    "identity",
    "symmetric",
    "skew_symmetric",
    "orthogonal",
    "diagonal",
    "normal",
]


@jit.script
def identity(x: Tensor) -> Tensor:
    r"""Return x as-is.

    .. math::
        \min_Y ½∥X-Y∥_F^2

    **Signature:** ``(..., n,n) ⟶ (..., n, n)``

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

    .. math::
        \min_Y ½∥X-Y∥_F^2 s.t. Yᵀ = Y

    One can show analytically that Y = ½(X + Xᵀ) is the unique minimizer.

    **Signature:** ``(..., n,n) ⟶ (..., n, n)``

    Parameters
    ----------
    x: Tensor

    Returns
    -------
    Tensor
    """
    return (x + x.swapaxes(-1, -2)) / 2


@jit.script
def skew_symmetric(x: Tensor) -> Tensor:
    r"""Return the closest skew-symmetric matrix to X.

    .. math::
        \min_Y ½∥X-Y∥_F^2 s.t. Yᵀ = -Y

    One can show analytically that Y = ½(X - Xᵀ) is the unique minimizer.

    **Signature:** ``(..., n,n) ⟶ (..., n, n)``

    Parameters
    ----------
    x: Tensor

    Returns
    -------
    Tensor
    """
    return (x - x.swapaxes(-1, -2)) / 2


@jit.script
def normal(x: Tensor) -> Tensor:
    r"""Return the closest normal matrix to X.

    .. math::
        \min_Y ½∥X-Y∥_F^2 s.t. YᵀY = YYᵀ

    The Lagrangian
    --------------

    .. math::
        ℒ(Y, Λ) = ½∥X-Y∥_F^2 + ⟨Λ, [Y, Yᵀ]⟩

    First order necessary KKT condition
    -----------------------------------

    .. math::
            0 &= ∇ℒ(Y, Λ) = (Y-X) + Y(Λ + Λᵀ) - (Λ + Λᵀ)Y
        \\⟺ Y &= X + [Y, Λ]

    Second order sufficient KKT condition
    -------------------------------------

    .. math::
             ⟨∇h|S⟩=0     &⟹ ⟨S|∇²ℒ|S⟩ ≥ 0
         \\⟺ ⟨[Y, Λ]|S⟩=0 &⟹ ⟨S|𝕀⊗𝕀 + Λ⊗𝕀 − 𝕀⊗Λ|S⟩ ≥ 0
         \\⟺ ⟨[Y, Λ]|S⟩=0 &⟹ ⟨S|S⟩ + ⟨[S, Λ]|S⟩ ≥ 0

    **Signature:** ``(..., n,n) ⟶ (..., n, n)``

    Parameters
    ----------
    x: Tensor

    Returns
    -------
    Tensor
    """
    raise NotImplementedError("TODO: implement Fixpoint / Gradient based algorithm.")


@jit.script
def orthogonal(x: Tensor) -> Tensor:
    r"""Return the closest orthogonal matrix to X.

    .. math::
        \min_Y ½∥X-Y∥_F^2 s.t. YᵀY = 𝕀 = YYᵀ

    One can show analytically that `Y = UVᵀ` is the unique minimizer,
    where `X=UΣVᵀ` is the SVD of `X`.

    **Signature:** ``(..., n,n) ⟶ (..., n, n)``

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

    .. math::
        \min_Y ½∥X-Y∥_F^2 s.t. Y = 𝕀⊙Y

    One can show analytically that `Y = diag(X)` is the unique minimizer.

    **Signature:** ``(..., n,n) ⟶ (..., n, n)``

    Parameters
    ----------
    x: Tensor

    Returns
    -------
    Tensor
    """
    eye = torch.eye(x.shape[-1], dtype=torch.bool, device=x.device)
    zero = torch.tensor(0.0, dtype=x.dtype, device=x.device)
    return torch.where(eye, x, zero)
