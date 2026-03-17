r"""Regularizations for the Linear ODE Networks.

Notes:
    - See `linodenet.regularizations.functional` for functional implementations.
    - See `linodenet.regularizations.modules` for module-based implementations.
"""

__all__ = [
    # ABCs & Protocols
    "Regularization",
    # Functions
    "banded",
    "contraction",
    "diagonal",
    "hamiltonian",
    "identity",
    "log_det_exp",
    "low_rank",
    "lower_triangular",
    "masked",
    "matrix_norm",
    "normal",
    "orthogonal",
    "skew_symmetric",
    "symmetric",
    "symplectic",
    "traceless",
    "upper_triangular",
]

from typing import Protocol, runtime_checkable

import torch
from torch import Tensor

from linodenet.projections import functional as projections
from linodenet.types import BoolTensor
from signatures import signature


@runtime_checkable
class Regularization(Protocol):
    r"""Protocol for Regularization Components."""

    def __call__(
        self, x: Tensor, /, *, p: int = ..., size_normalize: bool = ...
    ) -> Tensor:
        r"""Forward pass of the regularization."""
        ...


# region regularizations ---------------------------------------------------------------
@signature("(..., n, n) -> (...)")
def log_det_exp(x: Tensor, p: float = 1.0, size_normalize: bool = True) -> Tensor:
    r"""Bias $\det(eᴬ)$ towards 1.

    Returns:
        .. math:: |\tr(A)|ᵖ

    By Jacobi's formula

    .. math:: \det(eᴬ) = e^{\tr(A)} ⟺ \log(\det(eᴬ)) = \tr(A)
    """
    diag = torch.diagonal(x, dim1=-1, dim2=-2)
    traces = diag.mean(dim=-1) if size_normalize else diag.sum(dim=-1)
    return traces.abs().pow(p)


@signature("(..., m, n) -> (...)")
def matrix_norm(r: Tensor, p: str | int = "fro", size_normalize: bool = True) -> Tensor:
    r"""Return the matrix regularization term."""
    if isinstance(p, str):
        s = torch.linalg.matrix_norm(r, ord=p, dim=(-2, -1))
    else:
        s = torch.linalg.matrix_norm(r, ord=p, dim=(-2, -1))

    if size_normalize:
        s = s / r.shape[-1]
    return s


# region matrix groups -----------------------------------------------------------------
@signature("(..., m, n) -> (...)")
def identity(x: Tensor, p: str | int = "fro", size_normalize: bool = False) -> Tensor:
    r"""Bias the matrix towards being zero.

    .. math:: A ↦ ‖A-Π(A)‖ₚ

    where $Π(A) = \argmin_X ½‖X‖²$
    """
    return matrix_norm(x, p=p, size_normalize=size_normalize)


@signature("(..., m, n) -> (...)")
def low_rank(
    x: Tensor, rank: int, p: str | int = "fro", size_normalize: bool = False
) -> Tensor:
    r"""Bias the matrix towards being low rank.

    .. math:: A ↦ ‖A-Π(A)‖ₚ

    where $Π(A)$ is the closest rank-k matrix to $A$.
    """
    r = x - projections.low_rank(x, rank=rank)
    return matrix_norm(r, p=p, size_normalize=size_normalize)


@signature("(..., n, n) -> (...)")
def symmetric(x: Tensor, p: str | int = "fro", size_normalize: bool = False) -> Tensor:
    r"""Bias the matrix towards being symmetric.

    .. math:: A ↦ ‖A-Π(A)‖ₚ

    where $Π(A) = \argmin_X ½‖X-A‖²$ s.t. $Xᵀ = +X$
    """
    r = x - projections.symmetric(x)
    return matrix_norm(r, p=p, size_normalize=size_normalize)


@signature("(..., n, n) -> (...)")
def skew_symmetric(
    x: Tensor, p: str | int = "fro", size_normalize: bool = False
) -> Tensor:
    r"""Bias the matrix towards being skew-symmetric.

    .. math:: A ↦ ‖A-Π(A)‖ₚ

    where $Π(A) = \argmin_X ½‖X-A‖²$ s.t. $Xᵀ = -X$
    """
    r = x - projections.skew_symmetric(x)
    return matrix_norm(r, p=p, size_normalize=size_normalize)


@signature("(..., n, n) -> (...)")
def orthogonal(x: Tensor, p: str | int = "fro", size_normalize: bool = False) -> Tensor:
    r"""Bias the matrix towards being orthogonal.

    .. math:: A ↦ ‖A-Π(A)‖ₚ

    where $Π(A) = \argmin_X ½‖X-A‖² s.t. XᵀX = 𝕀$
    """
    r = x - projections.orthogonal(x)
    return matrix_norm(r, p=p, size_normalize=size_normalize)


@signature("(..., n, n) -> (...)")
def traceless(x: Tensor, p: str | int = "fro", size_normalize: bool = False) -> Tensor:
    r"""Bias the matrix towards being normal.

    .. math:: A ↦ ‖A-Π(A)‖ₚ

    where $Π(A) = \argmin_X ½‖X-A‖²$ s.t. $tr(X) = 0$

    Note:
        Traceless matrices are also called *trace-free* or *trace-zero* matrices.
        They have the important property that $\det(\exp(X)) = 1$,
        which follows from the fact that $\det(\exp(X)) = \exp(\r(X))$.
    """
    r = x - projections.traceless(x)
    return matrix_norm(r, p=p, size_normalize=size_normalize)


@signature("(..., n, n) -> (...)")
def normal(x: Tensor, p: str | int = "fro", size_normalize: bool = False) -> Tensor:
    r"""Bias the matrix towards being normal.

    .. math:: A ↦ ‖A-Π(A)‖ₚ

    where $Π(A) = \argmin_X ½‖X-A‖²$ s.t. $XᵀX = XXᵀ$
    """
    r = x - projections.normal(x)
    return matrix_norm(r, p=p, size_normalize=size_normalize)


@signature("(..., 2n, 2n) -> (...)")
def hamiltonian(
    x: Tensor, p: str | int = "fro", size_normalize: bool = False
) -> Tensor:
    r"""Bias the matrix towards being hamiltonian.

    .. math:: A ↦ ‖A-Π(A)‖ₚ

    where $Π(A) = \argmin_X ½‖X-A‖²$ s.t. $(JX)ᵀ = JX$
    """
    r = x - projections.hamiltonian(x)
    return matrix_norm(r, p=p, size_normalize=size_normalize)


@signature("(..., 2n, 2n) -> (...)")
def symplectic(x: Tensor, p: str | int = "fro", size_normalize: bool = False) -> Tensor:
    r"""Bias the matrix towards being symplectic.

    .. math:: A ↦ ‖A-Π(A)‖ₚ

    where $Π(A) = \argmin_X ½‖X-A‖²$ s.t. $JᵀXJ = X$
    """
    r = x - projections.symplectic(x)
    return matrix_norm(r, p=p, size_normalize=size_normalize)


# endregion matrix groups --------------------------------------------------------------


# region masked projections ------------------------------------------------------------
@signature("(..., m, n) -> (...)")
def diagonal(x: Tensor, p: str | int = "fro", size_normalize: bool = False) -> Tensor:
    r"""Bias the matrix towards being diagonal.

    .. math:: A ↦ ‖A-Π(A)‖ₚ

    where $Π(A) = \argmin_X ½‖X-A‖²$ s.t. $𝕀⊙X = X$
    """
    r = x - projections.diagonal(x)
    return matrix_norm(r, p=p, size_normalize=size_normalize)


@signature("(..., m, n) -> (...)")
def banded(
    x: Tensor,
    lower: int,
    upper: int,
    p: str | int = "fro",
    size_normalize: bool = False,
) -> Tensor:
    r"""Bias the matrix towards being banded.

    .. math:: A ↦ ‖A-Π(A)‖ₚ

    where $Π(A) = \argmin_X ½‖X-A‖²$ s.t. $B⊙X = X$
    """
    r = x - projections.banded(x, upper=upper, lower=lower)
    return matrix_norm(r, p=p, size_normalize=size_normalize)


@signature("(..., m, n) -> (...)")
def lower_triangular(
    x: Tensor,
    lower: int = 0,
    p: str | int = "fro",
    size_normalize: bool = False,
) -> Tensor:
    r"""Bias the matrix towards being lower triangular.

    .. math:: A ↦ ‖A-Π(A)‖ₚ

    where $Π(A) = \argmin_X ½‖X-A‖²$ s.t. $L⊙X = X$
    """
    r = x - projections.lower_triangular(x, lower=lower)
    return matrix_norm(r, p=p, size_normalize=size_normalize)


@signature("(..., m, n) -> (...)")
def upper_triangular(
    x: Tensor,
    upper: int = 0,
    p: str | int = "fro",
    size_normalize: bool = False,
) -> Tensor:
    r"""Bias the matrix towards being upper triangular.

    .. math:: A ↦ ‖A-Π(A)‖ₚ

    where $Π(A) = \argmin_X ½‖X-A‖²$ s.t. $U⊙X = X$
    """
    r = x - projections.upper_triangular(x, upper=upper)
    return matrix_norm(r, p=p, size_normalize=size_normalize)


@signature("(..., m, n) -> (...)")
def masked(
    x: Tensor,
    mask: BoolTensor,
    p: str | int = "fro",
    size_normalize: bool = False,
) -> Tensor:
    r"""Bias the matrix towards being masked.

    .. math:: A ↦ ‖A-Π(A)‖ₚ

    where $Π(A) = \argmin_X ½‖X-A‖²$ s.t. $M⊙X = X$
    """
    r = x - projections.masked(x, mask=mask)
    return matrix_norm(r, p=p, size_normalize=size_normalize)


# endregion masked projections ---------------------------------------------------------


# region other regularizations ---------------------------------------------------------
@signature("(..., m, n) -> (...)")
def contraction(
    x: Tensor, p: str | int = "fro", size_normalize: bool = False
) -> Tensor:
    r"""Bias the matrix towards being a contraction.

    .. math:: A ↦ ‖A-Π(A)‖ₚ

    where $Π(A) = \argmin_X ‖X-A‖₂$ s.t. $‖X‖₂≤1$
    """
    r = x - projections.contraction(x)
    return matrix_norm(r, p=p, size_normalize=size_normalize)


# endregion other regularizations ------------------------------------------------------
# endregion regularizations ------------------------------------------------------------
