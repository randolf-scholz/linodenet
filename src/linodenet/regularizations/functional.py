r"""Regularizations for the Linear ODE Networks.

Notes
-----
Contains regularizations in functional form.
  - See `~linodenet.regularizations.modular` for modular implementations.
"""


__all__ = [
    # Protocol
    "Regularization",
    # Functions
    "logdetexp",
    "matrix_norm",
    # linodenet.projections (matrix groups)
    "hamiltonian",
    "identity",
    "normal",
    "orthogonal",
    "skew_symmetric",
    "symmetric",
    "symplectic",
    "traceless",
    # linodenet.projections (masked)
    "banded",
    "diagonal",
    "lower_triangular",
    "masked",
    "upper_triangular",
]

from typing import Protocol, Union, runtime_checkable

import torch.linalg
from torch import BoolTensor, Tensor, jit

from linodenet.constants import TRUE
from linodenet.projections import functional as projections


@runtime_checkable
class Regularization(Protocol):
    """Protocol for Regularization Components."""

    def __call__(
        self, x: Tensor, /, *, p: int = ..., size_normalize: bool = ...
    ) -> Tensor:
        """Forward pass of the regularization."""
        ...


# region regularizations ---------------------------------------------------------------
@jit.script
def logdetexp(x: Tensor, p: float = 1.0, size_normalize: bool = True) -> Tensor:
    r"""Bias $\det(e^A)$ towards 1.

    .. Signature:: ``(..., n, n) -> ...``

    By Jacobi's formula

    .. math:: \det(e^A) = e^{\tr(A)} ⟺ \log(\det(e^A)) = \tr(A)

    In particular, we can regularize the LinODE model by adding a regularization term of the form

    .. math:: |\tr(A)|^p
    """
    diag = torch.diagonal(x, dim1=-1, dim2=-2)

    if size_normalize:
        traces = torch.mean(diag, dim=-1)
    else:
        traces = torch.sum(diag, dim=-1)

    return torch.abs(traces) ** p


@jit.script
def matrix_norm(
    r: Tensor, p: Union[str, int] = "fro", size_normalize: bool = True
) -> Tensor:
    r"""Return the matrix regularization term.

    .. Signature:: ``(..., n, n) -> ...``
    """
    if isinstance(p, str):
        s = torch.linalg.matrix_norm(r, ord=p, dim=(-2, -1))
    else:
        s = torch.linalg.matrix_norm(r, ord=p, dim=(-2, -1))

    if size_normalize:
        s = s / r.shape[-1]
    return s


# region linodenet.projections  --------------------------------------------------------
# region matrix groups -----------------------------------------------------------------
@jit.script
def identity(
    x: Tensor, p: Union[str, int] = "fro", size_normalize: bool = False
) -> Tensor:
    r"""Bias the matrix towards being zero.

    .. Signature:: ``(..., n, n) -> ...``

    .. math:: A ↦ ‖A-Π(A)‖_p
        where Π(A) = \argmin_X ½∥X∥_F^2
    """
    return matrix_norm(x, p=p, size_normalize=size_normalize)


@jit.script
def skew_symmetric(
    x: Tensor, p: Union[str, int] = "fro", size_normalize: bool = False
) -> Tensor:
    r"""Bias the matrix towards being skew-symmetric.

    .. Signature:: ``(..., n, n) -> ...``

    .. math:: A ↦ ‖A-Π(A)‖_p
        where Π(A) = \argmin_X ½∥X-A∥_F^2 s.t. X^⊤ = -X
    """
    r = x - projections.skew_symmetric(x)
    return matrix_norm(r, p=p, size_normalize=size_normalize)


@jit.script
def symmetric(
    x: Tensor, p: Union[str, int] = "fro", size_normalize: bool = False
) -> Tensor:
    r"""Bias the matrix towards being symmetric.

    .. Signature:: ``(..., n, n) -> ...``

    .. math:: A ↦ ‖A-Π(A)‖_p
        where Π(A) = \argmin_X ½∥X-A∥_F^2 s.t. X^⊤ = +X
    """
    r = x - projections.symmetric(x)
    return matrix_norm(r, p=p, size_normalize=size_normalize)


@jit.script
def orthogonal(
    x: Tensor, p: Union[str, int] = "fro", size_normalize: bool = False
) -> Tensor:
    r"""Bias the matrix towards being orthogonal.

    .. Signature:: ``(..., n, n) -> ...``

    .. math:: A ↦ ‖A-Π(A)‖_p
        where Π(A) = \argmin_X ½∥X-A∥_F^2 s.t. X^⊤X = 𝕀
    """
    r = x - projections.orthogonal(x)
    return matrix_norm(r, p=p, size_normalize=size_normalize)


@jit.script
def normal(
    x: Tensor, p: Union[str, int] = "fro", size_normalize: bool = False
) -> Tensor:
    r"""Bias the matrix towards being normal.

    .. Signature:: ``(..., n, n) -> ...``

    .. math:: A ↦ ‖A-Π(A)‖_p
        where Π(A) = \argmin_X ½∥X-A∥_F^2 s.t. X^⊤X = XX^⊤
    """
    r = x - projections.normal(x)
    return matrix_norm(r, p=p, size_normalize=size_normalize)


@jit.script
def hamiltonian(
    x: Tensor, p: Union[str, int] = "fro", size_normalize: bool = False
) -> Tensor:
    r"""Bias the matrix towards being hamiltonian.

    .. Signature:: ``(..., 2n, 2n) -> ...``

    .. math:: A ↦ ‖A-Π(A)‖_p
        where Π(A) = \argmin_X ½∥X-A∥_F^2 s.t. (JX)^T = JX
    """
    r = x - projections.hamiltonian(x)
    return matrix_norm(r, p=p, size_normalize=size_normalize)


@jit.script
def symplectic(
    x: Tensor, p: Union[str, int] = "fro", size_normalize: bool = False
) -> Tensor:
    r"""Bias the matrix towards being symplectic.

    .. Signature:: ``(..., 2n, 2n) -> ...``

    .. math:: A ↦ ‖A-Π(A)‖_p
        where Π(A) = \argmin_X ½∥X-A∥_F^2 s.t. J^TXJ = X
    """
    r = x - projections.symplectic(x)
    return matrix_norm(r, p=p, size_normalize=size_normalize)


@jit.script
def traceless(
    x: Tensor, p: Union[str, int] = "fro", size_normalize: bool = False
) -> Tensor:
    r"""Bias the matrix towards being normal.

    .. Signature:: ``(..., n, n) -> ...``

    .. math:: A ↦ ‖A-Π(A)‖_p
        where Π(A) = \argmin_X ½∥X-A∥_F^2 s.t. tr(X) = 0

    Note:
        Traceless matrices are also called *trace-free* or *trace-zero* matrices.
        They have the important property that $\det(\exp(X)) = 1$,
        which follows from the fact that $\det(\exp(X)) = \exp(\r(X))$.
    """
    r = x - projections.traceless(x)
    return matrix_norm(r, p=p, size_normalize=size_normalize)


# endregion matrix groups --------------------------------------------------------------


# region masked projections ------------------------------------------------------------
@jit.script
def diagonal(
    x: Tensor, p: Union[str, int] = "fro", size_normalize: bool = False
) -> Tensor:
    r"""Bias the matrix towards being diagonal.

    .. Signature:: ``(..., m, n) -> ...``

    .. math:: A ↦ ‖A-Π(A)‖_p
        where Π(A) = \argmin_X ½∥X-A∥_F^2 s.t. 𝕀⊙X = X
    """
    r = x - projections.diagonal(x)
    return matrix_norm(r, p=p, size_normalize=size_normalize)


@jit.script
def banded(
    x: Tensor,
    upper: int = 0,
    lower: int = 0,
    p: Union[str, int] = "fro",
    size_normalize: bool = False,
) -> Tensor:
    r"""Bias the matrix towards being banded.

    .. Signature:: ``(..., m, n) -> ...``

    .. math:: A ↦ ‖A-Π(A)‖_p
        where Π(A) = \argmin_X ½∥X-A∥_F^2 s.t. 𝔹⊙X = X
    """
    r = x - projections.banded(x, upper=upper, lower=lower)
    return matrix_norm(r, p=p, size_normalize=size_normalize)


@jit.script
def lower_triangular(
    x: Tensor,
    lower: int = 0,
    p: Union[str, int] = "fro",
    size_normalize: bool = False,
) -> Tensor:
    r"""Bias the matrix towards being lower triangular.

    .. Signature:: ``(..., m, n) -> ...``

    .. math:: A ↦ ‖A-Π(A)‖_p
        where Π(A) = \argmin_X ½∥X-A∥_F^2 s.t. 𝕃⊙X = X
    """
    r = x - projections.lower_triangular(x, lower=lower)
    return matrix_norm(r, p=p, size_normalize=size_normalize)


@jit.script
def upper_triangular(
    x: Tensor,
    upper: int = 0,
    p: Union[str, int] = "fro",
    size_normalize: bool = False,
) -> Tensor:
    r"""Bias the matrix towards being upper triangular.

    .. Signature:: ``(..., m, n) -> ...``

    .. math:: A ↦ ‖A-Π(A)‖_p
        where Π(A) = \argmin_X ½∥X-A∥_F^2 s.t. 𝕌⊙X = X
    """
    r = x - projections.upper_triangular(x, upper=upper)
    return matrix_norm(r, p=p, size_normalize=size_normalize)


@jit.script
def masked(
    x: Tensor,
    mask: BoolTensor = TRUE,
    p: Union[str, int] = "fro",
    size_normalize: bool = False,
) -> Tensor:
    r"""Bias the matrix towards being masked.

    .. Signature:: ``(..., m, n) -> ...``

    .. math:: A ↦ ‖A-Π(A)‖_p
        where Π(A) = \argmin_X ½∥X-A∥_F^2 s.t. 𝕄⊙X = X
    """
    r = x - projections.masked(x, mask=mask)
    return matrix_norm(r, p=p, size_normalize=size_normalize)


# endregion masked projections ---------------------------------------------------------
# endregion linodenet.projections  -----------------------------------------------------
# endregion regularizations ------------------------------------------------------------
