r"""Functional regularization penalties for Linear ODE Networks.

Each function returns one penalty per input vector or matrix. For penalties
based on a residual norm, `size_normalize=True` divides that norm by the vector
length or number of matrix columns. `log_det_exp` instead normalizes the trace
before taking its absolute value and raising it to the power $p$.

Notes:
    - See `imtskit.regularizations.modules` for module-based implementations.
"""

__all__ = [
    # ABCs & Protocols
    "Regularization",
    "RegularizationWithArgs",
    # Functions
    "banded",
    "contraction",
    "diagonal",
    "diagonally_dominant",
    "hamiltonian",
    "identity",
    "log_det_exp",
    "lipschitz_bounded",
    "low_rank",
    "lower_triangular",
    "masked",
    "normal",
    "orthogonal",
    "rank_one",
    "skew_symmetric",
    "spectrally_normalized",
    "symmetric",
    "symplectic",
    "traceless",
    "tridiagonal",
    "upper_triangular",
    "unit_vector",
    # helper
    "vector_norm",
    "matrix_norm",
]

from collections.abc import Callable
from typing import Concatenate, Protocol, runtime_checkable

import torch
from torch import Tensor

from imtskit.mappings import functional as projections
from imtskit.types import BoolTensor
from signatures import signature


@runtime_checkable
class Regularization(Protocol):
    r"""Protocol for callable regularization components."""

    def __call__(
        self, x: Tensor, /, *, p: int = ..., size_normalize: bool = ...
    ) -> Tensor:
        r"""Compute a regularization penalty for each input tensor.

        The exponent or norm order is controlled by `p`. The meaning of
        `size_normalize` depends on the implementation: residual norms are divided
        by the vector length or number of matrix columns, while `log_det_exp`
        normalizes the trace before taking its power.
        """
        ...


type RegularizationWithArgs = Callable[Concatenate[Tensor, ...], Tensor]


@signature("(..., m, n) -> (...)")
def matrix_norm(
    r: Tensor, /, *, p: str | int = "fro", size_normalize: bool = True
) -> Tensor:
    r"""Return the matrix norm for each input matrix.

    If `size_normalize` is true, divide the norm by the number of columns.
    """
    s = torch.linalg.matrix_norm(r, ord=p, dim=(-2, -1))

    if size_normalize:
        s = s / r.shape[-1]
    return s


@signature("(..., n) -> (...)")
def vector_norm(r: Tensor, /, *, p: float = 2.0, size_normalize: bool = True) -> Tensor:
    r"""Return the $p$-norm for each input vector.

    If `size_normalize` is true, divide the norm by the vector length.
    """
    s = torch.linalg.vector_norm(r, ord=p, dim=-1)

    if size_normalize:
        s = s / r.shape[-1]
    return s


# region regularizations ---------------------------------------------------------------
@signature("(..., n, n) -> (...)")
def log_det_exp(x: Tensor, /, *, p: float = 1.0, size_normalize: bool = True) -> Tensor:
    r"""Bias $\det(eᴬ)$ towards 1.

    For a real $n × n$ matrix $A$, the determinant identity

    $$ \det(eᴬ) = e^{\tr(A)} $$

    gives $\log(\det(eᴬ)) = \tr(A)$. The penalty is

    $$ |\tr(A)/s|ᵖ $$

    where $s = n$ if `size_normalize` is true and $s = 1$ otherwise.

    Returns:
        penalty (Tensor): One penalty per input matrix.
    """
    diag = torch.diagonal(x, dim1=-1, dim2=-2)
    traces = diag.mean(dim=-1) if size_normalize else diag.sum(dim=-1)
    return traces.abs().pow(p)


# region matrix groups -----------------------------------------------------------------
@signature("(..., m, n) -> (...)")
def identity(
    x: Tensor, /, *, p: str | int = "fro", size_normalize: bool = False
) -> Tensor:
    r"""Bias the matrix towards the identity matrix.

    $$ A ↦ ‖A - 𝕀‖ₚ $$

    Here $𝕀$ is the identity matrix with the same shape as $A$; for rectangular
    inputs, its main diagonal contains ones and all other entries are zero.
    """
    identity = torch.eye(*x.shape[-2:], dtype=x.dtype, device=x.device)
    r = x - identity
    return matrix_norm(r, p=p, size_normalize=size_normalize)


@signature("(..., m, n) -> (...)")
def rank_one(
    x: Tensor, /, *, p: str | int = "fro", size_normalize: bool = False
) -> Tensor:
    r"""Bias the matrix towards having rank at most one.

    $$ A ↦ ‖A - Π(A)‖ₚ $$

    Here $Π(A)$ is obtained by keeping only the leading singular value of $A$.
    It is a closest matrix of rank at most one in Frobenius norm.
    The projection of the zero matrix has rank zero.
    """
    r = x - projections.rank_one(x)
    return matrix_norm(r, p=p, size_normalize=size_normalize)


@signature("(..., n, n) -> (...)")
def symmetric(
    x: Tensor, /, *, p: str | int = "fro", size_normalize: bool = False
) -> Tensor:
    r"""Bias the matrix towards being symmetric.

    $$ A ↦ ‖A - Π(A)‖ₚ $$

    Here $Π(A) = ½(A + Aᵀ)$ is the closest symmetric matrix in Frobenius norm.
    """
    r = x - projections.symmetric(x)
    return matrix_norm(r, p=p, size_normalize=size_normalize)


@signature("(..., n, n) -> (...)")
def skew_symmetric(
    x: Tensor, /, *, p: str | int = "fro", size_normalize: bool = False
) -> Tensor:
    r"""Bias the matrix towards being skew-symmetric.

    $$ A ↦ ‖A - Π(A)‖ₚ $$

    Here $Π(A) = ½(A - Aᵀ)$ is the closest skew-symmetric matrix in Frobenius norm.
    """
    r = x - projections.skew_symmetric(x)
    return matrix_norm(r, p=p, size_normalize=size_normalize)


@signature("(..., n, n) -> (...)")
def orthogonal(
    x: Tensor, /, *, p: str | int = "fro", size_normalize: bool = False
) -> Tensor:
    r"""Bias the matrix towards being orthogonal.

    $$ A ↦ ‖A - Π(A)‖ₚ $$

    Here $Π(A)$ is a closest orthogonal matrix to $A$ in Frobenius norm,
    satisfying $Π(A)ᵀΠ(A) = 𝕀$. For an SVD $A = UΣVᵀ$, a minimizer is $Π(A) = UVᵀ$.
    """
    r = x - projections.orthogonal(x)
    return matrix_norm(r, p=p, size_normalize=size_normalize)


@signature("(..., n, n) -> (...)")
def traceless(
    x: Tensor, /, *, p: str | int = "fro", size_normalize: bool = False
) -> Tensor:
    r"""Bias the matrix towards being traceless.

    $$ A ↦ ‖A - Π(A)‖ₚ $$

    Here $Π(A) = A - \tr(A)𝕀ₙ/n$ is the closest traceless matrix in
    Frobenius norm.

    Note:
        Traceless matrices are also called *trace-free* or *trace-zero* matrices.
        They satisfy $\det(\exp(A)) = 1$, since
        $\det(\exp(A)) = \exp(\tr(A))$.
    """
    r = x - projections.traceless(x)
    return matrix_norm(r, p=p, size_normalize=size_normalize)


@signature("(..., n, n) -> (...)")
def normal(
    x: Tensor, /, *, p: str | int = "fro", size_normalize: bool = False
) -> Tensor:
    r"""Bias the matrix towards being normal.

    $$ A ↦ ‖A - Π(A)‖ₚ $$

    Here $Π(A)$ denotes a closest normal matrix in Frobenius norm.
    For real matrices, the constraint is $Π(A)ᵀΠ(A) = Π(A)Π(A)ᵀ$.

    Raises:
        NotImplementedError: The normal projection is not implemented.
    """
    r = x - projections.normal(x)
    return matrix_norm(r, p=p, size_normalize=size_normalize)


@signature("(..., 2n, 2n) -> (...)")
def hamiltonian(
    x: Tensor, /, *, p: str | int = "fro", size_normalize: bool = False
) -> Tensor:
    r"""Bias the matrix towards being Hamiltonian.

    $$ A ↦ ‖A - Π(A)‖ₚ $$

    Here $Π(A)$ denotes a closest Hamiltonian matrix in Frobenius norm,
    satisfying $(JΠ(A))ᵀ = JΠ(A)$, where

    $$ J = \bmat{ 0 & 𝕀ₙ \\ -𝕀ₙ & 0 } $$

    Raises:
        NotImplementedError: The Hamiltonian projection is not implemented.
    """
    r = x - projections.hamiltonian(x)
    return matrix_norm(r, p=p, size_normalize=size_normalize)


@signature("(..., 2n, 2n) -> (...)")
def symplectic(
    x: Tensor, /, *, p: str | int = "fro", size_normalize: bool = False
) -> Tensor:
    r"""Bias the matrix towards being symplectic.

    $$ A ↦ ‖A - Π(A)‖ₚ $$

    Here $Π(A)$ denotes a closest symplectic matrix in Frobenius norm,
    satisfying $Π(A)ᵀJΠ(A) = J$, where

    $$ J = \bmat{ 0 & 𝕀ₙ \\ -𝕀ₙ & 0 } $$

    Raises:
        NotImplementedError: The symplectic projection is not implemented.
    """
    r = x - projections.symplectic(x)
    return matrix_norm(r, p=p, size_normalize=size_normalize)


# endregion matrix groups --------------------------------------------------------------


# region masked projections ------------------------------------------------------------
@signature("(..., m, n) -> (...)")
def diagonal(
    x: Tensor, /, *, p: str | int = "fro", size_normalize: bool = False
) -> Tensor:
    r"""Bias the matrix towards being diagonal.

    $$ A ↦ ‖A - Π(A)‖ₚ $$

    Here $Π(A)$ retains the main diagonal of $A$ and sets all other entries to zero.
    """
    r = x - projections.diagonal(x)
    return matrix_norm(r, p=p, size_normalize=size_normalize)


@signature("(..., n, n) -> (...)")
def diagonally_dominant(
    x: Tensor, /, *, p: float = 2.0, size_normalize: bool = False
) -> Tensor:
    r"""Bias the matrix towards being row diagonally dominant.

    $$ A ↦ ‖d‖ₚ, \qquad dᵢ = \max(0, ∑_{j≠i}|Aᵢⱼ| - |Aᵢᵢ|) $$

    Each $dᵢ$ is the deficit in row $i$; it is zero when
    $|Aᵢᵢ| ≥ ∑_{j≠i}|Aᵢⱼ|$.
    """
    diagonal = torch.diagonal(x, dim1=-1, dim2=-2).abs()
    row_sums = x.abs().sum(dim=-1) - diagonal
    deficit = torch.relu(row_sums - diagonal)
    return vector_norm(deficit, p=p, size_normalize=size_normalize)


@signature("(..., m, n) -> (...)")
def tridiagonal(
    x: Tensor, /, *, p: str | int = "fro", size_normalize: bool = False
) -> Tensor:
    r"""Bias the matrix towards being tridiagonal.

    $$ A ↦ ‖A - Π(A)‖ₚ $$

    Here $Π(A)$ retains the main diagonal and the first diagonals above and
    below it, setting all other entries to zero.
    """
    r = x - projections.tridiagonal(x)
    return matrix_norm(r, p=p, size_normalize=size_normalize)


@signature("(..., m, n) -> (...)")
def lower_triangular(
    x: Tensor,
    /,
    lower: int = 0,
    *,
    p: str | int = "fro",
    size_normalize: bool = False,
) -> Tensor:
    r"""Bias the matrix towards being lower triangular.

    $$ A ↦ ‖A - Π(A)‖ₚ $$

    Here $Π(A)$ retains entries on and below the diagonal with offset `lower`
    and sets all other entries to zero. An offset of zero selects the main diagonal.
    """
    r = x - projections.lower_triangular(x, lower=lower)
    return matrix_norm(r, p=p, size_normalize=size_normalize)


@signature("(..., m, n) -> (...)")
def upper_triangular(
    x: Tensor,
    /,
    upper: int = 0,
    *,
    p: str | int = "fro",
    size_normalize: bool = False,
) -> Tensor:
    r"""Bias the matrix towards being upper triangular.

    $$ A ↦ ‖A - Π(A)‖ₚ $$

    Here $Π(A)$ retains entries on and above the diagonal with offset `upper`
    and sets all other entries to zero. An offset of zero selects the main diagonal.
    """
    r = x - projections.upper_triangular(x, upper=upper)
    return matrix_norm(r, p=p, size_normalize=size_normalize)


# endregion masked projections ---------------------------------------------------------


# region other regularizations ---------------------------------------------------------


@signature("(..., m, n) -> (...)")
def low_rank(
    x: Tensor, /, rank: int, *, p: str | int = "fro", size_normalize: bool = False
) -> Tensor:
    r"""Bias the matrix towards having rank at most $k$.

    $$ A ↦ ‖A - Π(A)‖ₚ $$

    Here $k$ is `rank`, and $Π(A)$ is obtained by keeping the leading $k$
    singular values of $A$. It is a closest matrix of rank at most $k$ in
    Frobenius norm.
    """
    r = x - projections.low_rank(x, rank=rank)
    return matrix_norm(r, p=p, size_normalize=size_normalize)


@signature("(..., m, n) -> (...)")
def banded(
    x: Tensor,
    /,
    lower: int,
    upper: int,
    *,
    p: str | int = "fro",
    size_normalize: bool = False,
) -> Tensor:
    r"""Bias the matrix towards being banded.

    $$ A ↦ ‖A - Π(A)‖ₚ $$

    Here $Π(A)$ retains entries with diagonal offsets between `lower` and
    `upper`, inclusive, and sets all other entries to zero.
    """
    r = x - projections.banded(x, upper=upper, lower=lower)
    return matrix_norm(r, p=p, size_normalize=size_normalize)


@signature("(..., m, n) -> (...)")
def masked(
    x: Tensor,
    /,
    mask: BoolTensor,
    *,
    p: str | int = "fro",
    size_normalize: bool = False,
) -> Tensor:
    r"""Bias the matrix towards having support on a Boolean mask.

    $$ A ↦ ‖A - Π(A)‖ₚ $$

    Here $Π(A) = M⊙A$, where $M$ is the Boolean `mask` and $⊙$ denotes
    elementwise multiplication. Entries outside the mask are set to zero.
    """
    r = x - projections.masked(x, mask=mask)
    return matrix_norm(r, p=p, size_normalize=size_normalize)


@signature("(..., m, n) -> (...)")
def contraction(
    x: Tensor,
    /,
    lipschitz_bound: float,
    *,
    p: str | int = "fro",
    size_normalize: bool = False,
) -> Tensor:
    r"""Bias the matrix towards having spectral norm at most $γ < 1$.

    $$ A ↦ ‖A - Π(A)‖ₚ $$

    Here $γ$ is `lipschitz_bound`, which must satisfy $0 < γ < 1$, and
    $Π(A)$ is a closest matrix to $A$ in spectral norm subject to $‖Π(A)‖₂ ≤ γ$.
    For a nonzero matrix, $Π(A) = \min(1, γ/‖A‖₂)A$.

    Raises:
        ValueError: The Lipschitz bound is not strictly between zero and one.
    """
    r = x - projections.contraction(x, lipschitz_bound=lipschitz_bound)
    return matrix_norm(r, p=p, size_normalize=size_normalize)


@signature("(..., m, n) -> (...)")
def lipschitz_bounded(
    x: Tensor,
    /,
    lipschitz_bound: float,
    *,
    p: str | int = "fro",
    size_normalize: bool = False,
) -> Tensor:
    r"""Bias the matrix towards having spectral norm at most $γ$.

    $$ A ↦ ‖A - Π(A)‖ₚ $$

    Here $γ ≥ 0$ is `lipschitz_bound`, and $Π(A)$ is a closest matrix to $A$
    in spectral norm subject to $‖Π(A)‖₂ ≤ γ$. For a nonzero matrix,
    $Π(A) = \min(1, γ/‖A‖₂)A$.
    """
    r = x - projections.lipschitz_bounded(x, lipschitz_bound=lipschitz_bound)
    return matrix_norm(r, p=p, size_normalize=size_normalize)


@signature("(..., m, n) -> (...)")
def spectrally_normalized(
    x: Tensor,
    /,
    *,
    p: str | int = "fro",
    size_normalize: bool = False,
) -> Tensor:
    r"""Bias the matrix towards having unit spectral norm.

    $$ A ↦ ‖A - Π(A)‖ₚ $$

    For a nonzero matrix, $Π(A) = A/‖A‖₂$ is a closest matrix to $A$ in
    spectral norm subject to $‖Π(A)‖₂ = 1$.
    The zero matrix is not supported; division by zero produces nonfinite values.
    """
    r = x - projections.spectrally_normalized(x)
    return matrix_norm(r, p=p, size_normalize=size_normalize)


# endregion other regularizations ------------------------------------------------------
# region vector groups -----------------------------------------------------------------
@signature("(..., n) -> (...)")
def unit_vector(
    x: Tensor, /, *, p: float = 2.0, size_normalize: bool = False
) -> Tensor:
    r"""Bias the vector towards having unit Euclidean norm.

    $$ x ↦ ‖x - Π(x)‖ₚ $$

    For a nonzero vector, $Π(x) = x/‖x‖₂$ is the closest unit vector to $x$.
    The zero vector is not supported; division by zero produces nonfinite values.
    """
    r = x - projections.unit_vector(x)
    return vector_norm(r, p=p, size_normalize=size_normalize)


# endregion vector groups --------------------------------------------------------------
# endregion regularizations ------------------------------------------------------------
