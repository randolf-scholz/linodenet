r"""Regularization modules for Linear ODE Networks.

Each module returns one penalty per input vector or matrix. By default,
`size_normalize=True` divides residual norms by the vector length or number
of matrix columns. `LogDetExp` instead normalizes the trace before taking
its absolute value and raising it to the power $p$.

Notes:
    - See `imtskit.regularizations.functional` for functional implementations.
"""

__all__ = [
    # Regularizations
    "Banded",
    "Contraction",
    "Diagonal",
    "DiagonallyDominant",
    "Hamiltonian",
    "Identity",
    "LogDetExp",
    "LipschitzBounded",
    "LowRank",
    "LowerTriangular",
    "Masked",
    "MatrixNorm",
    "Normal",
    "Orthogonal",
    "RankOne",
    "SkewSymmetric",
    "SpectrallyNormalized",
    "Symmetric",
    "Symplectic",
    "Traceless",
    "Tridiagonal",
    "UpperTriangular",
    "UnitVector",
]

from typing import Final

import torch
from torch import Tensor, nn

from imtskit.types import BoolTensor
from signatures import signature

from . import functional as F
from .functional import Regularization


# region regularizations ---------------------------------------------------------------
class LogDetExp(nn.Module, Regularization):
    r"""Bias $\det(eᴬ)$ towards 1.

    For a real $n × n$ matrix $A$, the determinant identity

    $$ \det(eᴬ) = e^{\tr(A)} $$

    gives $\log(\det(eᴬ)) = \tr(A)$. The penalty is

    $$ |\tr(A)/s|ᵖ $$

    where $s = n$ if `size_normalize` is true and $s = 1$ otherwise.

    Calling the module returns one penalty per input matrix.
    """

    p: Final[float]
    size_normalize: Final[bool]

    def __init__(self, *, p: float = 1.0, size_normalize: bool = True) -> None:
        super().__init__()
        self.p = p
        self.size_normalize = size_normalize

    @signature("(..., n, n) -> (...)")
    def forward(self, x: Tensor, /) -> Tensor:
        r"""Return the log-determinant penalty for $x$."""
        return F.log_det_exp(x, p=self.p, size_normalize=self.size_normalize)


class MatrixNorm(nn.Module, Regularization):
    r"""Compute a matrix norm as a regularization penalty.

    If `size_normalize` is true, divide the norm by the number of columns.
    """

    p: Final[str | int]
    size_normalize: Final[bool]

    def __init__(self, *, p: str | int = "fro", size_normalize: bool = True) -> None:
        super().__init__()
        self.p = p
        self.size_normalize = size_normalize

    @signature("(..., m, n) -> (...)")
    def forward(self, x: Tensor, /) -> Tensor:
        r"""Return the matrix norm penalty for $x$."""
        return F.matrix_norm(x, p=self.p, size_normalize=self.size_normalize)


# region matrix groups -----------------------------------------------------------------
class Identity(nn.Module, Regularization):
    r"""Bias the matrix towards the identity matrix."""

    p: Final[str | int]
    size_normalize: Final[bool]

    def __init__(self, *, p: str | int = "fro", size_normalize: bool = True) -> None:
        super().__init__()
        self.p = p
        self.size_normalize = size_normalize

    @signature("(..., m, n) -> (...)")
    def forward(self, x: Tensor, /) -> Tensor:
        r"""Return the identity penalty for $x$."""
        return F.identity(x, p=self.p, size_normalize=self.size_normalize)


class DiagonallyDominant(nn.Module, Regularization):
    r"""Bias the matrix towards being diagonally dominant."""

    p: Final[float]
    size_normalize: Final[bool]

    def __init__(self, *, p: float = 2.0, size_normalize: bool = True) -> None:
        super().__init__()
        self.p = p
        self.size_normalize = size_normalize

    @signature("(..., n, n) -> (...)")
    def forward(self, x: Tensor, /) -> Tensor:
        r"""Return the diagonal dominance penalty for $x$."""
        return F.diagonally_dominant(x, p=self.p, size_normalize=self.size_normalize)


class LowRank(nn.Module, Regularization):
    r"""Bias the matrix towards having rank at most $k$.

    $$ A ↦ ‖A - Π(A)‖ₚ $$

    Here $k$ is `rank`, and $Π(A)$ is obtained by keeping the leading $k$
    singular values of $A$. It is a closest matrix of rank at most $k$ in
    Frobenius norm.
    """

    rank: Final[int]
    p: Final[str | int]
    size_normalize: Final[bool]

    def __init__(
        self, rank: int, *, p: str | int = "fro", size_normalize: bool = True
    ) -> None:
        super().__init__()
        self.rank = rank
        self.p = p
        self.size_normalize = size_normalize

    @signature("(..., m, n) -> (...)")
    def forward(self, x: Tensor, /) -> Tensor:
        r"""Return the low rank penalty for $x$."""
        return F.low_rank(
            x, rank=self.rank, p=self.p, size_normalize=self.size_normalize
        )


class RankOne(nn.Module, Regularization):
    r"""Bias the matrix towards having rank at most one.

    $$ A ↦ ‖A - Π(A)‖ₚ $$

    Here $Π(A)$ is obtained by keeping only the leading singular value of $A$.
    It is a closest matrix of rank at most one in Frobenius norm.
    The projection of the zero matrix has rank zero.
    """

    p: Final[str | int]
    size_normalize: Final[bool]

    def __init__(self, *, p: str | int = "fro", size_normalize: bool = True) -> None:
        super().__init__()
        self.p = p
        self.size_normalize = size_normalize

    @signature("(..., m, n) -> (...)")
    def forward(self, x: Tensor, /) -> Tensor:
        r"""Return the rank-one penalty for $x$."""
        return F.rank_one(x, p=self.p, size_normalize=self.size_normalize)


class Symmetric(nn.Module, Regularization):
    r"""Bias the matrix towards being symmetric.

    $$ A ↦ ‖A - Π(A)‖ₚ $$

    Here $Π(A) = ½(A + Aᵀ)$ is the closest symmetric matrix in Frobenius norm.
    """

    p: Final[str | int]
    size_normalize: Final[bool]

    def __init__(self, *, p: str | int = "fro", size_normalize: bool = True) -> None:
        super().__init__()
        self.p = p
        self.size_normalize = size_normalize

    @signature("(..., n, n) -> (...)")
    def forward(self, x: Tensor, /) -> Tensor:
        r"""Return the symmetry penalty for $x$."""
        return F.symmetric(x, p=self.p, size_normalize=self.size_normalize)


class SkewSymmetric(nn.Module, Regularization):
    r"""Bias the matrix towards being skew-symmetric.

    $$ A ↦ ‖A - Π(A)‖ₚ $$

    Here $Π(A) = ½(A - Aᵀ)$ is the closest skew-symmetric matrix in Frobenius norm.
    """

    p: Final[str | int]
    size_normalize: Final[bool]

    def __init__(self, *, p: str | int = "fro", size_normalize: bool = True) -> None:
        super().__init__()
        self.p = p
        self.size_normalize = size_normalize

    @signature("(..., n, n) -> (...)")
    def forward(self, x: Tensor, /) -> Tensor:
        r"""Return the skew-symmetry penalty for $x$."""
        return F.skew_symmetric(x, p=self.p, size_normalize=self.size_normalize)


class Orthogonal(nn.Module, Regularization):
    r"""Bias the matrix towards being orthogonal.

    $$ A ↦ ‖A - Π(A)‖ₚ $$

    Here $Π(A)$ is a closest orthogonal matrix to $A$ in Frobenius norm,
    satisfying $Π(A)ᵀΠ(A) = 𝕀$. For an SVD $A = UΣVᵀ$, a minimizer is $Π(A) = UVᵀ$.
    """

    p: Final[str | int]
    size_normalize: Final[bool]

    def __init__(self, *, p: str | int = "fro", size_normalize: bool = True) -> None:
        super().__init__()
        self.p = p
        self.size_normalize = size_normalize

    @signature("(..., n, n) -> (...)")
    def forward(self, x: Tensor, /) -> Tensor:
        r"""Return the orthogonality penalty for $x$."""
        return F.orthogonal(x, p=self.p, size_normalize=self.size_normalize)


class Traceless(nn.Module, Regularization):
    r"""Bias the matrix towards being traceless."""

    p: Final[str | int]
    size_normalize: Final[bool]

    def __init__(self, *, p: str | int = "fro", size_normalize: bool = True) -> None:
        super().__init__()
        self.p = p
        self.size_normalize = size_normalize

    @signature("(..., n, n) -> (...)")
    def forward(self, x: Tensor, /) -> Tensor:
        r"""Return the tracelessness penalty for $x$."""
        return F.traceless(x, p=self.p, size_normalize=self.size_normalize)


class Normal(nn.Module, Regularization):
    r"""Bias the matrix towards being normal.

    $$ A ↦ ‖A - Π(A)‖ₚ $$

    Here $Π(A)$ denotes a closest normal matrix in Frobenius norm.
    For real matrices, the constraint is $Π(A)ᵀΠ(A) = Π(A)Π(A)ᵀ$.

    Raises:
        NotImplementedError: The normal projection is not implemented.
    """

    p: Final[str | int]
    size_normalize: Final[bool]

    def __init__(self, *, p: str | int = "fro", size_normalize: bool = True) -> None:
        super().__init__()
        self.p = p
        self.size_normalize = size_normalize

    def forward(self, x: Tensor, /) -> Tensor:
        r"""Return the normality penalty for $x$.

        Raises:
            NotImplementedError: The normal projection is not implemented.
        """
        return F.normal(x, p=self.p, size_normalize=self.size_normalize)


class Symplectic(nn.Module, Regularization):
    r"""Bias the matrix towards being symplectic.

    $$ A ↦ ‖A - Π(A)‖ₚ $$

    Here $Π(A)$ denotes a closest symplectic matrix in Frobenius norm,
    satisfying $Π(A)ᵀJΠ(A) = J$, where

    $$ J = \bmat{ 0 & 𝕀ₙ \\ -𝕀ₙ & 0 } $$

    Raises:
        NotImplementedError: The symplectic projection is not implemented.
    """

    p: Final[str | int]
    size_normalize: Final[bool]

    def __init__(self, *, p: str | int = "fro", size_normalize: bool = True) -> None:
        super().__init__()
        self.p = p
        self.size_normalize = size_normalize

    @signature("(..., 2n, 2n) -> (...)")
    def forward(self, x: Tensor, /) -> Tensor:
        r"""Return the symplectic penalty for $x$.

        Raises:
            NotImplementedError: The symplectic projection is not implemented.
        """
        return F.symplectic(x, p=self.p, size_normalize=self.size_normalize)


class Hamiltonian(nn.Module, Regularization):
    r"""Bias the matrix towards being Hamiltonian.

    $$ A ↦ ‖A - Π(A)‖ₚ $$

    Here $Π(A)$ denotes a closest Hamiltonian matrix in Frobenius norm,
    satisfying $(JΠ(A))ᵀ = JΠ(A)$, where

    $$ J = \bmat{ 0 & 𝕀ₙ \\ -𝕀ₙ & 0 } $$

    Raises:
        NotImplementedError: The Hamiltonian projection is not implemented.
    """

    p: Final[str | int]
    size_normalize: Final[bool]

    def __init__(self, *, p: str | int = "fro", size_normalize: bool = True) -> None:
        super().__init__()
        self.p = p
        self.size_normalize = size_normalize

    @signature("(..., 2n, 2n) -> (...)")
    def forward(self, x: Tensor, /) -> Tensor:
        r"""Return the Hamiltonian penalty for $x$.

        Raises:
            NotImplementedError: The Hamiltonian projection is not implemented.
        """
        return F.hamiltonian(x, p=self.p, size_normalize=self.size_normalize)


# endregion matrix groups --------------------------------------------------------------


# region masked projections ------------------------------------------------------------
class Diagonal(nn.Module, Regularization):
    r"""Bias the matrix towards being diagonal.

    $$ A ↦ ‖A - Π(A)‖ₚ $$

    Here $Π(A)$ retains the main diagonal of $A$ and sets all other entries to zero.
    """

    p: Final[str | int]
    size_normalize: Final[bool]

    def __init__(self, *, p: str | int = "fro", size_normalize: bool = True) -> None:
        super().__init__()
        self.p = p
        self.size_normalize = size_normalize

    @signature("(..., m, n) -> (...)")
    def forward(self, x: Tensor, /) -> Tensor:
        r"""Return the diagonal penalty for $x$."""
        return F.diagonal(x, p=self.p, size_normalize=self.size_normalize)


class LowerTriangular(nn.Module, Regularization):
    r"""Bias the matrix towards being lower triangular.

    $$ A ↦ ‖A - Π(A)‖ₚ $$

    Here $Π(A)$ retains entries on and below the diagonal with offset `lower`
    and sets all other entries to zero. An offset of zero selects the main diagonal.
    """

    p: Final[str | int]
    size_normalize: Final[bool]
    lower: Final[int]

    def __init__(
        self, lower: int = 0, *, p: str | int = "fro", size_normalize: bool = True
    ) -> None:
        super().__init__()
        self.lower = lower
        self.p = p
        self.size_normalize = size_normalize

    @signature("(..., m, n) -> (...)")
    def forward(self, x: Tensor, /) -> Tensor:
        r"""Return the lower triangular penalty for $x$."""
        return F.lower_triangular(
            x, lower=self.lower, p=self.p, size_normalize=self.size_normalize
        )


class UpperTriangular(nn.Module, Regularization):
    r"""Bias the matrix towards being upper triangular.

    $$ A ↦ ‖A - Π(A)‖ₚ $$

    Here $Π(A)$ retains entries on and above the diagonal with offset `upper`
    and sets all other entries to zero. An offset of zero selects the main diagonal.
    """

    p: Final[str | int]
    size_normalize: Final[bool]
    upper: Final[int]

    def __init__(
        self, upper: int = 0, *, p: str | int = "fro", size_normalize: bool = True
    ) -> None:
        super().__init__()
        self.upper = upper
        self.p = p
        self.size_normalize = size_normalize

    @signature("(..., m, n) -> (...)")
    def forward(self, x: Tensor, /) -> Tensor:
        r"""Return the upper triangular penalty for $x$."""
        return F.upper_triangular(
            x, upper=self.upper, p=self.p, size_normalize=self.size_normalize
        )


class Banded(nn.Module, Regularization):
    r"""Bias the matrix towards being banded.

    $$ A ↦ ‖A - Π(A)‖ₚ $$

    Here $Π(A)$ retains entries with diagonal offsets between `lower` and
    `upper`, inclusive, and sets all other entries to zero.
    """

    p: Final[str | int]
    size_normalize: Final[bool]
    upper: Final[int]
    lower: Final[int]

    def __init__(
        self,
        lower: int,
        upper: int,
        *,
        p: str | int = "fro",
        size_normalize: bool = True,
    ) -> None:
        super().__init__()
        self.lower = lower
        self.upper = upper
        self.p = p
        self.size_normalize = size_normalize

    @signature("(..., m, n) -> (...)")
    def forward(self, x: Tensor, /) -> Tensor:
        r"""Return the banded penalty for $x$."""
        return F.banded(
            x,
            lower=self.lower,
            upper=self.upper,
            p=self.p,
            size_normalize=self.size_normalize,
        )


class Tridiagonal(nn.Module, Regularization):
    r"""Bias the matrix towards being tridiagonal.

    $$ A ↦ ‖A - Π(A)‖ₚ $$

    Here $Π(A)$ retains the main diagonal and the first diagonals above and
    below it, setting all other entries to zero.
    """

    p: Final[str | int]
    size_normalize: Final[bool]

    def __init__(self, *, p: str | int = "fro", size_normalize: bool = True) -> None:
        super().__init__()
        self.p = p
        self.size_normalize = size_normalize

    @signature("(..., m, n) -> (...)")
    def forward(self, x: Tensor, /) -> Tensor:
        r"""Return the tridiagonal penalty for $x$."""
        return F.tridiagonal(x, p=self.p, size_normalize=self.size_normalize)


class Masked(nn.Module, Regularization):
    r"""Bias the matrix towards having support on a Boolean mask.

    $$ A ↦ ‖A - Π(A)‖ₚ $$

    Here $Π(A) = M⊙A$, where $M$ is the Boolean `mask` and $⊙$ denotes
    elementwise multiplication. Entries outside the mask are set to zero.
    """

    p: Final[str | int]
    size_normalize: Final[bool]
    mask: BoolTensor

    def __init__(
        self,
        mask: BoolTensor,
        *,
        p: str | int = "fro",
        size_normalize: bool = True,
    ) -> None:
        super().__init__()
        self.mask = torch.as_tensor(mask, dtype=torch.bool)
        self.p = p
        self.size_normalize = size_normalize

    @signature("(..., m, n) -> (...)")
    def forward(self, x: Tensor, /) -> Tensor:
        r"""Return the masked penalty for $x$."""
        return F.masked(x, mask=self.mask, p=self.p, size_normalize=self.size_normalize)


# endregion masked projections ---------------------------------------------------------


# region other regularizations ---------------------------------------------------------
class Contraction(nn.Module, Regularization):
    r"""Bias the matrix towards having spectral norm at most $γ < 1$.

    $$ A ↦ ‖A - Π(A)‖ₚ $$

    Here $γ$ is `lipschitz_bound`, which must satisfy $0 < γ < 1$, and
    $Π(A)$ is a closest matrix to $A$ in spectral norm subject to $‖Π(A)‖₂ ≤ γ$.
    For a nonzero matrix, $Π(A) = \min(1, γ/‖A‖₂)A$.

    Raises:
        ValueError: The Lipschitz bound is not strictly between zero and one.
    """

    lipschitz_bound: Final[float]

    p: Final[str | int]
    size_normalize: Final[bool]

    def __init__(
        self,
        lipschitz_bound: float,
        *,
        p: str | int = "fro",
        size_normalize: bool = True,
    ) -> None:
        super().__init__()
        self.p = p
        self.size_normalize = size_normalize
        self.lipschitz_bound = lipschitz_bound

    @signature("(..., m, n) -> (...)")
    def forward(self, x: Tensor, /) -> Tensor:
        r"""Return the contraction penalty for $x$."""
        return F.contraction(
            x, self.lipschitz_bound, p=self.p, size_normalize=self.size_normalize
        )


class LipschitzBounded(nn.Module, Regularization):
    r"""Bias the matrix towards having spectral norm at most $γ$.

    $$ A ↦ ‖A - Π(A)‖ₚ $$

    Here $γ ≥ 0$ is `lipschitz_bound`, and $Π(A)$ is a closest matrix to $A$
    in spectral norm subject to $‖Π(A)‖₂ ≤ γ$. For a nonzero matrix,
    $Π(A) = \min(1, γ/‖A‖₂)A$.
    """

    lipschitz_bound: Final[float]
    p: Final[str | int]
    size_normalize: Final[bool]

    def __init__(
        self,
        lipschitz_bound: float,
        *,
        p: str | int = "fro",
        size_normalize: bool = True,
    ) -> None:
        super().__init__()
        self.lipschitz_bound = lipschitz_bound
        self.p = p
        self.size_normalize = size_normalize

    @signature("(..., m, n) -> (...)")
    def forward(self, x: Tensor, /) -> Tensor:
        r"""Return the Lipschitz bound penalty for $x$."""
        return F.lipschitz_bounded(
            x,
            self.lipschitz_bound,
            p=self.p,
            size_normalize=self.size_normalize,
        )


class SpectrallyNormalized(nn.Module, Regularization):
    r"""Bias the matrix towards having unit spectral norm.

    $$ A ↦ ‖A - Π(A)‖ₚ $$

    For a nonzero matrix, $Π(A) = A/‖A‖₂$ is a closest matrix to $A$ in
    spectral norm subject to $‖Π(A)‖₂ = 1$.
    The zero matrix is not supported; division by zero produces nonfinite values.
    """

    p: Final[str | int]
    size_normalize: Final[bool]

    def __init__(self, *, p: str | int = "fro", size_normalize: bool = True) -> None:
        super().__init__()
        self.p = p
        self.size_normalize = size_normalize

    @signature("(..., m, n) -> (...)")
    def forward(self, x: Tensor, /) -> Tensor:
        r"""Return the spectral normalization penalty for $x$."""
        return F.spectrally_normalized(x, p=self.p, size_normalize=self.size_normalize)


# endregion other regularizations ------------------------------------------------------
# region vector groups -----------------------------------------------------------------
class UnitVector(nn.Module, Regularization):
    r"""Bias the vector towards having unit Euclidean norm.

    $$ x ↦ ‖x - Π(x)‖ₚ $$

    For a nonzero vector, $Π(x) = x/‖x‖₂$ is the closest unit vector to $x$.
    The zero vector is not supported; division by zero produces nonfinite values.
    """

    p: Final[float]
    size_normalize: Final[bool]

    def __init__(self, *, p: float = 2.0, size_normalize: bool = True) -> None:
        super().__init__()
        self.p = p
        self.size_normalize = size_normalize

    @signature("(..., n) -> (...)")
    def forward(self, x: Tensor, /) -> Tensor:
        r"""Return the unit-vector penalty for $x$."""
        return F.unit_vector(x, p=self.p, size_normalize=self.size_normalize)


# endregion vector groups --------------------------------------------------------------
# endregion regularizations ------------------------------------------------------------
