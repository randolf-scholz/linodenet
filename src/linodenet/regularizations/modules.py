r"""Regularizations for the Linear ODE Networks.

Notes:
    - See `linodenet.regularizations.functional` for functional implementations.
    - See `linodenet.regularizations.modules` for module-based implementations.
"""

__all__ = [
    # ABCs & Protocols
    "RegularizationBase",
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
    "SpectralNormalized",
    "Symmetric",
    "Symplectic",
    "Traceless",
    "Tridiagonal",
    "UpperTriangular",
    "UnitVector",
]

from abc import abstractmethod
from typing import Final

import torch
from torch import Tensor, nn

from linodenet.types import BoolTensor
from signatures import signature

from .functional import (
    banded,
    contraction,
    diagonal,
    diagonally_dominant,
    hamiltonian,
    identity,
    lipschitz_bounded,
    log_det_exp,
    low_rank,
    lower_triangular,
    masked,
    matrix_norm,
    normal,
    orthogonal,
    rank_one,
    skew_symmetric,
    spectral_normalized,
    symmetric,
    symplectic,
    traceless,
    tridiagonal,
    unit_vector,
    upper_triangular,
)


class RegularizationBase(nn.Module):
    r"""Abstract Base Class for Regularization components."""

    @abstractmethod
    @signature("(..., *ds) -> (...)")
    def forward(self, x: Tensor, /) -> Tensor:
        r"""Forward pass of the regularization.

        Args:
            x: The input tensor to be regularized.

        Returns:
            r: The (scalar) regularization value .
        """


# region regularizations ---------------------------------------------------------------
class LogDetExp(RegularizationBase):
    r"""Bias $\det(eᴬ)$ towards 1.

    By Jacobi's formula

    .. math:: \det(eᴬ) = e^{\tr(A)} ⟺ \log(\det(eᴬ)) = \tr(A)

    In particular, we can regularize the LinODE model by adding a regularization term of the form

    .. math:: \abs{\tr(A)}ᵖ
    """

    p: Final[float]
    size_normalize: Final[bool]

    def __init__(self, *, p: float = 1.0, size_normalize: bool = True) -> None:
        super().__init__()
        self.p = p
        self.size_normalize = size_normalize

    @signature("(..., n, n) -> (...)")
    def forward(self, x: Tensor) -> Tensor:
        r"""Bias $\det(eᴬ)$ towards 1."""
        return log_det_exp(x, p=self.p, size_normalize=self.size_normalize)


class MatrixNorm(RegularizationBase):
    r"""Return the matrix regularization term."""

    p: Final[str | int]
    size_normalize: Final[bool]

    def __init__(self, *, p: str | int = "fro", size_normalize: bool = True) -> None:
        super().__init__()
        self.p = p
        self.size_normalize = size_normalize

    @signature("(..., m, n) -> (...)")
    def forward(self, x: Tensor) -> Tensor:
        r"""Bias x towards zero matrix."""
        return matrix_norm(x, p=self.p, size_normalize=self.size_normalize)


# region matrix groups -----------------------------------------------------------------
class Identity(RegularizationBase):
    r"""Bias the matrix towards the identity matrix."""

    p: Final[str | int]
    size_normalize: Final[bool]

    def __init__(self, *, p: str | int = "fro", size_normalize: bool = True) -> None:
        super().__init__()
        self.p = p
        self.size_normalize = size_normalize

    @signature("(..., m, n) -> (...)")
    def forward(self, x: Tensor) -> Tensor:
        r"""Bias x towards the identity matrix."""
        return identity(x, p=self.p, size_normalize=self.size_normalize)


class DiagonallyDominant(RegularizationBase):
    r"""Bias the matrix towards being diagonally dominant."""

    p: Final[float]
    size_normalize: Final[bool]

    def __init__(self, *, p: float = 2.0, size_normalize: bool = True) -> None:
        super().__init__()
        self.p = p
        self.size_normalize = size_normalize

    @signature("(..., n, n) -> (...)")
    def forward(self, x: Tensor) -> Tensor:
        r"""Bias x towards diagonal dominance."""
        return diagonally_dominant(x, p=self.p, size_normalize=self.size_normalize)


class LowRank(RegularizationBase):
    r"""Bias the matrix towards being low-rank.

    .. math:: A ↦ ‖A-Π(A)‖ₚ

    where $Π(A) = \argmin_X ½‖X-A‖²$ s.t. $rank(X) ≤ k$
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
    def forward(self, x: Tensor) -> Tensor:
        r"""Bias x towards low-rank matrix."""
        return low_rank(x, rank=self.rank, p=self.p, size_normalize=self.size_normalize)


class RankOne(RegularizationBase):
    r"""Bias the matrix towards being rank-1.

    .. math:: A ↦ ‖A-Π(A)‖ₚ

    where $Π(A)$ is the closest rank-1 matrix to $A$.
    """

    p: Final[str | int]
    size_normalize: Final[bool]

    def __init__(self, *, p: str | int = "fro", size_normalize: bool = True) -> None:
        super().__init__()
        self.p = p
        self.size_normalize = size_normalize

    @signature("(..., m, n) -> (...)")
    def forward(self, x: Tensor) -> Tensor:
        r"""Bias x towards rank-1 matrix."""
        return rank_one(x, p=self.p, size_normalize=self.size_normalize)


class Symmetric(RegularizationBase):
    r"""Bias the matrix towards being symmetric.

    .. math:: A ↦ ‖A-Π(A)‖ₚ Π(A) = \argmin_X ½‖X-A‖² s.t. Xᵀ = +X
    """

    p: Final[str | int]
    size_normalize: Final[bool]

    def __init__(self, *, p: str | int = "fro", size_normalize: bool = True) -> None:
        super().__init__()
        self.p = p
        self.size_normalize = size_normalize

    @signature("(..., n, n) -> (...)")
    def forward(self, x: Tensor) -> Tensor:
        r"""Bias x towards symmetric matrix."""
        return symmetric(x, p=self.p, size_normalize=self.size_normalize)


class SkewSymmetric(RegularizationBase):
    r"""Bias the matrix towards being skew-symmetric.

    .. math:: A ↦ ‖A-Π(A)‖ₚ Π(A) = \argmin_X ½‖X-A‖² s.t. Xᵀ = -X
    """

    p: Final[str | int]
    size_normalize: Final[bool]

    def __init__(self, *, p: str | int = "fro", size_normalize: bool = True) -> None:
        super().__init__()
        self.p = p
        self.size_normalize = size_normalize

    @signature("(..., n, n) -> (...)")
    def forward(self, x: Tensor) -> Tensor:
        r"""Bias x towards skew-symmetric matrix."""
        return skew_symmetric(x, p=self.p, size_normalize=self.size_normalize)


class Orthogonal(RegularizationBase):
    r"""Bias the matrix towards being orthogonal.

    .. math:: A ↦ ‖A-Π(A)‖ₚ

    where $Π(A) = \argmin_X ½‖X-A‖²$ s.t. $XᵀX = 𝕀$
    """

    p: Final[str | int]
    size_normalize: Final[bool]

    def __init__(self, *, p: str | int = "fro", size_normalize: bool = True) -> None:
        super().__init__()
        self.p = p
        self.size_normalize = size_normalize

    @signature("(..., n, n) -> (...)")
    def forward(self, x: Tensor) -> Tensor:
        r"""Bias x towards orthogonal matrix."""
        return orthogonal(x, p=self.p, size_normalize=self.size_normalize)


class Traceless(RegularizationBase):
    r"""Bias the matrix towards being traceless."""

    p: Final[str | int]
    size_normalize: Final[bool]

    def __init__(self, *, p: str | int = "fro", size_normalize: bool = True) -> None:
        super().__init__()
        self.p = p
        self.size_normalize = size_normalize

    @signature("(..., n, n) -> (...)")
    def forward(self, x: Tensor) -> Tensor:
        r"""Bias x towards normal matrix."""
        return traceless(x, p=self.p, size_normalize=self.size_normalize)


class Normal(RegularizationBase):
    r"""Bias the matrix towards being orthogonal.

    .. math:: A ↦ ‖A-Π(A)‖ₚ

    where $Π(A) = \argmin_X ½‖X-A‖²$ s.t. $XᵀX = 𝕀$
    """

    p: Final[str | int]
    size_normalize: Final[bool]

    def __init__(self, *, p: str | int = "fro", size_normalize: bool = True) -> None:
        super().__init__()
        self.p = p
        self.size_normalize = size_normalize

    def forward(self, x: Tensor) -> Tensor:
        r"""Bias x towards normal matrix."""
        return normal(x, p=self.p, size_normalize=self.size_normalize)


class Symplectic(RegularizationBase):
    r"""Bias the matrix towards being symplectic.

    .. math:: A ↦ ‖A-Π(A)‖ₚ

    where $Π(A) = \argmin_X ½‖X-A‖²$ s.t. $JᵀXJ = X$
    """

    p: Final[str | int]
    size_normalize: Final[bool]

    def __init__(self, *, p: str | int = "fro", size_normalize: bool = True) -> None:
        super().__init__()
        self.p = p
        self.size_normalize = size_normalize

    @signature("(..., 2n, 2n) -> (...)")
    def forward(self, x: Tensor) -> Tensor:
        r"""Bias x towards normal matrix."""
        return symplectic(x, p=self.p, size_normalize=self.size_normalize)


class Hamiltonian(RegularizationBase):
    r"""Bias the matrix towards being hamiltonian.

    .. math:: A ↦ ‖A-Π(A)‖ₚ

    where $Π(A) = \argmin_X ½‖X-A‖²$ s.t. $(JX)ᵀ = JX$
    """

    p: Final[str | int]
    size_normalize: Final[bool]

    def __init__(self, *, p: str | int = "fro", size_normalize: bool = True) -> None:
        super().__init__()
        self.p = p
        self.size_normalize = size_normalize

    @signature("(..., 2n, 2n) -> (...)")
    def forward(self, x: Tensor) -> Tensor:
        r"""Bias x towards normal matrix."""
        return hamiltonian(x, p=self.p, size_normalize=self.size_normalize)


# endregion matrix groups --------------------------------------------------------------


# region masked projections ------------------------------------------------------------
class Diagonal(RegularizationBase):
    r"""Bias the matrix towards being diagonal.

    .. math:: A ↦ ‖A-Π(A)‖ₚ

    where $Π(A) = \argmin_X ½‖X-A‖²$ s.t. $𝕀⊙X = X$
    """

    p: Final[str | int]
    size_normalize: Final[bool]

    def __init__(self, *, p: str | int = "fro", size_normalize: bool = True) -> None:
        super().__init__()
        self.p = p
        self.size_normalize = size_normalize

    @signature("(..., m, n) -> (...)")
    def forward(self, x: Tensor) -> Tensor:
        r"""Bias x towards diagonal matrix."""
        return diagonal(x, p=self.p, size_normalize=self.size_normalize)


class LowerTriangular(RegularizationBase):
    r"""Bias the matrix towards being lower triangular.

    .. math:: A ↦ ‖A-Π(A)‖ₚ

    where $Π(A) = \argmin_X ½‖X-A‖²$ s.t. $L⊙X = X$
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
    def forward(self, x: Tensor) -> Tensor:
        r"""Bias x towards lower triangular matrix."""
        return lower_triangular(
            x, lower=self.lower, p=self.p, size_normalize=self.size_normalize
        )


class UpperTriangular(RegularizationBase):
    r"""Bias the matrix towards being upper triangular.

    .. math:: A ↦ ‖A-Π(A)‖ₚ

    where $Π(A) = \argmin_X ½‖X-A‖²$ s.t. $U⊙X = X$
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
    def forward(self, x: Tensor) -> Tensor:
        r"""Bias x towards upper triangular matrix."""
        return upper_triangular(
            x, upper=self.upper, p=self.p, size_normalize=self.size_normalize
        )


class Banded(RegularizationBase):
    r"""Bias the matrix towards being banded.

    .. math:: A ↦ ‖A-Π(A)‖ₚ

    where $Π(A) = \argmin_X ½‖X-A‖²$ s.t. $B⊙X = X$
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
    def forward(self, x: Tensor) -> Tensor:
        r"""Bias x towards banded matrix."""
        return banded(
            x,
            lower=self.lower,
            upper=self.upper,
            p=self.p,
            size_normalize=self.size_normalize,
        )


class Tridiagonal(RegularizationBase):
    r"""Bias the matrix towards being tridiagonal.

    .. math:: A ↦ ‖A-Π(A)‖ₚ

    where $Π(A)$ is the closest tridiagonal matrix to $A$.
    """

    p: Final[str | int]
    size_normalize: Final[bool]

    def __init__(self, *, p: str | int = "fro", size_normalize: bool = True) -> None:
        super().__init__()
        self.p = p
        self.size_normalize = size_normalize

    @signature("(..., m, n) -> (...)")
    def forward(self, x: Tensor) -> Tensor:
        r"""Bias x towards tridiagonal matrix."""
        return tridiagonal(x, p=self.p, size_normalize=self.size_normalize)


class Masked(RegularizationBase):
    r"""Bias the matrix towards being masked.

    .. math:: A ↦ ‖A-Π(A)‖ₚ

    where $Π(A) = \argmin_X ½‖X-A‖²$ s.t. $M⊙X = X$
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
    def forward(self, x: Tensor) -> Tensor:
        r"""Bias x towards masked matrix."""
        return masked(x, mask=self.mask, p=self.p, size_normalize=self.size_normalize)


# endregion masked projections ---------------------------------------------------------


# region other regularizations ---------------------------------------------------------
class Contraction(RegularizationBase):
    r"""Bias the matrix towards being a contraction.

    .. math:: A ↦ ‖A-Π(A)‖ₚ

    where $Π(A) = \argmin_X ‖X-A‖₂$ s.t. $‖X‖₂≤1$
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
    def forward(self, x: Tensor) -> Tensor:
        r"""Bias x towards contraction."""
        return contraction(
            x, self.lipschitz_bound, p=self.p, size_normalize=self.size_normalize
        )


class LipschitzBounded(RegularizationBase):
    r"""Bias the matrix towards having spectral norm at most γ.

    .. math:: A ↦ ‖A-Π(A)‖ₚ

    where $Π(A) = \argmin_X ‖X-A‖₂$ s.t. $‖X‖₂≤γ$
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
    def forward(self, x: Tensor) -> Tensor:
        r"""Bias x towards a Lipschitz-bounded matrix."""
        return lipschitz_bounded(
            x,
            self.lipschitz_bound,
            p=self.p,
            size_normalize=self.size_normalize,
        )


class SpectralNormalized(RegularizationBase):
    r"""Bias the matrix towards having unit spectral norm.

    .. math:: A ↦ ‖A-Π(A)‖ₚ

    where $Π(A) = \argmin_X ‖X-A‖₂$ s.t. $‖X‖₂=1$
    """

    p: Final[str | int]
    size_normalize: Final[bool]

    def __init__(self, *, p: str | int = "fro", size_normalize: bool = True) -> None:
        super().__init__()
        self.p = p
        self.size_normalize = size_normalize

    @signature("(..., m, n) -> (...)")
    def forward(self, x: Tensor) -> Tensor:
        r"""Bias x towards a spectrally normalized matrix."""
        return spectral_normalized(x, p=self.p, size_normalize=self.size_normalize)


# endregion other regularizations ------------------------------------------------------
# region vector groups -----------------------------------------------------------------
class UnitVector(RegularizationBase):
    r"""Bias the vector towards having unit norm."""

    p: Final[float]
    size_normalize: Final[bool]

    def __init__(self, *, p: float = 2.0, size_normalize: bool = True) -> None:
        super().__init__()
        self.p = p
        self.size_normalize = size_normalize

    @signature("(..., n) -> (...)")
    def forward(self, x: Tensor) -> Tensor:
        r"""Bias x towards a unit vector."""
        return unit_vector(x, p=self.p, size_normalize=self.size_normalize)


# endregion vector groups --------------------------------------------------------------
# endregion regularizations ------------------------------------------------------------
