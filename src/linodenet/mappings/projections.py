r"""Projections for the Linear ODE Networks.

Projections are mappings ϕ:X→X such that ϕ∘ϕ=ϕ.
Then the identity map on the image of ϕ is its right inverse,
i.e., ϕ∘\id_{\im ϕ} = \id_{\im ϕ}.

Notes:
    - See `linodenet.mappings.functional` for functional implementations.
    - See `linodenet.mappings.projections` for module-based implementations.
"""

__all__ = [
    # ABCs & Protocols
    # Matrix Projections
    "Banded",
    "Diagonal",
    "DiagonallyDominant",
    "LipschitzBounded",
    "SpectralNormalized",
    "Contraction",
    "Hamiltonian",
    "LowerTriangular",
    "LowRank",
    "Masked",
    "Normal",
    "Orthogonal",
    "RankOne",
    "SkewSymmetric",
    "Symmetric",
    "Symplectic",
    "Traceless",
    "Tridiagonal",
    "UpperTriangular",
    # Vector Projections
    "UnitVector",
]

from typing import ClassVar, Final, Optional

import torch
from torch import Tensor

import linodenet.mappings.functional as F
from linodenet.constants import ATOL, RTOL
from linodenet.domains import MatrixDomains, VectorDomains
from linodenet_special.fallbacks import singular_triplet
from signatures import signature

from .base import ProjectionBase


class LipschitzBounded(ProjectionBase):
    r"""Return the closest matrix to X with spectral norm (=lipschitz constant) at most γ.

    .. math:: \min_Y ‖X-Y‖₂  s.t. ‖Y‖₂ ≤ γ

    One can show analytically that the unique smallest norm minimizer is

    .. math:: f(X) = \min(1, γ/‖X‖₂)⋅X

    Args:
        lipschitz_bound: The constant γ, the transformation ensures $$.
        atol: The absolute tolerance for the power method.
        rtol: The relative tolerance for the power method.
        maxiter: The maximum number of iterations for the power method.

    Note:
        Uses a power iteration method with cached initial guesses.
        This is especially useful for parametrization, but means this method expects
        the same input shape for each forward pass.

    Note:
        For $‖A‖₂<1$, it follows that $x↦Ax$ is a contraction mapping. In particular,
        the residual mapping $x↦x ± Ax$ is invertible in this case, and the inverse
        can be computed via fixpoint iteration.

    See Also:
        - `SpectralNormalized` for the special case of γ=1.
        - `Contraction` for the special case of $0<‖Y‖₂<1$.
    """

    DOMAIN: ClassVar[MatrixDomains] = MatrixDomains.RECTANGULAR
    CODOMAIN: ClassVar[MatrixDomains] = MatrixDomains.LIPSCHITZ_BOUNDED

    sigma: Tensor | None
    r"""BUFFER: The cached singular value."""
    u: Tensor | None
    r"""BUFFER: The cached left singular vector."""
    v: Tensor | None
    r"""BUFFER: The cached right singular vector."""

    GAMMA: Tensor
    r"""CONST: The constant γ, the transformation ensures $‖A‖₂≤γ$."""
    ONE: Tensor
    r"""CONST: The constant 1."""
    maxiter: Final[Optional[int]]
    r"""CONST: The maximum number of iterations for the power method."""
    atol: Final[float]
    r"""CONST: The absolute tolerance for the power method."""
    rtol: Final[float]
    r"""CONST: The relative tolerance for the power method."""

    def __init__(
        self,
        lipschitz_bound: float,
        *,
        atol: float = ATOL,
        rtol: float = RTOL,
        maxiter: Optional[int] = None,
    ) -> None:
        super().__init__()

        # constants
        self.atol = atol
        self.rtol = rtol
        self.maxiter = maxiter

        # shape-dependent buffers are initialized lazily on first use
        self.register_buffer("sigma", None, persistent=True)
        self.register_buffer("u", None, persistent=True)
        self.register_buffer("v", None, persistent=True)
        self.register_buffer("ONE", torch.tensor(1.0), persistent=True)
        self.register_buffer(
            "GAMMA", torch.tensor(float(lipschitz_bound)), persistent=True
        )

    @signature("(..., m, n) -> (..., m, n)")
    def forward(self, x: Tensor) -> Tensor:
        r"""Perform spectral normalization w ↦ w/‖w‖₂."""
        # We use the cached singular vectors as initial guess for the power method.
        sigma, u, v = singular_triplet(
            x,
            u0=self.u,
            v0=self.v,
            atol=self.atol,
            rtol=self.rtol,
            maxiter=self.maxiter,
        )

        # store the buffers
        self.sigma = sigma
        self.u = u
        self.v = v

        # map A' ← A ⋅ min(1, γ/‖A₂‖), which is the largest value that ensures
        # ‖A'‖₂ ≤ min(γ, ‖A‖₂)
        gamma = torch.minimum(self.ONE, self.GAMMA / sigma)

        # return the parametrized weight and the cached singular triplet
        return gamma * x


class SpectralNormalized(LipschitzBounded):
    r"""Return the closest matrix to X with unit spectral norm.

    .. math:: \min_Y ‖X-Y‖₂  s.t. ‖Y‖₂ = 1

    One can show analytically that the unique smallest norm minimizer is

    .. math:: f(X) = X/‖X‖₂

    Args:
        atol: The absolute tolerance for the power method.
        rtol: The relative tolerance for the power method.
        maxiter: The maximum number of iterations for the power method.

    Note:
        Uses a power iteration method with cached initial guesses.
        This is especially useful for parametrization, but means this method expects
        the same input shape for each forward pass.

    See Also:
        - `SpectralNormBounded` for the general case $‖Y‖₂≤γ$
        - `Contraction` for the special case of $0<‖Y‖₂<1$.
    """

    DOMAIN: ClassVar[MatrixDomains] = MatrixDomains.RECTANGULAR
    CODOMAIN: ClassVar[MatrixDomains] = MatrixDomains.SPECTRAL_NORMALIZED

    def __init__(
        self, atol: float = ATOL, rtol: float = RTOL, maxiter: Optional[int] = None
    ) -> None:
        super().__init__(lipschitz_bound=1.0, atol=atol, rtol=rtol, maxiter=maxiter)

    @signature("(..., m, n) -> (..., m, n)")
    def forward(self, x: Tensor) -> Tensor:
        r"""Perform spectral normalization w ↦ w/‖w‖₂."""
        # We use the cached singular vectors as initial guess for the power method.
        sigma, u, v = singular_triplet(
            x,
            u0=self.u,
            v0=self.v,
            atol=self.atol,
            rtol=self.rtol,
            maxiter=self.maxiter,
        )

        # store the buffers
        self.sigma = sigma
        self.u = u
        self.v = v

        return x / sigma


class Contraction(LipschitzBounded):
    r"""Return the closest matrix to X with spectral norm (=lipschitz constant) at most γ<1.

    .. math:: \min_Y ‖X-Y‖₂  s.t. ‖Y‖₂ ≤ 1

    One can show analytically that the unique smallest norm minimizer is

    .. math:: f(X) = \min(1, γ/‖X‖₂)⋅X

    Args:
        lipschitz_bound: The constant γ, the transformation ensures $‖Y‖₂≤γ<1$.
        atol: The absolute tolerance for the power method.
        rtol: The relative tolerance for the power method.
        maxiter: The maximum number of iterations for the power method.

    Note:
        Uses a power iteration method with cached initial guesses.
        This is especially useful for parametrization, but means this method expects
        the same input shape for each forward pass.

    See Also:
        - `SpectralNormBounded` for the general case $‖Y‖₂≤γ$
        - `SpectralNormalized` for the special case of $‖Y‖₂=1$.
    """

    DOMAIN: ClassVar[MatrixDomains] = MatrixDomains.RECTANGULAR
    CODOMAIN: ClassVar[MatrixDomains] = MatrixDomains.CONTRACTION

    def __init__(
        self,
        lipschitz_bound: float,
        *,
        atol: float = ATOL,
        rtol: float = RTOL,
        maxiter: Optional[int] = None,
    ) -> None:
        if not 0 < lipschitz_bound < 1:
            raise ValueError("lipschitz_bound must be between 0 and 1")
        super().__init__(
            lipschitz_bound=lipschitz_bound, atol=atol, rtol=rtol, maxiter=maxiter
        )


# region projections -------------------------------------------------------------------
# region matrix groups -----------------------------------------------------------------
class Symmetric(ProjectionBase):
    r"""Return the closest symmetric matrix to X.

    .. math:: \min_Y ½‖X-Y‖² s.t. Yᵀ = Y

    One can show analytically that Y = ½(X + Xᵀ) is the unique minimizer.
    """

    DOMAIN: ClassVar[MatrixDomains] = MatrixDomains.SQUARE
    CODOMAIN: ClassVar[MatrixDomains] = MatrixDomains.SYMMETRIC

    @signature("(..., n, n) -> (..., n, n)")
    def forward(self, x: Tensor) -> Tensor:
        r"""Project into space of symmetric matrices."""
        return F.symmetric(x)


class SkewSymmetric(ProjectionBase):
    r"""Return the closest skew-symmetric matrix to X.

    .. math:: \min_Y ½‖X-Y‖² s.t. Yᵀ = -Y

    One can show analytically that Y = ½(X - Xᵀ) is the unique minimizer.
    """

    DOMAIN: ClassVar[MatrixDomains] = MatrixDomains.SQUARE
    CODOMAIN: ClassVar[MatrixDomains] = MatrixDomains.SKEW_SYMMETRIC

    @signature("(..., n, n) -> (..., n, n)")
    def forward(self, x: Tensor) -> Tensor:
        r"""Project into space of skew-symmetric matrices."""
        return F.skew_symmetric(x)


class Orthogonal(ProjectionBase):
    r"""Return the closest orthogonal matrix to X.

    .. math:: \min_Y ½‖X-Y‖² s.t. Yᵀ Y = 𝕀 = YYᵀ

    One can show analytically that $Y = UVᵀ$ is the unique minimizer,
    where $X=UΣVᵀ$ is the SVD of $X$.

    References:
        https://math.stackexchange.com/q/2215359
    """

    DOMAIN: ClassVar[MatrixDomains] = MatrixDomains.SQUARE
    CODOMAIN: ClassVar[MatrixDomains] = MatrixDomains.ORTHOGONAL

    @signature("(..., n, n) -> (..., n, n)")
    def forward(self, x: Tensor) -> Tensor:
        r"""Project into space of orthogonal matrices."""
        return F.orthogonal(x)


class Traceless(ProjectionBase):
    r"""Return the closest traceless matrix to X.

    .. math:: \min_Y ½‖X-Y‖² s.t. Yᵀ = -Y

    One can show analytically that Y = ½(X - Xᵀ) is the unique minimizer.
    """

    DOMAIN: ClassVar[MatrixDomains] = MatrixDomains.SQUARE
    CODOMAIN: ClassVar[MatrixDomains] = MatrixDomains.TRACELESS

    @signature("(..., n, n) -> (..., n, n)")
    def forward(self, x: Tensor) -> Tensor:
        r"""Project into space of traceless matrices."""
        return F.traceless(x)


class Normal(ProjectionBase):
    r"""Return the closest normal matrix to X.

    .. math:: \min_Y ½‖X-Y‖² s.t. YᵀY = YYᵀ

    **The Lagrangian:**

    .. math:: ℒ(Y, Λ) = ½‖X-Y‖² + ⟨Λ, [Y, Yᵀ]⟩

    **First order necessary KKT condition:**

    .. math::
            0 &= ∇ℒ(Y, Λ) = (Y-X) + Y(Λ + Λᵀ) - (Λ + Λᵀ)Y
        \\⟺ Y &= X + [Y, Λ]

    **Second order sufficient KKT condition:**

    .. math::
             ⟨∇h|S⟩=0     &⟹ ⟨S|∇²ℒ|S⟩ ≥ 0
         \\⟺ ⟨[Y, Λ]|S⟩=0 &⟹ ⟨S|𝕀⊗𝕀 + Λ⊗𝕀 − 𝕀⊗Λ|S⟩ ≥ 0
         \\⟺ ⟨[Y, Λ]|S⟩=0 &⟹ ⟨S|S⟩ + ⟨[S, Λ]|S⟩ ≥ 0
    """

    DOMAIN: ClassVar[MatrixDomains] = MatrixDomains.SQUARE
    CODOMAIN: ClassVar[MatrixDomains] = MatrixDomains.NORMAL

    @signature("(..., n, n) -> (..., n, n)")
    def forward(self, x: Tensor) -> Tensor:
        r"""Project into space of normal matrices."""
        return F.normal(x)


class Hamiltonian(ProjectionBase):
    r"""Return the closest hamiltonian matrix to X.

    .. math:: \min_Y ½‖X-Y‖²   s.t.   (JY)ᵀ = JA   where   J=[𝟎, 𝕀; -𝕀, 𝟎]

    Alternatively, the above is equivalent to

    .. math:: \min_Y ½‖X-Y‖²   s.t.   Yᵀ J Y = J   where   J= 𝔻₊₁-𝔻₋₁

    where $𝔻ₖ$ is the $2n×2n$ matrix with ones on the k-th diagonal.

    Note:
        The Hamiltonian matrices are the skew-symmetric matrices
        with respect to the symplectic inner product.
        - The matrix exponential of a Hamiltonian matrix is symplectic.
    """

    DOMAIN: ClassVar[MatrixDomains] = MatrixDomains.EVEN_SQUARE
    CODOMAIN: ClassVar[MatrixDomains] = MatrixDomains.HAMILTONIAN

    @signature("(..., 2n, 2n) -> (..., 2n, 2n)")
    def forward(self, x: Tensor) -> Tensor:
        r"""Project into space of hamiltonian matrices."""
        return F.hamiltonian(x)


class Symplectic(ProjectionBase):
    r"""Return the closest symplectic matrix to X.

    .. math:: \min_Y ½‖X-Y‖²   s.t.   YᵀJY = J   where   J=[𝟎, 𝕀; -𝕀, 𝟎]

    Alternatively, the above is equivalent to

    .. math:: \min_Y ½‖X-Y‖²   s.t.   YᵀJY = J   where   J= 𝔻₊₁-𝔻₋₁

    where $𝔻ₖ$ is the $2n×2n$ matrix with ones on the k-th diagonal.
    """

    DOMAIN: ClassVar[MatrixDomains] = MatrixDomains.EVEN_SQUARE
    CODOMAIN: ClassVar[MatrixDomains] = MatrixDomains.SYMPLECTIC

    @signature("(..., 2n, 2n) -> (..., 2n, 2n)")
    def forward(self, x: Tensor) -> Tensor:
        r"""Project into space of symplectic matrices."""
        return F.symplectic(x)


# endregion matrix groups --------------------------------------------------------------


# region masked projections ------------------------------------------------------------
class Diagonal(ProjectionBase):
    r"""Return the closest diagonal matrix to X.

    .. math:: \min_Y ½‖X-Y‖² s.t. Y = 𝕀⊙Y

    One can show analytically that the unique smallest norm minimizer is $Y = 𝕀⊙X$.

    See Also:
        - `projections.Masked`
        - `projections.Diagonal`
        - `projections.LowerTriangular`
        - `projections.UpperTriangular`
        - `projections.Banded`
    """

    DOMAIN: ClassVar[MatrixDomains] = MatrixDomains.SQUARE
    CODOMAIN: ClassVar[MatrixDomains] = MatrixDomains.DIAGONAL

    @signature("(..., m, n) -> (..., m, n)")
    def forward(self, x: Tensor) -> Tensor:
        r"""Project into space of diagonal matrices."""
        return F.diagonal(x)


class UpperTriangular(ProjectionBase):
    r"""Return the closest upper triangular matrix to X.

    .. math:: \min_Y ½‖X-Y‖²   s.t.   U⊙Y = Y

    One can show analytically that the unique smallest norm minimizer is $Y = U⊙X$.

    See Also:
        - `projections.Masked`
        - `projections.Diagonal`
        - `projections.LowerTriangular`
        - `projections.UpperTriangular`
        - `projections.Banded`
    """

    DOMAIN: ClassVar[MatrixDomains] = MatrixDomains.SQUARE
    CODOMAIN: ClassVar[MatrixDomains] = MatrixDomains.UPPER_TRIANGULAR

    upper: Final[int]
    r"""CONST: The diagonal to consider"""

    def __init__(self, *, upper: int = 0) -> None:
        super().__init__()
        self.upper = upper

    @signature("(..., n, n) -> (..., n, n)")
    def forward(self, x: Tensor) -> Tensor:
        r"""Project into space of upper triangular matrices."""
        return F.upper_triangular(x, upper=self.upper)


class LowerTriangular(ProjectionBase):
    r"""Return the closest lower triangular matrix to X.

    .. math:: \min_Y ½‖X-Y‖²   s.t.   𝕃⊙Y = Y

    One can show analytically that the unique smallest norm minimizer is $Y = L⊙X$.

    See Also:
        - `projections.Masked`
        - `projections.Diagonal`
        - `projections.LowerTriangular`
        - `projections.UpperTriangular`
        - `projections.Banded`
    """

    DOMAIN: ClassVar[MatrixDomains] = MatrixDomains.SQUARE
    CODOMAIN: ClassVar[MatrixDomains] = MatrixDomains.LOWER_TRIANGULAR

    lower: Final[int]
    r"""CONST: The diagonal to consider"""

    def __init__(self, *, lower: int = 0) -> None:
        super().__init__()
        self.lower = lower

    @signature("(..., n, n) -> (..., n, n)")
    def forward(self, x: Tensor) -> Tensor:
        r"""Project into space of lower triangular matrices."""
        return F.lower_triangular(x, lower=self.lower)


class Tridiagonal(ProjectionBase):
    r"""Return the closest tridiagonal matrix to X.

    .. math:: \min_Y ½‖X-Y‖² s.t. Y = T⊙Y

    One can show analytically that the unique smallest norm minimizer is
    $Y = T⊙X$.
    """

    DOMAIN: ClassVar[MatrixDomains] = MatrixDomains.SQUARE
    CODOMAIN: ClassVar[MatrixDomains] = MatrixDomains.TRIDIAGONAL

    @signature("(..., n, n) -> (..., n, n)")
    def forward(self, x: Tensor) -> Tensor:
        r"""Project into space of tridiagonal matrices."""
        return F.tridiagonal(x)


# endregion masked projections ---------------------------------------------------------


# region other projections -------------------------------------------------------------
class DiagonallyDominant(ProjectionBase):
    r"""Return the closest diagonally dominant matrix to X.

    .. math:: \min_Y ‖X-Y‖_F  s.t. |Y_{ii}| ≥ ∑_{j≠i} |Y_{ij}| for all i = 1, …, n

    References:
        Computing the nearest diagonally dominant matrix (Mendoza et al. 1998)
    """

    DOMAIN: ClassVar[MatrixDomains] = MatrixDomains.SQUARE
    CODOMAIN: ClassVar[MatrixDomains] = MatrixDomains.DIAGONALLY_DOMINANT

    @signature("(..., n, n) -> (..., n, n)")
    def forward(self, x: Tensor) -> Tensor:
        r"""Project into space of diagonally dominant matrices."""
        return F.diagonally_dominant(x)


class RankOne(ProjectionBase):
    r"""Return the closest rank-1 matrix to X.

    .. math:: \min_Y ½‖X-Y‖²   s.t.   rank(Y) ≤ 1

    This is the special case of `LowRank` with `rank=1`.
    """

    DOMAIN: ClassVar[MatrixDomains] = MatrixDomains.RECTANGULAR
    CODOMAIN: ClassVar[MatrixDomains] = MatrixDomains.RANK_ONE

    @signature("(..., m, n) -> (..., m, n)")
    def forward(self, x: Tensor) -> Tensor:
        r"""Project into space of rank-1 matrices."""
        return F.rank_one(x)


# endregion other projections ----------------------------------------------------------


# region special -----------------------------------------------------------------------
class Masked(ProjectionBase):
    r"""Return the closest banded matrix to X.

    .. math:: \min_Y ½‖X-Y‖² s.t. Y = 𝕄⊙Y

    One can show analytically that the unique smallest norm minimizer is $Y = M⊙X$.

    See Also:
        - `projections.Masked`
        - `projections.Diagonal`
        - `projections.LowerTriangular`
        - `projections.UpperTriangular`
        - `projections.Banded`
    """

    DOMAIN: ClassVar[MatrixDomains] = MatrixDomains.RECTANGULAR
    CODOMAIN: ClassVar[MatrixDomains] = MatrixDomains.MASKED

    mask: Tensor
    r"""CONST: Boolean mask to consider"""

    def __init__(self, mask: bool | Tensor) -> None:
        super().__init__()
        self.mask = torch.as_tensor(mask, dtype=torch.bool)

    @signature("(..., m, n) -> (..., m, n)")
    def forward(self, x: Tensor) -> Tensor:
        r"""Project into space of masked matrices."""
        return F.masked(x, mask=self.mask)


class LowRank(ProjectionBase):
    r"""Return the closest low rank matrix to X.

    .. math:: \min_Y ½‖X-Y‖²   s.t.   rank(Y) ≤ k

    One can show analytically that Y = UₖΣₖVₖᵀ is the unique minimizer,
    where X=UΣVᵀ is the SVD of X.
    """

    DOMAIN: ClassVar[MatrixDomains] = MatrixDomains.RECTANGULAR
    CODOMAIN: ClassVar[MatrixDomains] = MatrixDomains.LOW_RANK
    rank: Final[int]

    def __init__(self, *, rank: int = 1) -> None:
        super().__init__()
        self.rank = rank

    @signature("(..., m, n) -> (..., m, n)")
    def forward(self, x: Tensor) -> Tensor:
        r"""Project into space of low rank matrices."""
        return F.low_rank(x, rank=self.rank)


class Banded(ProjectionBase):
    r"""Return the closest banded matrix to X.

    .. math:: \min_Y ½‖X-Y‖² s.t. Y = 𝔹⊙Y

    One can show analytically that the unique smallest norm minimizer is $Y = B⊙X$.

    See Also:
        - `projections.Masked`
        - `projections.Diagonal`
        - `projections.LowerTriangular`
        - `projections.UpperTriangular`
        - `projections.Banded`
    """

    DOMAIN: ClassVar[MatrixDomains] = MatrixDomains.RECTANGULAR
    CODOMAIN: ClassVar[MatrixDomains] = MatrixDomains.BANDED

    upper: Final[int]
    r"""CONST: The upper diagonal to consider"""
    lower: Final[int]
    r"""CONST: The lower diagonal to consider"""

    def __init__(self, lower: int, upper: int) -> None:
        super().__init__()
        self.upper = upper
        self.lower = lower
        if not (lower <= 0 <= upper):
            raise ValueError(
                f"lower must be ≤ 0 and upper must be ≥ 0,"
                f" got lower={lower} and upper={upper}"
            )

    @signature("(..., m, n) -> (..., m, n)")
    def forward(self, x: Tensor) -> Tensor:
        r"""Project into space of banded matrices."""
        return F.banded(x, lower=self.lower, upper=self.upper)


class UnitVector(ProjectionBase):
    r"""Project vectors onto the unit sphere."""

    DOMAIN: Final[VectorDomains] = VectorDomains.NONZERO
    CODOMAIN: Final[VectorDomains] = VectorDomains.UNIT_VECTOR

    @signature("(..., n) -> (..., n)")
    def forward(self, x: Tensor) -> Tensor:
        return F.unit_vector(x)


# endregion special --------------------------------------------------------------------


# endregion projections ----------------------------------------------------------------
