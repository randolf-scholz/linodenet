r"""Module-based projections for vectors and matrices.

Projections are mappings $ϕ:X→X$ such that $ϕ∘ϕ=ϕ$. Viewed as a surjection
onto its image, a projection has the identity inclusion as its right inverse.

Notes:
    - See `imtskit.mappings.functional` for functional implementations.
"""

__all__ = [
    # ABCs & Protocols
    # Matrix Projections
    "Banded",
    "Diagonal",
    "DiagonallyDominant",
    "LipschitzBounded",
    "SpectrallyNormalized",
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
from torch import Tensor, nn

import imtskit.mappings.functional as F
from imtskit.constants import ATOL, RTOL
from imtskit.domains import MatrixDomains, VectorDomains
from imtskit.special import singular_triplet
from signatures import signature

from .abstract import Projection


class LipschitzBounded(nn.Module, Projection):
    r"""Return the closest matrix to X with spectral norm at most $γ$.

    $$  \min_Y ‖X-Y‖₂ \qq{subject to} ‖Y‖₂ ≤ γ  $$

    A spectral-norm minimizer is $f(X) = \min(1, γ/‖X‖₂)⋅X$.

    Args:
        lipschitz_bound: The bound $γ≥0$, ensuring $‖Y‖₂≤γ$.
        atol: The absolute tolerance for the power method.
        rtol: The relative tolerance for the power method.
        maxiter: The maximum number of iterations for the power method.

    Note:
        Uses a power iteration method with cached initial guesses.
        This is especially useful for parametrization, but means this method expects
        the same input shape for each forward pass.

        The bound is enforced using a power-iteration estimate of the spectral norm,
        so its accuracy depends on convergence and the configured tolerances.

    Note:
        For $‖A‖₂<1$, it follows that $x↦Ax$ is a contraction mapping. In particular,
        the residual mapping $x↦x ± Ax$ is invertible in this case, and the inverse
        can be computed via fixed-point iteration.

    See Also:
        - `SpectrallyNormalized` for normalization to $‖Y‖₂=1$.
        - `Contraction` for the special case of $‖Y‖₂≤γ<1$.
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
    def forward(self, x: Tensor, /) -> Tensor:
        r"""Bound the spectral norm via $X ↦ \min(1,γ/‖X‖₂)⋅X$."""
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


class SpectrallyNormalized(LipschitzBounded):
    r"""Return the closest matrix to X with unit spectral norm.

    $$  \min_Y ‖X-Y‖₂ \qq{subject to} ‖Y‖₂ = 1  $$

    A spectral-norm minimizer is $f(X) = X/‖X‖₂$.

    Args:
        atol: The absolute tolerance for the power method.
        rtol: The relative tolerance for the power method.
        maxiter: The maximum number of iterations for the power method.

    Note:
        Uses a power iteration method with cached initial guesses.
        This is especially useful for parametrization, but means this method expects
        the same input shape for each forward pass.

        The input must be nonzero; normalization is undefined for the zero matrix.

    See Also:
        - `LipschitzBounded` for the general case $‖Y‖₂≤γ$
        - `Contraction` for the special case of $‖Y‖₂≤γ<1$.
    """

    DOMAIN: ClassVar[MatrixDomains] = MatrixDomains.RECTANGULAR
    CODOMAIN: ClassVar[MatrixDomains] = MatrixDomains.SPECTRALLY_NORMALIZED

    def __init__(
        self, atol: float = ATOL, rtol: float = RTOL, maxiter: Optional[int] = None
    ) -> None:
        super().__init__(lipschitz_bound=1.0, atol=atol, rtol=rtol, maxiter=maxiter)

    @signature("(..., m, n) -> (..., m, n)")
    def forward(self, x: Tensor, /) -> Tensor:
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
    r"""Return the closest matrix to X with spectral norm at most $0<γ<1$.

    $$  \min_Y ‖X-Y‖₂ \qq{subject to} ‖Y‖₂ ≤ γ < 1  $$

    A spectral-norm minimizer is $f(X) = \min(1, γ/‖X‖₂)⋅X$.

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
        - `LipschitzBounded` for the general case $‖Y‖₂≤γ$
        - `SpectrallyNormalized` for the special case of $‖Y‖₂=1$.
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
class Symmetric(nn.Module, Projection):
    r"""Return the closest symmetric matrix to X.

    $$  \min_Y ½‖X-Y‖_F² \qq{subject to} Yᵀ = Y  $$

    The unique minimizer is $Y=½(X+Xᵀ)$.
    """

    DOMAIN: ClassVar[MatrixDomains] = MatrixDomains.SQUARE
    CODOMAIN: ClassVar[MatrixDomains] = MatrixDomains.SYMMETRIC

    @signature("(..., n, n) -> (..., n, n)")
    def forward(self, x: Tensor, /) -> Tensor:
        r"""Project into space of symmetric matrices."""
        return F.symmetric(x)


class SkewSymmetric(nn.Module, Projection):
    r"""Return the closest skew-symmetric matrix to X.

    $$  \min_Y ½‖X-Y‖_F² \qq{subject to} Yᵀ = -Y  $$

    The unique minimizer is $Y=½(X-Xᵀ)$.
    """

    DOMAIN: ClassVar[MatrixDomains] = MatrixDomains.SQUARE
    CODOMAIN: ClassVar[MatrixDomains] = MatrixDomains.SKEW_SYMMETRIC

    @signature("(..., n, n) -> (..., n, n)")
    def forward(self, x: Tensor, /) -> Tensor:
        r"""Project into space of skew-symmetric matrices."""
        return F.skew_symmetric(x)


class Orthogonal(nn.Module, Projection):
    r"""Return the closest orthogonal matrix to X.

    $$  \min_Y ½‖X-Y‖_F² \qq{subject to} Yᵀ Y = 𝕀 = YYᵀ  $$

    An SVD $X=UΣVᵀ$ gives a minimizer $Y=UVᵀ$. The minimizer is
    unique when $X$ is nonsingular.

    References:
        - [Nearest orthogonal matrix discussion](https://math.stackexchange.com/q/2215359)
    """

    DOMAIN: ClassVar[MatrixDomains] = MatrixDomains.SQUARE
    CODOMAIN: ClassVar[MatrixDomains] = MatrixDomains.ORTHOGONAL

    @signature("(..., n, n) -> (..., n, n)")
    def forward(self, x: Tensor, /) -> Tensor:
        r"""Project into space of orthogonal matrices."""
        return F.orthogonal(x)


class Traceless(nn.Module, Projection):
    r"""Return the closest traceless matrix to $X$ in Frobenius norm.

    $$  \min_Y ½‖X-Y‖_F² \qq{subject to} \tr(Y)=0  $$

    The unique minimizer is $Y=X-\tr(X)𝕀ₙ/n$.
    """

    DOMAIN: ClassVar[MatrixDomains] = MatrixDomains.SQUARE
    CODOMAIN: ClassVar[MatrixDomains] = MatrixDomains.TRACELESS

    @signature("(..., n, n) -> (..., n, n)")
    def forward(self, x: Tensor, /) -> Tensor:
        r"""Project into space of traceless matrices."""
        return F.traceless(x)


class Normal(nn.Module, Projection):
    r"""Describe the nearest-normal-matrix problem in Frobenius norm.

    $$  \min_Y ½‖X-Y‖_F² \qq{subject to} YYᵀ=YᵀY  $$

    The Lagrangian is

    $$  ℒ(Y,Λ)=½‖X-Y‖_F²+⟨Λ,YYᵀ-YᵀY⟩  $$

    For a stationary point admitting a Lagrange multiplier, the first-order
    condition is

    $$  0=Y-X+(Λ+Λᵀ)Y-Y(Λ+Λᵀ)  $$

    For a perturbation $S$ satisfying the linearized constraint
    $SYᵀ+YSᵀ-SᵀY-YᵀS=0$, the second variation is

    $$  ∂²ℒ(Y,Λ)[S,S]=‖S‖_F²+2⟨Λ,SSᵀ-SᵀS⟩  $$

    Nonnegativity is a second-order necessary condition under the usual
    constraint regularity assumptions; it is not a sufficient condition.

    Raises:
        NotImplementedError: The nearest-normal-matrix solver is not implemented.
    """

    DOMAIN: ClassVar[MatrixDomains] = MatrixDomains.SQUARE
    CODOMAIN: ClassVar[MatrixDomains] = MatrixDomains.NORMAL

    @signature("(..., n, n) -> (..., n, n)")
    def forward(self, x: Tensor, /) -> Tensor:
        r"""Project into space of normal matrices."""
        return F.normal(x)


class Hamiltonian(nn.Module, Projection):
    r"""Describe the nearest-Hamiltonian-matrix problem in Frobenius norm.

    $$  \min_Y ½‖X-Y‖_F² \quad \text{subject to } (JY)ᵀ=JY  $$

    Here $J$ is the canonical symplectic matrix

    $$  J=\begin{bmatrix}𝟎&𝕀ₙ\\-𝕀ₙ&𝟎\end{bmatrix}  $$

    Equivalently, $YᵀJ+JY=0$. The exponential of a Hamiltonian matrix is
    symplectic.

    Raises:
        NotImplementedError: The Hamiltonian projection is not implemented.
    """

    DOMAIN: ClassVar[MatrixDomains] = MatrixDomains.EVEN_SQUARE
    CODOMAIN: ClassVar[MatrixDomains] = MatrixDomains.HAMILTONIAN

    @signature("(..., 2n, 2n) -> (..., 2n, 2n)")
    def forward(self, x: Tensor, /) -> Tensor:
        r"""Project into space of hamiltonian matrices."""
        return F.hamiltonian(x)


class Symplectic(nn.Module, Projection):
    r"""Describe the nearest-symplectic-matrix problem in Frobenius norm.

    $$  \min_Y ½‖X-Y‖_F² \quad \text{subject to } YᵀJY=J  $$

    Here $J$ is the canonical symplectic matrix

    $$  J=\begin{bmatrix}𝟎&𝕀ₙ\\-𝕀ₙ&𝟎\end{bmatrix}  $$

    Raises:
        NotImplementedError: The symplectic projection is not implemented.
    """

    DOMAIN: ClassVar[MatrixDomains] = MatrixDomains.EVEN_SQUARE
    CODOMAIN: ClassVar[MatrixDomains] = MatrixDomains.SYMPLECTIC

    @signature("(..., 2n, 2n) -> (..., 2n, 2n)")
    def forward(self, x: Tensor, /) -> Tensor:
        r"""Project into space of symplectic matrices."""
        return F.symplectic(x)


# endregion matrix groups --------------------------------------------------------------


# region masked projections ------------------------------------------------------------
class Diagonal(nn.Module, Projection):
    r"""Return the closest diagonal matrix to X.

    $$  \min_Y ½‖X-Y‖_F² \qq{subject to} Y = 𝕀⊙Y  $$

    The unique minimizer is $Y = 𝕀⊙X$.

    See Also:
        - `imtskit.mappings.projections.Masked`
        - `imtskit.mappings.projections.Diagonal`
        - `imtskit.mappings.projections.LowerTriangular`
        - `imtskit.mappings.projections.UpperTriangular`
        - `imtskit.mappings.projections.Banded`
    """

    DOMAIN: ClassVar[MatrixDomains] = MatrixDomains.SQUARE
    CODOMAIN: ClassVar[MatrixDomains] = MatrixDomains.DIAGONAL

    @signature("(..., m, n) -> (..., m, n)")
    def forward(self, x: Tensor, /) -> Tensor:
        r"""Project into space of diagonal matrices."""
        return F.diagonal(x)


class UpperTriangular(nn.Module, Projection):
    r"""Return the closest upper triangular matrix to X.

    $$  \min_Y ½‖X-Y‖_F² \quad \text{subject to } U⊙Y = Y  $$

    The unique minimizer is $Y = U⊙X$.

    See Also:
        - `imtskit.mappings.projections.Masked`
        - `imtskit.mappings.projections.Diagonal`
        - `imtskit.mappings.projections.LowerTriangular`
        - `imtskit.mappings.projections.UpperTriangular`
        - `imtskit.mappings.projections.Banded`
    """

    DOMAIN: ClassVar[MatrixDomains] = MatrixDomains.SQUARE
    CODOMAIN: ClassVar[MatrixDomains] = MatrixDomains.UPPER_TRIANGULAR

    upper: Final[int]
    r"""CONST: The smallest retained diagonal offset."""

    def __init__(self, *, upper: int = 0) -> None:
        super().__init__()
        self.upper = upper

    @signature("(..., n, n) -> (..., n, n)")
    def forward(self, x: Tensor, /) -> Tensor:
        r"""Project into space of upper triangular matrices."""
        return F.upper_triangular(x, upper=self.upper)


class LowerTriangular(nn.Module, Projection):
    r"""Return the closest lower triangular matrix to X.

    $$  \min_Y ½‖X-Y‖_F² \quad \text{subject to } L⊙Y = Y  $$

    The unique minimizer is $Y = L⊙X$.

    See Also:
        - `imtskit.mappings.projections.Masked`
        - `imtskit.mappings.projections.Diagonal`
        - `imtskit.mappings.projections.LowerTriangular`
        - `imtskit.mappings.projections.UpperTriangular`
        - `imtskit.mappings.projections.Banded`
    """

    DOMAIN: ClassVar[MatrixDomains] = MatrixDomains.SQUARE
    CODOMAIN: ClassVar[MatrixDomains] = MatrixDomains.LOWER_TRIANGULAR

    lower: Final[int]
    r"""CONST: The largest retained diagonal offset."""

    def __init__(self, *, lower: int = 0) -> None:
        super().__init__()
        self.lower = lower

    @signature("(..., n, n) -> (..., n, n)")
    def forward(self, x: Tensor, /) -> Tensor:
        r"""Project into space of lower triangular matrices."""
        return F.lower_triangular(x, lower=self.lower)


class Tridiagonal(nn.Module, Projection):
    r"""Return the closest tridiagonal matrix to X.

    $$  \min_Y ½‖X-Y‖_F² \qq{subject to} Y = T⊙Y  $$

    The unique minimizer is
    $Y = T⊙X$.
    """

    DOMAIN: ClassVar[MatrixDomains] = MatrixDomains.SQUARE
    CODOMAIN: ClassVar[MatrixDomains] = MatrixDomains.TRIDIAGONAL

    @signature("(..., n, n) -> (..., n, n)")
    def forward(self, x: Tensor, /) -> Tensor:
        r"""Project into space of tridiagonal matrices."""
        return F.tridiagonal(x)


# endregion masked projections ---------------------------------------------------------


# region other projections -------------------------------------------------------------
class DiagonallyDominant(nn.Module, Projection):
    r"""Describe the nearest-diagonally-dominant-matrix problem.

    $$  \min_Y ‖X-Y‖_F \qq{subject to} |Yᵢᵢ| ≥ ∑_{j≠i} |Yᵢⱼ| \qq{for all} i=1,…,n  $$

    References:
        Computing the nearest diagonally dominant matrix (Mendoza et al. 1998)

    Raises:
        NotImplementedError: The diagonally dominant projection is not implemented.
    """

    DOMAIN: ClassVar[MatrixDomains] = MatrixDomains.SQUARE
    CODOMAIN: ClassVar[MatrixDomains] = MatrixDomains.DIAGONALLY_DOMINANT

    @signature("(..., n, n) -> (..., n, n)")
    def forward(self, x: Tensor, /) -> Tensor:
        r"""Project into space of diagonally dominant matrices."""
        return F.diagonally_dominant(x)


class RankOne(nn.Module, Projection):
    r"""Return the closest matrix of rank at most one to X.

    $$  \min_Y ½‖X-Y‖_F² \qq{subject to} \rank(Y) ≤ 1  $$

    This is the special case of `LowRank` with `rank=1`.
    """

    DOMAIN: ClassVar[MatrixDomains] = MatrixDomains.RECTANGULAR
    CODOMAIN: ClassVar[MatrixDomains] = MatrixDomains.RANK_ONE

    @signature("(..., m, n) -> (..., m, n)")
    def forward(self, x: Tensor, /) -> Tensor:
        r"""Project into space of rank-1 matrices."""
        return F.rank_one(x)


# endregion other projections ----------------------------------------------------------


# region special -----------------------------------------------------------------------
class Masked(nn.Module, Projection):
    r"""Return the closest matrix supported on the Boolean mask to X.

    $$  \min_Y ½‖X-Y‖_F² \qq{subject to} Y = M⊙Y  $$

    The unique minimizer is $Y = M⊙X$.

    See Also:
        - `imtskit.mappings.projections.Masked`
        - `imtskit.mappings.projections.Diagonal`
        - `imtskit.mappings.projections.LowerTriangular`
        - `imtskit.mappings.projections.UpperTriangular`
        - `imtskit.mappings.projections.Banded`
    """

    DOMAIN: ClassVar[MatrixDomains] = MatrixDomains.RECTANGULAR
    CODOMAIN: ClassVar[MatrixDomains] = MatrixDomains.MASKED

    mask: Tensor
    r"""CONST: Boolean mask defining the retained entries."""

    def __init__(self, mask: bool | Tensor) -> None:
        super().__init__()
        self.mask = torch.as_tensor(mask, dtype=torch.bool)

    @signature("(..., m, n) -> (..., m, n)")
    def forward(self, x: Tensor, /) -> Tensor:
        r"""Project into space of masked matrices."""
        return F.masked(x, mask=self.mask)


class LowRank(nn.Module, Projection):
    r"""Return the closest low rank matrix to X.

    $$  \min_Y ½‖X-Y‖_F² \qq{subject to} \rank(Y) ≤ k $$

    An SVD $X=UΣVᵀ$ gives a minimizer $Y=UₖΣₖVₖᵀ$. If the truncation
    cuts through equal nonzero singular values, the minimizer need not be unique.
    """

    DOMAIN: ClassVar[MatrixDomains] = MatrixDomains.RECTANGULAR
    CODOMAIN: ClassVar[MatrixDomains] = MatrixDomains.LOW_RANK
    rank: Final[int]

    def __init__(self, *, rank: int = 1) -> None:
        super().__init__()
        self.rank = rank

    @signature("(..., m, n) -> (..., m, n)")
    def forward(self, x: Tensor, /) -> Tensor:
        r"""Project into space of low rank matrices."""
        return F.low_rank(x, rank=self.rank)


class Banded(nn.Module, Projection):
    r"""Return the closest banded matrix to X.

    $$  \min_Y ½‖X-Y‖_F² \qq{subject to} Y = B⊙Y  $$

    The unique minimizer is $Y = B⊙X$.

    See Also:
        - `imtskit.mappings.projections.Masked`
        - `imtskit.mappings.projections.Diagonal`
        - `imtskit.mappings.projections.LowerTriangular`
        - `imtskit.mappings.projections.UpperTriangular`
        - `imtskit.mappings.projections.Banded`
    """

    DOMAIN: ClassVar[MatrixDomains] = MatrixDomains.RECTANGULAR
    CODOMAIN: ClassVar[MatrixDomains] = MatrixDomains.BANDED

    upper: Final[int]
    r"""CONST: The largest retained diagonal offset."""
    lower: Final[int]
    r"""CONST: The smallest retained diagonal offset."""

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
    def forward(self, x: Tensor, /) -> Tensor:
        r"""Project into space of banded matrices."""
        return F.banded(x, lower=self.lower, upper=self.upper)


class UnitVector(nn.Module, Projection):
    r"""Project vectors onto the unit sphere.

    The input must be nonzero; normalization is undefined for the zero vector.
    """

    DOMAIN: Final[VectorDomains] = VectorDomains.NONZERO
    CODOMAIN: Final[VectorDomains] = VectorDomains.UNIT_VECTOR

    @signature("(..., n) -> (..., n)")
    def forward(self, x: Tensor, /) -> Tensor:
        return F.unit_vector(x)


# endregion special --------------------------------------------------------------------


# endregion projections ----------------------------------------------------------------
