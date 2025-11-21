r"""Projections for the Linear ODE Networks.

Projections are mappings ϕ:X→X such that ϕ∘ϕ=ϕ.
Then the identity map on the image of ϕ is its right inverse,
i.e., ϕ∘\id_{\im ϕ} = \id_{\im ϕ}.

Notes:
    - See `linodenet.projections.functional` for functional implementations.
    - See `linodenet.projections.modules` for module-based implementations.
"""

__all__ = [
    # ABCs & Protocols
    "Projection",
    "ProjectionBase",
    # Classes
    "Banded",
    "Contraction",
    "Diagonal",
    "DiagonallyDominant",
    "Hamiltonian",
    "Identity",
    "LowerTriangular",
    "LowRank",
    "Masked",
    "Normal",
    "Orthogonal",
    "SkewSymmetric",
    "Symmetric",
    "Symplectic",
    "Traceless",
    "UpperTriangular",
]

from abc import abstractmethod
from typing import Final, Protocol, runtime_checkable

import torch
from torch import Tensor, jit, nn

import linodenet.projections.functional as F
from linodenet.constants import FALSE
from linodenet.domains import MatrixDomains


@runtime_checkable
class Projection[T](Protocol):
    r"""Protocol for projections.

    A projection is a mapping $φ:X→X$ such that $φ∘φ=φ$.
    In particular, $φ=i∘π$ for the embedding $i:\Im(φ)→X$ where $π:X→\Im(φ)$ is
    $φ$ viewed as a surjection onto its image.
    Then the identity map on the image of $φ$ is the right inverse of $π$.

    References:
        - https://en.wikipedia.org/wiki/Projection_(mathematics)
        - https://en.wikipedia.org/wiki/Projection_(linear_algebra)
    """

    @abstractmethod
    def forward(self, x: T, /) -> T:
        r"""Forward pass of the projection.

        .. Signature: ``(..., *xs) -> (..., *xs)``.
        """
        ...

    def right_inverse(self, y: T, /) -> T:
        r"""Right inverse of the projection, i.e. the identity on the image.

        .. Signature: ``(..., *xs) -> (..., *xs)``.
        """
        return y


class ProjectionBase(nn.Module, Projection[Tensor]):
    r"""Abstract Base Class for Projection components."""

    @abstractmethod
    def forward(self, x: Tensor, /) -> Tensor:
        r"""Forward pass of the projection.

        .. Signature: ``(..., *xs) -> (..., *ys)``.

        Args:
            x: The input tensor to be projected.

        Returns:
            y: The projected tensor.
        """

    @jit.export
    def right_inverse(self, y: Tensor) -> Tensor:
        r"""Right inverse of the projection, i.e. the identity on the image.

        .. Signature: ``(..., *ys) -> (..., *xs)``.

        Args:
            y: The projected tensor.

        Returns:
            The input tensor as-is.
        """
        return y

    @jit.export
    def encode(self, x: Tensor) -> Tensor:
        r"""Alias for `forward` method."""
        return self.forward(x)

    @jit.export
    def decode(self, y: Tensor) -> Tensor:
        r"""Alias for `right_inverse` method."""
        return self.right_inverse(y)


# region projections -------------------------------------------------------------------
# region matrix groups -----------------------------------------------------------------
class Identity(ProjectionBase):
    r"""Return x as-is.

    .. Signature:: ``(..., n, n) -> (..., n, n)``

    .. math:: \min_Y ½∥X-Y∥_F^2
    """

    @jit.export
    def forward(self, x: Tensor) -> Tensor:
        r"""Project x into space of matrices."""
        return F.identity(x)


class Symmetric(ProjectionBase):
    r"""Return the closest symmetric matrix to X.

    .. Signature:: ``(..., n, n) -> (..., n, n)``

    .. math:: \min_Y ½∥X-Y∥_F^2 s.t. Y^⊤ = Y

    One can show analytically that Y = ½(X + X^⊤) is the unique minimizer.
    """

    DOMAIN: Final[MatrixDomains] = MatrixDomains.SQUARE
    CODOMAIN: Final[MatrixDomains] = MatrixDomains.SYMMETRIC

    @jit.export
    def forward(self, x: Tensor) -> Tensor:
        r"""Project x into space of symmetric matrices."""
        return F.symmetric(x)


class SkewSymmetric(ProjectionBase):
    r"""Return the closest skew-symmetric matrix to X.

    .. Signature:: ``(..., n, n) -> (..., n, n)``

    .. math:: \min_Y ½∥X-Y∥_F^2 s.t. Y^⊤ = -Y

    One can show analytically that Y = ½(X - X^⊤) is the unique minimizer.
    """

    DOMAIN: Final[MatrixDomains] = MatrixDomains.SQUARE
    CODOMAIN: Final[MatrixDomains] = MatrixDomains.SKEW_SYMMETRIC

    @jit.export
    def forward(self, x: Tensor) -> Tensor:
        r"""Project x into space of skew-symmetric matrices."""
        return F.skew_symmetric(x)


class Orthogonal(ProjectionBase):
    r"""Return the closest orthogonal matrix to X.

    .. Signature:: ``(..., n, n) -> (..., n, n)``

    .. math:: \min_Y ½∥X-Y∥_F^2 s.t. Y^𝖳 Y = 𝕀 = YY^𝖳

    One can show analytically that $Y = UV^𝖳$ is the unique minimizer,
    where $X=UΣV^𝖳$ is the SVD of $X$.

    References:
        https://math.stackexchange.com/q/2215359
    """

    DOMAIN: Final[MatrixDomains] = MatrixDomains.SQUARE
    CODOMAIN: Final[MatrixDomains] = MatrixDomains.ORTHOGONAL

    @jit.export
    def forward(self, x: Tensor) -> Tensor:
        r""".. Signature:: ``(..., n, n) -> (..., n, n)``."""
        return F.orthogonal(x)


class Traceless(ProjectionBase):
    r"""Return the closest traceless matrix to X.

    .. Signature:: ``(..., n, n) -> (..., n, n)``

    .. math:: \min_Y ½∥X-Y∥_F^2 s.t. Y^⊤ = -Y

    One can show analytically that Y = ½(X - X^⊤) is the unique minimizer.
    """

    DOMAIN: Final[MatrixDomains] = MatrixDomains.SQUARE
    CODOMAIN: Final[MatrixDomains] = MatrixDomains.TRACELESS

    @jit.export
    def forward(self, x: Tensor) -> Tensor:
        r""".. Signature:: ``(..., n, n) -> (..., n, n)``."""
        return F.traceless(x)


class Normal(ProjectionBase):
    r"""Return the closest normal matrix to X.

    .. Signature:: ``(..., n, n) -> (..., n, n)``

    .. math:: \min_Y ½∥X-Y∥_F^2 s.t. Y^⊤Y = YY^⊤

    **The Lagrangian:**

    .. math:: ℒ(Y, Λ) = ½∥X-Y∥_F^2 + ⟨Λ, [Y, Y^⊤]⟩

    **First order necessary KKT condition:**

    .. math::
            0 &= ∇ℒ(Y, Λ) = (Y-X) + Y(Λ + Λ^⊤) - (Λ + Λ^⊤)Y
        \\⟺ Y &= X + [Y, Λ]

    **Second order sufficient KKT condition:**

    .. math::
             ⟨∇h|S⟩=0     &⟹ ⟨S|∇²ℒ|S⟩ ≥ 0
         \\⟺ ⟨[Y, Λ]|S⟩=0 &⟹ ⟨S|𝕀⊗𝕀 + Λ⊗𝕀 − 𝕀⊗Λ|S⟩ ≥ 0
         \\⟺ ⟨[Y, Λ]|S⟩=0 &⟹ ⟨S|S⟩ + ⟨[S, Λ]|S⟩ ≥ 0
    """

    DOMAIN: Final[MatrixDomains] = MatrixDomains.SQUARE
    CODOMAIN: Final[MatrixDomains] = MatrixDomains.NORMAL

    @jit.export
    def forward(self, x: Tensor) -> Tensor:
        r""".. Signature:: ``(..., n, n) -> (..., n, n)``."""
        return F.normal(x)


class Hamiltonian(ProjectionBase):
    r"""Return the closest symplectic matrix to X.

    .. Signature:: ``(..., 2n, 2n) -> (..., 2n, 2n)``

    .. math:: \min_Y ½∥X-Y∥_F^2   s.t.   Y^𝖳 J Y = J   where   J=[𝟎, 𝕀; -𝕀, 𝟎]

    Alternatively, the above is equivalent to

    .. math:: \min_Y ½∥X-Y∥_F^2   s.t.   Y^𝖳 J Y = J   where   J= 𝔻₊₁-𝔻₋₁

    where $𝔻ₖ$ is the $2n×2n$ matrix with ones on the k-th diagonal.
    """

    DOMAIN: Final[MatrixDomains] = MatrixDomains.EVEN_SQUARE
    CODOMAIN: Final[MatrixDomains] = MatrixDomains.HAMILTONIAN

    @jit.export
    def forward(self, x: Tensor) -> Tensor:
        r""".. Signature:: ``(..., 2n, 2n) -> (..., 2n, 2n)``."""
        return F.hamiltonian(x)


class Symplectic(ProjectionBase):
    r"""Return the closest symplectic matrix to X.

    .. Signature:: ``(..., 2n, 2n) -> (..., 2n, 2n)``

    .. math:: \min_Y ½∥X-Y∥_F^2   s.t.   Y^𝖳 J Y = J   where   J=[𝟎, 𝕀; -𝕀, 𝟎]

    Alternatively, the above is equivalent to

    .. math:: \min_Y ½∥X-Y∥_F^2   s.t.   Y^𝖳 J Y = J   where   J= 𝔻₊₁-𝔻₋₁

    where $𝔻ₖ$ is the $2n×2n$ matrix with ones on the k-th diagonal.
    """

    DOMAIN: Final[MatrixDomains] = MatrixDomains.EVEN_SQUARE
    CODOMAIN: Final[MatrixDomains] = MatrixDomains.SYMPLECTIC

    @jit.export
    def forward(self, x: Tensor) -> Tensor:
        r""".. Signature:: ``(..., 2n, 2n) -> (..., 2n, 2n)``."""
        return F.symplectic(x)


# endregion matrix groups --------------------------------------------------------------


# region masked projections ------------------------------------------------------------
class Diagonal(ProjectionBase):
    r"""Return the closest diagonal matrix to X.

    .. Signature:: ``(..., n, n) -> (..., n, n)``

    .. math:: \min_Y ½∥X-Y∥_F^2 s.t. Y = 𝕀⊙Y

    One can show analytically that the unique smallest norm minimizer is $Y = 𝕀⊙X$.

    See Also:
        - `projections.Masked`
        - `projections.Diagonal`
        - `projections.LowerTriangular`
        - `projections.UpperTriangular`
        - `projections.Banded`
    """

    DOMAIN: Final[MatrixDomains] = MatrixDomains.SQUARE
    CODOMAIN: Final[MatrixDomains] = MatrixDomains.DIAGONAL

    @jit.export
    def forward(self, x: Tensor) -> Tensor:
        r""".. Signature:: ``(..., m, n) -> (..., m, n)``."""
        return F.diagonal(x)


class UpperTriangular(ProjectionBase):
    r"""Return the closest upper triangular matrix to X.

    .. Signature:: ``(..., m, n) -> (..., m, n)``

    .. math:: \min_Y ½∥X-Y∥_F^2   s.t.   𝕌⊙Y = Y

    One can show analytically that the unique smallest norm minimizer is $Y = 𝕌⊙X$.

    See Also:
        - `projections.Masked`
        - `projections.Diagonal`
        - `projections.LowerTriangular`
        - `projections.UpperTriangular`
        - `projections.Banded`
    """

    DOMAIN: Final[MatrixDomains] = MatrixDomains.GENERAL
    CODOMAIN: Final[MatrixDomains] = MatrixDomains.UPPER_TRIANGULAR

    upper: Final[int]
    r"""CONST: The diagonal to consider"""

    def __init__(self, *, upper: int = 0) -> None:
        super().__init__()
        self.upper = upper

    @jit.export
    def forward(self, x: Tensor) -> Tensor:
        r""".. Signature:: ``(..., m, n) -> (..., m, n)``."""
        return F.upper_triangular(x, upper=self.upper)


class LowerTriangular(ProjectionBase):
    r"""Return the closest lower triangular matrix to X.

    .. Signature:: ``(..., m, n) -> (..., m, n)``

    .. math:: \min_Y ½∥X-Y∥_F^2   s.t.   𝕃⊙Y = Y

    One can show analytically that the unique smallest norm minimizer is $Y = 𝕃⊙X$.

    See Also:
        - `projections.Masked`
        - `projections.Diagonal`
        - `projections.LowerTriangular`
        - `projections.UpperTriangular`
        - `projections.Banded`
    """

    DOMAIN: Final[MatrixDomains] = MatrixDomains.GENERAL
    CODOMAIN: Final[MatrixDomains] = MatrixDomains.LOWER_TRIANGULAR

    lower: Final[int]
    r"""CONST: The diagonal to consider"""

    def __init__(self, *, lower: int = 0) -> None:
        super().__init__()
        self.lower = lower

    @jit.export
    def forward(self, x: Tensor) -> Tensor:
        r""".. Signature:: ``(..., m, n) -> (..., m, n)``."""
        return F.lower_triangular(x, lower=self.lower)


class Banded(ProjectionBase):
    r"""Return the closest banded matrix to X.

    .. Signature:: ``(..., n, n) -> (..., n, n)``

    .. math:: \min_Y ½∥X-Y∥_F^2 s.t. Y = 𝔹⊙Y

    One can show analytically that the unique smallest norm minimizer is $Y = 𝔹⊙X$.

    See Also:
        - `projections.Masked`
        - `projections.Diagonal`
        - `projections.LowerTriangular`
        - `projections.UpperTriangular`
        - `projections.Banded`
    """

    DOMAIN: Final[MatrixDomains] = MatrixDomains.GENERAL
    CODOMAIN: Final[MatrixDomains] = MatrixDomains.BANDED

    upper: Final[int]
    r"""CONST: The upper diagonal to consider"""
    lower: Final[int]
    r"""CONST: The lower diagonal to consider"""

    def __init__(self, *, upper: int = 0, lower: int = 0) -> None:
        super().__init__()
        self.upper = upper
        self.lower = lower

    @jit.export
    def forward(self, x: Tensor) -> Tensor:
        r""".. Signature:: ``(..., m, n) -> (..., m, n)``."""
        return F.banded(x, upper=self.upper, lower=self.lower)


class Masked(ProjectionBase):
    r"""Return the closest banded matrix to X.

    .. Signature:: ``(..., n, n) -> (..., n, n)``

    .. math:: \min_Y ½∥X-Y∥_F^2 s.t. Y = 𝕄⊙Y

    One can show analytically that the unique smallest norm minimizer is $Y = 𝕄⊙X$.

    See Also:
        - `projections.Masked`
        - `projections.Diagonal`
        - `projections.LowerTriangular`
        - `projections.UpperTriangular`
        - `projections.Banded`
    """

    DOMAIN: Final[MatrixDomains] = MatrixDomains.GENERAL
    CODOMAIN: Final[MatrixDomains] = MatrixDomains.MASKED

    mask: Tensor
    r"""CONST: Boolean mask to consider"""

    def __init__(self, mask: bool | Tensor = FALSE) -> None:
        super().__init__()
        self.mask = torch.as_tensor(mask, dtype=torch.bool)

    @jit.export
    def forward(self, x: Tensor) -> Tensor:
        r""".. Signature:: ``(..., m, n) -> (..., m, n)``."""
        return F.masked(x, mask=self.mask)


# endregion masked projections ---------------------------------------------------------


# region other projections -------------------------------------------------------------
class DiagonallyDominant(ProjectionBase):
    r"""Return the closest diagonally dominant matrix to X.

    .. Signature:: ``(..., n, n) -> (..., n, n)``

    .. math:: \min_Y ∥X-Y∥_F  s.t. |Y_{ii}| ≥ ∑_{j≠i} |Y_{ij}| for all i = 1, …, n

    References:
        Computing the nearest diagonally dominant matrix (Mendoza et al. 1998)
    """

    DOMAIN: Final[MatrixDomains] = MatrixDomains.SQUARE
    CODOMAIN: Final[MatrixDomains] = MatrixDomains.SYMMETRIC

    @jit.export
    def forward(self, x: Tensor) -> Tensor:
        r""".. Signature:: ``(..., n, n) -> (..., n, n)``."""
        return F.diagonally_dominant(x)


class Contraction(ProjectionBase):
    r"""Return the closest contraction matrix to X.

    .. Signature:: ``(..., m, n) -> (..., m, n)``

    .. math:: \min_Y ∥X-Y∥₂  s.t. ‖Y‖₂ ≤ 1

    One can show analytically that the unique smallest norm minimizer is
    $Y = \min(1, σ⁻¹) X$, where $σ = ‖X‖₂$ is the spectral norm of $X$.

    See Also:
        - `projections.functional.contraction`
    """

    DOMAIN: Final[MatrixDomains] = MatrixDomains.GENERAL
    CODOMAIN: Final[MatrixDomains] = MatrixDomains.GENERAL

    @jit.export
    def forward(self, x: Tensor) -> Tensor:
        r"""Project x into space of contraction matrices."""
        return F.contraction(x)


class LowRank(ProjectionBase):
    r"""Return the closest low rank matrix to X.

    .. Signature:: ``(..., m, n) -> (..., m, n)``

    .. math:: \min_Y ½∥X-Y∥_F^2   s.t.   rank(Y) ≤ k

    One can show analytically that Y = UₖΣₖVₖ^𝖳 is the unique minimizer,
    where X=UΣV^𝖳 is the SVD of X.
    """

    DOMAIN: Final[MatrixDomains] = MatrixDomains.GENERAL
    CODOMAIN: Final[MatrixDomains] = MatrixDomains.LOW_RANK
    rank: Final[int]

    def __init__(self, *, rank: int = 1) -> None:
        super().__init__()
        self.rank = rank

    @jit.export
    def forward(self, x: Tensor) -> Tensor:
        r""".. Signature:: ``(..., m, n) -> (..., m, n)``."""
        return F.low_rank(x, rank=self.rank)


# endregion other projections ----------------------------------------------------------
# endregion projections ----------------------------------------------------------------
