r"""Projections for the Linear ODE Networks.

Notes:
    - See `linodenet.projections.functional` for functional implementations.
    - See `linodenet.projections.modules` for module-based implementations.
"""

__all__ = [
    # ABCs & Protocols
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
from typing import Final

import torch
from torch import Tensor, jit, nn

from linodenet.constants import TRUE
from linodenet.projections.functional import (
    banded,
    contraction,
    diagonal,
    diagonally_dominant,
    hamiltonian,
    identity,
    low_rank,
    lower_triangular,
    masked,
    normal,
    orthogonal,
    skew_symmetric,
    symmetric,
    symplectic,
    traceless,
    upper_triangular,
)
from linodenet.types import BoolTensor


class ProjectionBase(nn.Module):
    r"""Abstract Base Class for Projection components."""

    @abstractmethod
    def forward(self, z: Tensor, /) -> Tensor:
        r"""Forward pass of the projection.

        .. Signature: ``(..., d) -> (..., f)``.

        Args:
            z: The input tensor to be projected.

        Returns:
            x: The projected tensor.
        """


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
        return identity(x)


class Symmetric(ProjectionBase):
    r"""Return the closest symmetric matrix to X.

    .. Signature:: ``(..., n, n) -> (..., n, n)``

    .. math:: \min_Y ½∥X-Y∥_F^2 s.t. Y^⊤ = Y

    One can show analytically that Y = ½(X + X^⊤) is the unique minimizer.
    """

    @jit.export
    def forward(self, x: Tensor) -> Tensor:
        r"""Project x into space of symmetric matrices."""
        return symmetric(x)


class SkewSymmetric(ProjectionBase):
    r"""Return the closest skew-symmetric matrix to X.

    .. Signature:: ``(..., n, n) -> (..., n, n)``

    .. math:: \min_Y ½∥X-Y∥_F^2 s.t. Y^⊤ = -Y

    One can show analytically that Y = ½(X - X^⊤) is the unique minimizer.
    """

    @jit.export
    def forward(self, x: Tensor) -> Tensor:
        r"""Project x into space of skew-symmetric matrices."""
        return skew_symmetric(x)


class LowRank(ProjectionBase):
    r"""Return the closest low rank matrix to X.

    .. Signature:: ``(..., m, n) -> (..., m, n)``

    .. math:: \min_Y ½∥X-Y∥_F^2   s.t.   rank(Y) ≤ k

    One can show analytically that Y = UₖΣₖVₖ^𝖳 is the unique minimizer,
    where X=UΣV^𝖳 is the SVD of X.
    """

    rank: Final[int]

    def __init__(self, *, rank: int = 1) -> None:
        super().__init__()
        self.rank = rank

    @jit.export
    def forward(self, x: Tensor) -> Tensor:
        r"""Project x into space of low rank matrices."""
        return low_rank(x, rank=self.rank)


class Orthogonal(ProjectionBase):
    r"""Return the closest orthogonal matrix to X.

    .. Signature:: ``(..., n, n) -> (..., n, n)``

    .. math:: \min_Y ½∥X-Y∥_F^2 s.t. Y^𝖳 Y = 𝕀 = YY^𝖳

    One can show analytically that $Y = UV^𝖳$ is the unique minimizer,
    where $X=UΣV^𝖳$ is the SVD of $X$.

    References:
        https://math.stackexchange.com/q/2215359
    """

    @jit.export
    def forward(self, x: Tensor) -> Tensor:
        r"""Project x into space of orthogonal matrices."""
        return orthogonal(x)


class Traceless(ProjectionBase):
    r"""Return the closest traceless matrix to X.

    .. Signature:: ``(..., n, n) -> (..., n, n)``

    .. math:: \min_Y ½∥X-Y∥_F^2 s.t. Y^⊤ = -Y

    One can show analytically that Y = ½(X - X^⊤) is the unique minimizer.
    """

    @jit.export
    def forward(self, x: Tensor) -> Tensor:
        r"""Project x into space of traceless matrices."""
        return traceless(x)


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

    @jit.export
    def forward(self, x: Tensor) -> Tensor:
        r"""Project x into space of normal matrices."""
        return normal(x)


class Hamiltonian(ProjectionBase):
    r"""Return the closest symplectic matrix to X.

    .. Signature:: ``(..., 2n, 2n) -> (..., 2n, 2n)``

    .. math:: \min_Y ½∥X-Y∥_F^2   s.t.   Y^𝖳 J Y = J   where   J=[𝟎, 𝕀; -𝕀, 𝟎]

    Alternatively, the above is equivalent to

    .. math:: \min_Y ½∥X-Y∥_F^2   s.t.   Y^𝖳 J Y = J   where   J= 𝔻₊₁-𝔻₋₁

    where $𝔻ₖ$ is the $2n×2n$ matrix with ones on the k-th diagonal.
    """

    def forward(self, x: Tensor) -> Tensor:
        return hamiltonian(x)


class Symplectic(ProjectionBase):
    r"""Return the closest symplectic matrix to X.

    .. Signature:: ``(..., 2n, 2n) -> (..., 2n, 2n)``

    .. math:: \min_Y ½∥X-Y∥_F^2   s.t.   Y^𝖳 J Y = J   where   J=[𝟎, 𝕀; -𝕀, 𝟎]

    Alternatively, the above is equivalent to

    .. math:: \min_Y ½∥X-Y∥_F^2   s.t.   Y^𝖳 J Y = J   where   J= 𝔻₊₁-𝔻₋₁

    where $𝔻ₖ$ is the $2n×2n$ matrix with ones on the k-th diagonal.
    """

    def forward(self, x: Tensor) -> Tensor:
        return symplectic(x)


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

    @jit.export
    def forward(self, x: Tensor) -> Tensor:
        r"""Project x into space of diagonal matrices."""
        return diagonal(x)


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

    upper: Final[int]
    lower: Final[int]

    def __init__(self, upper: int = 0, lower: int = 0) -> None:
        super().__init__()
        self.upper = upper
        self.lower = upper if lower is None else lower

    @jit.export
    def forward(self, x: Tensor) -> Tensor:
        r"""Project x into space of banded matrices."""
        return banded(x, upper=self.upper, lower=self.lower)


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

    upper: Final[int]

    def __init__(self, upper: int = 0) -> None:
        super().__init__()
        self.upper = upper

    @jit.export
    def forward(self, x: Tensor) -> Tensor:
        r"""Project x into space of upper triangular matrices."""
        return upper_triangular(x, upper=self.upper)


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

    lower: Final[int]

    def __init__(self, lower: int = 0) -> None:
        super().__init__()
        self.lower = lower

    @jit.export
    def forward(self, x: Tensor) -> Tensor:
        r"""Project x into space of lower triangular matrices."""
        return lower_triangular(x, lower=self.lower)


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

    mask: BoolTensor

    def __init__(self, mask: bool | Tensor = TRUE) -> None:
        super().__init__()
        self.mask = torch.as_tensor(mask, dtype=torch.bool)

    @jit.export
    def forward(self, x: Tensor) -> Tensor:
        r"""Project x into space of masked matrices."""
        return masked(x, self.mask)


# endregion masked projections ---------------------------------------------------------


# region other projections -------------------------------------------------------------
class Contraction(ProjectionBase):
    r"""Return the closest contraction matrix to X.

    .. Signature:: ``(..., m, n) -> (..., m, n)``

    .. math:: \min_Y ∥X-Y∥₂  s.t. ‖Y‖₂ ≤ 1

    One can show analytically that the unique smallest norm minimizer is
    $Y = \min(1, σ⁻¹) X$, where $σ = ‖X‖₂$ is the spectral norm of $X$.

    See Also:
        - `projections.functional.contraction`
    """

    @jit.export
    def forward(self, x: Tensor) -> Tensor:
        r"""Project x into space of contraction matrices."""
        return contraction(x)


class DiagonallyDominant(ProjectionBase):
    r"""Return the closest diagonally dominant matrix to X.

    .. Signature:: ``(..., n, n) -> (..., n, n)``

    .. math:: \min_Y ∥X-Y∥_F  s.t. |Y_{ii}| ≥ ∑_{j≠i} |Y_{ij}| for all i = 1, …, n

    References:
        Computing the nearest diagonally dominant matrix (Mendoza et al. 1998)
    """

    @jit.export
    def forward(self, x: Tensor) -> Tensor:
        r"""Project x into space of diagonally dominant matrices."""
        return diagonally_dominant(x)


# endregion other projections ----------------------------------------------------------
# endregion projections ----------------------------------------------------------------
