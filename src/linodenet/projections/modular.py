r"""Projections for the Linear ODE Networks.

Notes
-----
Contains projections in modular form.
  - See `~linodenet.projections.functional` for functional implementations.
"""

__all__ = [
    # Base Classes
    "ProjectionABC",
    # Classes
    "Hamiltonian",
    "Identity",
    "Normal",
    "Orthogonal",
    "SkewSymmetric",
    "Symmetric",
    "Symplectic",
    "Traceless",
    # Masked
    "Banded",
    "Diagonal",
    "LowerTriangular",
    "Masked",
    "UpperTriangular",
]

from abc import abstractmethod
from typing import Final, Optional

import torch
from torch import BoolTensor, Tensor, jit, nn

from linodenet.projections.functional import (
    banded,
    diagonal,
    hamiltonian,
    identity,
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


class ProjectionABC(nn.Module):
    """Abstract Base Class for Projection components."""

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
class Identity(nn.Module):
    r"""Return x as-is.

    .. Signature:: ``(..., n, n) -> (..., n, n)``

    .. math:: \min_Y ½∥X-Y∥_F^2
    """

    @jit.export
    def forward(self, x: Tensor) -> Tensor:
        r"""Project x into space of matrices."""
        return identity(x)


class Symmetric(nn.Module):
    r"""Return the closest symmetric matrix to X.

    .. Signature:: ``(..., n, n) -> (..., n, n)``

    .. math:: \min_Y ½∥X-Y∥_F^2 s.t. Y^⊤ = Y

    One can show analytically that Y = ½(X + X^⊤) is the unique minimizer.
    """

    @jit.export
    def forward(self, x: Tensor) -> Tensor:
        r"""Project x into space of symmetric matrices."""
        return symmetric(x)


class SkewSymmetric(nn.Module):
    r"""Return the closest skew-symmetric matrix to X.

    .. Signature:: ``(..., n, n) -> (..., n, n)``

    .. math:: \min_Y ½∥X-Y∥_F^2 s.t. Y^⊤ = -Y

    One can show analytically that Y = ½(X - X^⊤) is the unique minimizer.
    """

    @jit.export
    def forward(self, x: Tensor) -> Tensor:
        r"""Project x into space of skew-symmetric matrices."""
        return skew_symmetric(x)


class Orthogonal(nn.Module):
    r"""Return the closest orthogonal matrix to X.

    .. Signature:: ``(..., n, n) -> (..., n, n)``

    .. math:: \min_Y ½∥X-Y∥_F^2 s.t. Y^𝖳 Y = 𝕀 = YY^𝖳

    One can show analytically that $Y = UV^𝖳$ is the unique minimizer,
    where $X=UΣV^𝖳$ is the SVD of $X$.

    References
    ----------
    - `<https://math.stackexchange.com/q/2215359>`_
    """

    @jit.export
    def forward(self, x: Tensor) -> Tensor:
        r"""Project x into space of orthogonal matrices."""
        return orthogonal(x)


class Normal(nn.Module):
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


class Traceless(nn.Module):
    r"""Return the closest traceless matrix to X.

    .. Signature:: ``(..., n, n) -> (..., n, n)``

    .. math:: \min_Y ½∥X-Y∥_F^2 s.t. Y^⊤ = -Y

    One can show analytically that Y = ½(X - X^⊤) is the unique minimizer.
    """

    @jit.export
    def forward(self, x: Tensor) -> Tensor:
        r"""Project x into space of traceless matrices."""
        return traceless(x)


class Hamiltonian(nn.Module):
    r"""Return the closest symplectic matrix to X.

    .. Signature:: ``(..., 2n, 2n) -> (..., 2n, 2n)``

    .. math:: \min_Y ½∥X-Y∥_F^2   s.t.   Y^𝖳 J Y = J   where   J=[𝟎, 𝕀; -𝕀, 𝟎]

    Alternatively, the above is equivalent to

    .. math:: \min_Y ½∥X-Y∥_F^2   s.t.   Y^𝖳 J Y = J   where   J= 𝔻₊₁-𝔻₋₁

    where $𝔻ₖ$ is the $2n×2n$ matrix with ones on the k-th diagonal.
    """

    def forward(self, x: Tensor) -> Tensor:
        return hamiltonian(x)


class Symplectic(nn.Module):
    r"""Return the closest symplectic matrix to X.

    .. Signature:: ``(..., 2n, 2n) -> (..., 2n, 2n)``

    .. math:: \min_Y ½∥X-Y∥_F^2   s.t.   Y^𝖳 J Y = J   where   J=[𝟎, 𝕀; -𝕀, 𝟎]

    Alternatively, the above is equivalent to

    .. math:: \min_Y ½∥X-Y∥_F^2   s.t.   Y^𝖳 J Y = J   where   J= 𝔻₊₁-𝔻₋₁

    where $𝔻ₖ$ is the $2n×2n$ matrix with ones on the k-th diagonal.
    """

    def forward(self, x: Tensor) -> Tensor:
        return symplectic(x)


# region masked projections ------------------------------------------------------------
class Diagonal(nn.Module):
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


class Banded(nn.Module):
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

    def __init__(self, upper: int = 0, lower: Optional[int] = None) -> None:
        super().__init__()
        self.upper = upper
        self.lower = upper if lower is None else lower

    @jit.export
    def forward(self, x: Tensor) -> Tensor:
        r"""Project x into space of banded matrices."""
        return banded(x, upper=self.upper, lower=self.lower)


class UpperTriangular(nn.Module):
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


class LowerTriangular(nn.Module):
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


class Masked(nn.Module):
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

    def __init__(self, mask: bool | Tensor) -> None:
        super().__init__()
        self.mask = torch.as_tensor(mask, dtype=torch.bool)  # type: ignore[assignment]

    @jit.export
    def forward(self, x: Tensor) -> Tensor:
        r"""Project x into space of masked matrices."""
        return masked(x, self.mask)


# endregion masked projections ---------------------------------------------------------
# endregion projections ----------------------------------------------------------------
