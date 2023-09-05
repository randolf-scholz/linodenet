r"""Projections for the Linear ODE Networks.

Notes
-----
Contains projections in modular form.
  - See `~linodenet.projections.functional` for functional implementations.
"""

__all__ = [
    # Classes
    "Banded",
    "Diagonal",
    "Identity",
    "Masked",
    "Normal",
    "Orthogonal",
    "SkewSymmetric",
    "Symmetric",
    "Traceless",
]

from typing import Final, Optional

import torch
from torch import BoolTensor, Tensor, jit, nn

from linodenet.projections.functional import (
    banded,
    diagonal,
    identity,
    masked,
    normal,
    orthogonal,
    skew_symmetric,
    symmetric,
    traceless,
)


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


class Diagonal(nn.Module):
    r"""Return the closest diagonal matrix to X.

    .. Signature:: ``(..., n, n) -> (..., n, n)``

    .. math:: \min_Y ½∥X-Y∥_F^2 s.t. Y = 𝕀⊙Y

    One can show analytically that the unique smallest norm minimizer is $Y = 𝕀⊙X$.
    """

    @jit.export
    def forward(self, x: Tensor) -> Tensor:
        r"""Project x into space of diagonal matrices."""
        return diagonal(x)


class Banded(nn.Module):
    r"""Return the closest banded matrix to X.

    .. Signature:: ``(..., n, n) -> (..., n, n)``

    .. math:: \min_Y ½∥X-Y∥_F^2 s.t. Y = B⊙Y

    One can show analytically that the unique smallest norm minimizer is $Y = B⊙X$.
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


class Masked(nn.Module):
    r"""Return the closest banded matrix to X.

    .. Signature:: ``(..., n, n) -> (..., n, n)``

    .. math:: \min_Y ½∥X-Y∥_F^2 s.t. Y = M⊙Y

    One can show analytically that the unique smallest norm minimizer is $Y = M⊙X$.
    """

    mask: BoolTensor

    def __init__(self, mask: bool | Tensor) -> None:
        super().__init__()
        self.mask = torch.as_tensor(mask, dtype=torch.bool)  # type: ignore[assignment]

    @jit.export
    def forward(self, x: Tensor) -> Tensor:
        r"""Project x into space of masked matrices."""
        return masked(x, self.mask)
