r"""Bijections."""

__all__ = [
    "MatrixExponential",
    "PositiveScalarMatrix",
    "PositiveDiagonal",
    "CayleyMap",
]

from typing import Final

import torch
from torch import Tensor

from linodenet.domains import MatrixDomains, ScalarDomains, VectorDomains
from linodenet_special import matrix_log
from signatures import signature

from .base import BijectionBase


class MatrixExponential(BijectionBase):
    r"""Parametrize a matrix via matrix exponential.

    Note: The following restrictions hold:
        Mₙ(ℝ)  --exp-->  GLₙ(ℝ)∩{A ∣ A=B² for some B∈Mₙ(ℝ)}
        Mₙ(ℂ)  --exp-->  GLₙ(ℂ)
        𝕊ₙ(ℝ)  --exp-->  𝕊ₙ⁺(ℝ)
        𝔸ₙ(ℝ)  --exp-->  Oₙ(ℝ)
    """

    DOMAIN: Final[MatrixDomains] = MatrixDomains.SQUARE
    CODOMAIN: Final[MatrixDomains] = MatrixDomains.INVERTIBLE

    @signature("(..., n, n) -> (..., n, n)")
    def forward(self, x: Tensor, /) -> Tensor:
        return torch.matrix_exp(x)

    @signature("(..., n, n) -> (..., n, n)")
    def inverse(self, y: Tensor, /) -> Tensor:
        # FIXME: https://github.com/pytorch/pytorch/issues/9983 (matrix_log)
        return matrix_log(y).real.to(dtype=y.dtype)


class PositiveScalarMatrix(BijectionBase):
    r"""Map scalars to positive scalar matrices via $x ↦ \exp(x) I$."""

    DOMAIN: Final[ScalarDomains] = ScalarDomains.REAL_LINE
    CODOMAIN: Final[MatrixDomains] = MatrixDomains.POSITIVE_SCALAR_MATRIX

    size: int

    def __init__(self, size: int) -> None:
        super().__init__()
        if size <= 0:
            raise ValueError("size must be a positive integer.")
        self.size = size

    @signature("(...) -> (..., n, n)")
    def forward(self, x: Tensor, /) -> Tensor:
        eye = torch.eye(self.size, dtype=x.dtype, device=x.device)
        return torch.exp(x).unsqueeze(-1).unsqueeze(-1) * eye

    @signature("(..., n, n) -> (...)")
    def inverse(self, y: Tensor, /) -> Tensor:
        return torch.log(y.diagonal(dim1=-2, dim2=-1)[..., 0])


class PositiveDiagonal(BijectionBase):
    r"""Map vectors to positive diagonal matrices via $v ↦ \operatorname{diag}(\exp(v))$."""

    DOMAIN: Final[VectorDomains] = VectorDomains.REAL
    CODOMAIN: Final[MatrixDomains] = MatrixDomains.POSITIVE_DIAGONAL

    @signature("(..., n) -> (..., n, n)")
    def forward(self, x: Tensor, /) -> Tensor:
        return torch.diag_embed(torch.exp(x))

    @signature("(..., n, n) -> (..., n)")
    def inverse(self, y: Tensor, /) -> Tensor:
        return torch.log(y.diagonal(dim1=-2, dim2=-1))


class CayleyMap(BijectionBase):
    r"""Parametrize a matrix to be orthogonal via Cayley-Map.

    References:
        - https://pytorch.org/tutorials/intermediate/parametrizations.html
        - https://en.wikipedia.org/wiki/Cayley_transform#Matrix_map
    """

    DOMAIN: Final[MatrixDomains] = MatrixDomains.SKEW_SYMMETRIC
    CODOMAIN: Final[MatrixDomains] = MatrixDomains.CAYLEY_ORTHOGONAL

    @signature("(..., n, n) -> (..., n, n)")
    def forward(self, x: Tensor, /) -> Tensor:
        I = torch.eye(x.shape[-1], dtype=x.dtype, device=x.device)
        return torch.linalg.lstsq(I + x, I - x).solution

    @signature("(..., n, n) -> (..., n, n)")
    def inverse(self, y: Tensor, /) -> Tensor:
        I = torch.eye(y.shape[-1], dtype=y.dtype, device=y.device)
        return torch.linalg.lstsq(I + y, I - y).solution
