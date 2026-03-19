r"""Bijections."""

__all__ = [
    "MatrixExponential",
    "CayleyMap",
]

from typing import Final

import torch
from torch import Tensor

from linodenet.domains import MatrixDomains
from linodenet_special import matrix_log
from signatures import signature

from .base import BijectionBase


class MatrixExponential(BijectionBase):
    r"""Parametrize a matrix via matrix exponential.

    Note: The following restrictions hold:
        Mₙ(ℝ)  --exp-->  GLₙ(ℝ)
        𝕊ₙ(ℝ)  --exp-->  𝕊ₙ⁺(ℝ)
        𝔸ₙ(ℝ)  --exp-->  Oₙ(ℝ)
    """

    DOMAIN: Final[MatrixDomains] = MatrixDomains.SQUARE
    CODOMAIN: Final[MatrixDomains] = MatrixDomains.INVERTIBLE

    @signature("(..., n, n) -> (..., n, n)")
    def forward(self, x: Tensor) -> Tensor:
        return torch.matrix_exp(x)

    @signature("(..., n, n) -> (..., n, n)")
    def inverse(self, y: Tensor) -> Tensor:
        # FIXME: https://github.com/pytorch/pytorch/issues/9983 (matrix_log)
        return matrix_log(y).real.to(dtype=y.dtype)


class CayleyMap(BijectionBase):
    r"""Parametrize a matrix to be orthogonal via Cayley-Map.

    References:
        - https://pytorch.org/tutorials/intermediate/parametrizations.html
        - https://en.wikipedia.org/wiki/Cayley_transform#Matrix_map
    """

    DOMAIN: Final[MatrixDomains] = MatrixDomains.SKEW_SYMMETRIC
    CODOMAIN: Final[MatrixDomains] = MatrixDomains.CAYLEY_ORTHOGONAL

    @signature("(..., n, n) -> (..., n, n)")
    def forward(self, x: Tensor) -> Tensor:
        I = torch.eye(x.shape[-1], dtype=x.dtype, device=x.device)
        return torch.linalg.lstsq(I + x, I - x).solution

    @signature("(..., n, n) -> (..., n, n)")
    def inverse(self, y: Tensor) -> Tensor:
        I = torch.eye(y.shape[-1], dtype=y.dtype, device=y.device)
        return torch.linalg.lstsq(I + y, I - y).solution
