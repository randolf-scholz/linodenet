__all__ = ["MatrixExponential"]


from typing import Final

import torch
from torch import Tensor, jit

from linodenet.domains import MatrixDomains
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

    @jit.export
    @signature("(..., n, n) -> (..., n, n)")
    def forward(self, x: Tensor) -> Tensor:
        return torch.matrix_exp(x)

    @jit.export
    @signature("(..., n, n) -> (..., n, n)")
    def inverse(self, y: Tensor) -> Tensor:
        r"""This requires the matrix logarithm, which is not implemented in PyTorch.

        See: https://github.com/pytorch/pytorch/issues/9983
        """
        raise NotImplementedError
