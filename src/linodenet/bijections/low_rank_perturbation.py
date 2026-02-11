r"""Low-rank perturbation layer."""

__all__ = ["iLowRankLayer"]

from typing import Final

import torch
from torch import Tensor, nn

from linodenet.initializations import low_rank
from linodenet.signatures import signature


class iLowRankLayer(nn.Module):
    r"""An invertible, efficient low rank perturbation layer.

    With the help of the Matrix Inversion Lemma [1]_ (also known as Woodbury matrix identity),
    we have

    .. math:: (𝕀ₙ + UVᵀ)⁻¹ = 𝕀ₙ - U(𝕀ₖ + VᵀU)⁻¹Vᵀ

    I.e. to compute the inverse of the perturbed matrix, it is sufficient to compute the
    inverse of the lower dimensional low rank matrix $𝕀ₖ + VᵀU$.
    In particular, when $k=1$ the formula reduces to

    .. math:: (𝕀ₙ + uvᵀ)⁻¹ = 𝕀ₙ - \frac{1}{1+uᵀv} uvᵀ

    To calculate the log determinant of the Jacobian, we use the Matrix Determinant Lemma [2]_:

    .. math:: \log|\det(𝕀ₙ + UVᵀ)| = \log|\det(𝕀ₖ + VᵀU)| + \log|\det(𝕀ₙ + VᵀU)|

    References:
        .. [1] https://en.wikipedia.org/wiki/Woodbury_matrix_identity
        .. [2] https://en.wikipedia.org/wiki/Matrix_determinant_lemma
    """

    # CONSTANTS
    rank: Final[int]
    r"""CONST: The rank of the low rank matrix."""

    # PARAMETERS
    U: Tensor
    r"""PARAM: $n×k$ tensor"""
    V: Tensor
    r"""PARAM: $n×k$ tensor"""

    @property
    def config(self) -> dict:
        return {
            "input_size": self.input_size,
            "rank": self.rank,
        }

    def __init__(self, input_size: int, *, rank: int) -> None:
        super().__init__()
        self.input_size = input_size
        self.rank = rank
        self.U = low_rank(input_size)
        self.V = low_rank(input_size)

    @signature("(..., n) -> (..., n)")
    def forward(self, x: Tensor) -> Tensor:
        z = torch.einsum("...n, nk -> ...k", self.V, x)
        y = torch.einsum("...k, nk -> ...n", self.U, z)
        return x + y

    @signature("(..., n) -> (..., n)")
    def inverse(self, x: Tensor) -> Tensor:
        z = torch.einsum("...n, nk -> ...k", self.V, x)
        A = torch.eye(self.rank) + torch.einsum("nk, nk -> kk", self.U, self.V)
        y = torch.linalg.solve(A, z)
        return x - torch.einsum("...k, nk -> ...n", self.U, y)
