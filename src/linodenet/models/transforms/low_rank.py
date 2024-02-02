"""Low-rank perturbation layer."""

__all__ = ["iLowRankLayer"]

from typing import Any, Final

import torch
from torch import Tensor, nn

from linodenet.initializations import low_rank
from linodenet.utils import deep_dict_update


class iLowRankLayer(nn.Module):
    r"""An invertible, efficient low rank perturbation layer.

    With the help of the Matrix Inversion Lemma [1] (also known as Woodbury matrix identity),
    we have

    .. math:: (𝕀_n + UV^⊤)^{-1} = 𝕀_n - U(𝕀_k + V^⊤U)^{-1}V^⊤

    I.e. to compute the inverse of the perturbed matrix, it is sufficient to compute the
    inverse of the lower dimensional low rank matrix `𝕀_k + V^⊤U`.
    In particular, when `k=1` the formula reduces to

    .. math:: (𝕀_n + uv^⊤)^{-1} = 𝕀_n - \frac{1}{1+u^⊤v} uv^⊤

    To calculate the log determinant of the Jacobian, we use the the Matrix Determinant Lemma [2]:

    References:
        .. [1] https://en.wikipedia.org/wiki/Woodbury_matrix_identity
        .. [2] https://en.wikipedia.org/wiki/Matrix_determinant_lemma
    """

    HP = {
        "__name__": __qualname__,
        "__module__": __module__,
    }
    r"""The hyperparameter dictionary"""

    # CONSTANTS
    rank: Final[int]
    r"""CONST: The rank of the low rank matrix."""

    # PARAMETERS
    U: Tensor
    r"""PARAM: $n×k$ tensor"""
    V: Tensor
    r"""PARAM: $n×k$ tensor"""

    def __init__(self, input_size: int, rank: int, **HP: Any):
        super().__init__()
        self.HP = deep_dict_update(self.HP, HP)
        self.U = low_rank(input_size)
        self.V = low_rank(input_size)
        self.rank = rank

    def forward(self, x: Tensor) -> Tensor:
        r""".. Signature:: ``(..., n) -> (..., n)``."""
        z = torch.einsum("...n, nk -> ...k", self.V, x)
        y = torch.einsum("...k, nk -> ...n", self.U, z)
        return x + y

    def inverse(self, x: Tensor) -> Tensor:
        r""".. Signature:: ``(..., n) -> (..., n)``."""
        z = torch.einsum("...n, nk -> ...k", self.V, x)
        A = torch.eye(self.rank) + torch.einsum("nk, nk -> kk", self.U, self.V)
        y = torch.linalg.solve(A, z)
        return x - torch.einsum("...k, nk -> ...n", self.U, y)

    # def __invert__(self):
    #     r"""Compute the inverse of the low rank layer.
    #     Returns
    #     -------
    #     iLowRankLayer
    #     """
    #     return iLowRankLayer(self.V, self.U)
