r"""Low-rank perturbation layer."""

__all__ = ["LowRankFlow"]

from math import sqrt
from typing import Final

import torch
from torch import Tensor, nn

from signatures import signature


class LowRankFlow(nn.Module):
    r"""An invertible, efficient low rank perturbation layer.

    .. math:: y = (𝕀ₙ + UVᵀ)x

    where U and V are both is n×k. With the help of the Matrix Inversion Lemma [1]_,
    also known as Woodbury matrix identity, holds:

    .. math:: (𝕀ₙ + UVᵀ)⁻¹ = 𝕀ₙ - U(𝕀ₖ + VᵀU)⁻¹Vᵀ

    Thus, to compute the inverse pass, we do not need to solve an n×n system,
    but only a k×k one. In particular, when k=1, the formula reduces to

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
    eye: Tensor
    r"""BUFFER: Identity matrix in latent rank space."""

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
        self.U = nn.Parameter(torch.empty(input_size, rank))
        self.V = nn.Parameter(torch.empty(input_size, rank))
        nn.init.normal_(self.U, std=1 / sqrt(rank))
        nn.init.normal_(self.V, std=1 / sqrt(input_size))
        self.register_buffer("eye", torch.eye(rank), persistent=True)

    @signature("(..., n) -> (..., n)")
    def encode(self, x: Tensor) -> Tensor:
        r"""Computes $y = (𝕀ₙ + UVᵀ)x$."""
        v = torch.einsum("nk, ...n -> ...k", self.V, x)  # v = Vᵀx
        u = torch.einsum("nk, ...k -> ...n", self.U, v)  # u = Uv
        return x + u

    @signature("(..., n) -> (..., n)")
    def decode(self, y: Tensor) -> Tensor:
        r"""Computes $x = (𝕀+UVᵀ)⁻¹y = y - U(𝕀ₖ + VᵀU)⁻¹Vᵀy$."""
        A = self.eye + torch.einsum("nk, nk -> kk", self.U, self.V)
        v = torch.einsum("nk, ...n -> ...k", self.V, y)  # v = Vᵀy
        z = torch.linalg.solve(A, v)  # z = (𝕀ₖ + VᵀU)⁻¹v
        u = torch.einsum("nk, ...k -> ...n", self.U, z)  # u = Uz
        return y - u

    @signature("(..., n) -> [(..., n), (...)]")
    def encode_and_logabsdet(self, x: Tensor, /) -> tuple[Tensor, Tensor]:
        A = self.eye + torch.einsum("nk, nk -> kk", self.U, self.V)
        _, logabsdet = torch.linalg.slogdet(A)
        v = torch.einsum("nk, ...n -> ...k", self.V, x)  # v = Vᵀx
        u = torch.einsum("nk, ...k -> ...n", self.U, v)  # u = Uv
        y = x + u
        logabsdet = logabsdet.expand(x.shape[:-1])
        return y, logabsdet

    @signature("(..., n) -> [(..., n), (...)]")
    def decode_and_logabsdet(self, y: Tensor, /) -> tuple[Tensor, Tensor]:
        A = self.eye + torch.einsum("nk, nk -> kk", self.U, self.V)
        v = torch.einsum("nk, ...n -> ...k", self.V, y)  # v = Vᵀy
        z = torch.linalg.solve(A, v)  # z = (𝕀ₖ + VᵀU)⁻¹v
        u = torch.einsum("nk, ...k -> ...n", self.U, z)  # u = Uz
        x = y - u
        _, logabsdet = torch.linalg.slogdet(A)
        logabsdet = (-logabsdet).expand(y.shape[:-1])
        return x, logabsdet
