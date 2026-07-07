r"""Low-rank perturbation layers."""

__all__ = ["LowRankTransform", "SymmetricLowRankTransform"]

from math import sqrt
from typing import Final

import torch
from torch import Tensor, nn

from linodenet.mappings.abstract import Transform
from signatures import signature


class LowRankTransform(nn.Module, Transform):
    r"""An invertible, efficient diagonal-scaled low-rank perturbation layer.

    .. math:: y = (𝕀ₙ + USVᵀ)x

    where $U, V ∈ ℝⁿˣᵏ$ and $S = \operatorname{diag}(s)$ with $s ∈ ℝᵏ$ stored as a
    length-$k$ parameter vector. With the help of the Woodbury matrix identity [1]_,
    we have the general formula

    .. math:: (A + UCVᵀ)⁻¹ = A⁻¹ - A⁻¹U(C⁻¹ + VᵀA⁻¹U)⁻¹VᵀA⁻¹

    whenever the displayed inverses exist. Setting $A = 𝕀ₙ$ and $C = S$ yields

    .. math:: (𝕀ₙ + USVᵀ)⁻¹ = 𝕀ₙ - US(𝕀ₖ + VᵀUS)⁻¹Vᵀ

    Thus, to compute the inverse pass, we do not need to solve an $n×n$ system,
    but only a $k×k$ one. In particular, when $k = 1$, the formula reduces to

    .. math:: (𝕀ₙ + suvᵀ)⁻¹ = 𝕀ₙ - \frac{s}{1 + s vᵀu} uvᵀ

    To calculate the log determinant of the Jacobian, we use Sylvester's determinant
    identity [2]_:

    .. math:: \log|\det(𝕀ₙ + USVᵀ)| = \log|\det(𝕀ₖ + VᵀUS)|

    References:
        .. [1] https://en.wikipedia.org/wiki/Woodbury_matrix_identity
        .. [2] https://en.wikipedia.org/wiki/Matrix_determinant_lemma
        .. [3] https://en.wikipedia.org/wiki/Sylvester%27s_determinant_theorem
    """

    # CONSTANTS
    rank: Final[int]
    r"""CONST: The rank of the low rank matrix."""

    # PARAMETERS
    U: Tensor
    r"""PARAM: $n×k$ tensor"""
    V: Tensor
    r"""PARAM: $n×k$ tensor"""
    S: Tensor
    r"""PARAM: Length-$k$ tensor storing the diagonal of $S$."""
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
        self.S = nn.Parameter(torch.ones(rank))
        nn.init.normal_(self.U, std=1 / sqrt(rank))
        nn.init.normal_(self.V, std=1 / sqrt(input_size))
        self.register_buffer("eye", torch.eye(rank), persistent=True)

    @signature("(..., n) -> (..., n)")
    def encode(self, x: Tensor, /) -> Tensor:
        r"""Computes $y = (𝕀ₙ + USVᵀ)x$."""
        v = torch.einsum("nk, ...n -> ...k", self.V, x)  # v = Vᵀx
        sv = self.S * v  # sv = SVᵀx
        u = torch.einsum("nk, ...k -> ...n", self.U, sv)  # u = USVᵀx
        return x + u

    @signature("(..., n) -> (..., n)")
    def decode(self, y: Tensor, /) -> Tensor:
        r"""Computes $x = (𝕀 + USVᵀ)⁻¹y = y - US(𝕀ₖ + VᵀUS)⁻¹Vᵀy$."""
        A = self.eye + torch.einsum(  # A = 𝕀ₖ + VᵀUS
            "ni, nj, j -> ij", self.V, self.U, self.S
        )
        v = torch.einsum("nk, ...n -> ...k", self.V, y)  # v = Vᵀy
        z = torch.linalg.solve(A, v[..., None]).squeeze(-1)  # z = (𝕀ₖ + VᵀUS)⁻¹Vᵀy
        sz = self.S * z  # sz = S(𝕀ₖ + VᵀUS)⁻¹Vᵀy
        u = torch.einsum("nk, ...k -> ...n", self.U, sz)  # u = US(𝕀ₖ + VᵀUS)⁻¹Vᵀy
        return y - u

    @signature("(..., n) -> [(..., n), (...)]")
    def encode_and_logabsdet(self, x: Tensor, /) -> tuple[Tensor, Tensor]:
        A = self.eye + torch.einsum(  # A = 𝕀ₖ + VᵀUS
            "ni, nj, j -> ij", self.V, self.U, self.S
        )
        _, logabsdet = torch.linalg.slogdet(A)
        v = torch.einsum("nk, ...n -> ...k", self.V, x)  # v = Vᵀx
        sv = self.S * v  # sv = SVᵀx
        u = torch.einsum("nk, ...k -> ...n", self.U, sv)  # u = USVᵀx
        return x + u, logabsdet.expand(x.shape[:-1])

    @signature("(..., n) -> [(..., n), (...)]")
    def decode_and_logabsdet(self, y: Tensor, /) -> tuple[Tensor, Tensor]:
        A = self.eye + torch.einsum(  # A = 𝕀ₖ + VᵀUS
            "ni, nj, j -> ij", self.V, self.U, self.S
        )
        _, logabsdet = torch.linalg.slogdet(A)
        v = torch.einsum("nk, ...n -> ...k", self.V, y)  # v = Vᵀy
        z = torch.linalg.solve(A, v[..., None]).squeeze(-1)  # z = (𝕀ₖ + VᵀUS)⁻¹Vᵀy
        sz = self.S * z  # sz = S(𝕀ₖ + VᵀUS)⁻¹Vᵀy
        u = torch.einsum("nk, ...k -> ...n", self.U, sz)  # u = US(𝕀ₖ + VᵀUS)⁻¹Vᵀy
        return y - u, -logabsdet.expand(y.shape[:-1])


class SymmetricLowRankTransform(nn.Module, Transform):
    r"""An invertible, efficient symmetric low-rank perturbation layer.

    .. math:: y = (𝕀ₙ + UUᵀ)x

    where $U ∈ ℝⁿˣᵏ$. This is the symmetric specialization of the Woodbury identity [1]_

    .. math:: (A + UCVᵀ)⁻¹ = A⁻¹ - A⁻¹U(C⁻¹ + VᵀA⁻¹U)⁻¹VᵀA⁻¹

    obtained by setting $A = 𝕀ₙ$, $C = 𝕀ₖ$, and $V = U$, yielding

    .. math:: (𝕀ₙ + UUᵀ)⁻¹ = 𝕀ₙ - U(𝕀ₖ + UᵀU)⁻¹Uᵀ

    Thus, to compute the inverse pass, we only need to solve a $k×k$ system. To
    calculate the log determinant of the Jacobian, we use Sylvester's determinant
    identity [2]_:

    .. math:: \log|\det(𝕀ₙ + UUᵀ)| = \log|\det(𝕀ₖ + UᵀU)|

    References:
        .. [1] https://en.wikipedia.org/wiki/Woodbury_matrix_identity
        .. [2] https://en.wikipedia.org/wiki/Sylvester%27s_determinant_theorem
    """

    # CONSTANTS
    rank: Final[int]
    r"""CONST: The rank of the low rank matrix."""

    # PARAMETERS
    U: Tensor
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
        nn.init.normal_(self.U, std=1 / sqrt(rank))
        self.register_buffer("eye", torch.eye(rank), persistent=True)

    @signature("(..., n) -> (..., n)")
    def encode(self, x: Tensor, /) -> Tensor:
        r"""Computes $y = (𝕀ₙ + UUᵀ)x$."""
        u = torch.einsum("nk, ...n -> ...k", self.U, x)  # u = Uᵀx
        v = torch.einsum("nk, ...k -> ...n", self.U, u)  # v = UUᵀx
        return x + v

    @signature("(..., n) -> (..., n)")
    def decode(self, y: Tensor, /) -> Tensor:
        r"""Computes $x = (𝕀 + UUᵀ)⁻¹y = y - U(𝕀ₖ + UᵀU)⁻¹Uᵀy$."""
        A = self.eye + torch.einsum("ni, nj -> ij", self.U, self.U)  # A = 𝕀ₖ + UᵀU
        u = torch.einsum("nk, ...n -> ...k", self.U, y)  # u = Uᵀy
        z = torch.linalg.solve(A, u[..., None]).squeeze(-1)  # z = (𝕀ₖ + UᵀU)⁻¹Uᵀy
        v = torch.einsum("nk, ...k -> ...n", self.U, z)  # v = U(𝕀ₖ + UᵀU)⁻¹Uᵀy
        return y - v

    @signature("(..., n) -> [(..., n), (...)]")
    def encode_and_logabsdet(self, x: Tensor, /) -> tuple[Tensor, Tensor]:
        A = self.eye + torch.einsum("ni, nj -> ij", self.U, self.U)  # A = 𝕀ₖ + UᵀU
        _, logabsdet = torch.linalg.slogdet(A)
        u = torch.einsum("nk, ...n -> ...k", self.U, x)  # u = Uᵀx
        v = torch.einsum("nk, ...k -> ...n", self.U, u)  # v = UUᵀx
        return x + v, logabsdet.expand(x.shape[:-1])

    @signature("(..., n) -> [(..., n), (...)]")
    def decode_and_logabsdet(self, y: Tensor, /) -> tuple[Tensor, Tensor]:
        A = self.eye + torch.einsum("ni, nj -> ij", self.U, self.U)  # A = 𝕀ₖ + UᵀU
        _, logabsdet = torch.linalg.slogdet(A)
        u = torch.einsum("nk, ...n -> ...k", self.U, y)  # u = Uᵀy
        z = torch.linalg.solve(A, u[..., None]).squeeze(-1)  # z = (𝕀ₖ + UᵀU)⁻¹Uᵀy
        v = torch.einsum("nk, ...k -> ...n", self.U, z)  # v = U(𝕀ₖ + UᵀU)⁻¹Uᵀy
        return y - v, -logabsdet.expand(y.shape[:-1])
