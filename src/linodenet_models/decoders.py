__all__ = ["LowRankTransform"]

from math import sqrt
from typing import Final

import torch
from torch import Tensor, nn


class LowRankTransform(nn.Module):
    r"""An invertible, efficient diagonal-scaled low-rank perturbation layer.

    .. math:: y = (𝕀ₙ + USVᵀ)x

    where $U, V ∈ ℝⁿˣᵏ$ and $S = \operatorname{diag}(s)$ with

    .. math:: sᵢ = \frac{ρ \tanh(θᵢ)}{‖Uᵀvᵢ‖₁ + δ}

    for unconstrained parameters $θ ∈ ℝᵏ$, contraction factor $0 < ρ < 1$, and
    stability constant $δ > 0$. Since the $i$-th row of $SVᵀU$ is
    $sᵢ (Uᵀvᵢ)ᵀ$, its row sum satisfies

    .. math:: ‖(SVᵀU)ᵢ•‖₁ = |sᵢ| ‖Uᵀvᵢ‖₁ < ρ

    and therefore $‖SVᵀU‖∞ < ρ < 1$. Hence $𝕀ₖ + SVᵀU$ is always invertible by
    the Neumann series, which in turn guarantees that $𝕀ₙ + USVᵀ$ is invertible.
    With the help of the Woodbury matrix identity [1]_, we have the general formula

    .. math:: (A + UCVᵀ)⁻¹ = A⁻¹ - A⁻¹U(C⁻¹ + VᵀA⁻¹U)⁻¹VᵀA⁻¹

    whenever the displayed inverses exist. Setting $A = 𝕀ₙ$ and $C = S$ yields

    .. math:: (𝕀ₙ + USVᵀ)⁻¹ = 𝕀ₙ - U(𝕀ₖ + SVᵀU)⁻¹SVᵀ

    Thus, to compute the inverse pass, we do not need to solve an $n×n$ system,
    but only a $k×k$ one. In particular, when $k = 1$, the formula reduces to

    .. math:: (𝕀ₙ + suvᵀ)⁻¹ = 𝕀ₙ - \frac{s}{1 + s vᵀu} uvᵀ

    To calculate the log determinant of the Jacobian, we use Sylvester's
    determinant identity [2]_:

    .. math:: \log|\det(𝕀ₙ + USVᵀ)| = \log|\det(𝕀ₖ + SVᵀU)|

    References:
        .. [1] https://en.wikipedia.org/wiki/Woodbury_matrix_identity
        .. [2] https://en.wikipedia.org/wiki/Sylvester%27s_determinant_theorem
    """

    # CONSTANTS
    rank: Final[int]
    r"""CONST: The rank of the low rank matrix."""
    rho: Final[float]
    r"""CONST: Strict upper bound for $‖SVᵀU‖∞$."""
    delta: Final[float]
    r"""CONST: Stability offset used in the diagonal scale parameterization."""

    # PARAMETERS
    U: Tensor
    r"""PARAM: $n×k$ tensor"""
    V: Tensor
    r"""PARAM: $n×k$ tensor"""
    theta: Tensor
    r"""PARAM: Unconstrained length-$k$ tensor parameterizing the diagonal of $S$."""
    eye: Tensor
    r"""BUFFER: Identity matrix in latent rank space."""

    def __init__(
        self,
        input_size: int,
        *,
        rank: int,
        rho: float = 0.9,
        delta: float = 1e-6,
    ) -> None:
        super().__init__()
        if not 0 < rho < 1:
            raise ValueError(f"Expected 0 < rho < 1, got {rho=}.")
        if delta <= 0:
            raise ValueError(f"Expected delta > 0, got {delta=}.")
        self.input_size = input_size
        self.rank = rank
        self.rho = rho
        self.delta = delta
        self.U = nn.Parameter(torch.empty(input_size, rank))
        self.V = nn.Parameter(torch.empty(input_size, rank))
        self.theta = nn.Parameter(torch.empty(rank))
        nn.init.normal_(self.U, std=1 / sqrt(rank))
        nn.init.normal_(self.V, std=1 / sqrt(input_size))
        nn.init.zeros_(self.theta)
        self.register_buffer("eye", torch.eye(rank), persistent=True)

    def diag_values(self, VtU: Tensor, /) -> Tensor:
        r"""Compute the diagonals of $S$ from the unconstrained parameters."""
        row_norms = torch.linalg.vector_norm(VtU, ord=1, dim=-1)
        # Let R = SVᵀU. Then the i-th row of R is sᵢ(vᵢᵀU), so
        # ‖Rᵢ•‖₁ = |sᵢ| ‖vᵢᵀU‖₁ < ρ. Therefore ‖R‖∞ = maxᵢ ‖Rᵢ•‖₁ < ρ < 1,
        # which guarantees that I + R is invertible by the Neumann series.
        return self.rho * self.theta.tanh() / (row_norms + self.delta)

    # @signature("(..., n) -> (..., n)")
    def encode(self, x: Tensor, /) -> Tensor:
        r"""Computes $y = (𝕀ₙ + USVᵀ)x$."""
        VtU = torch.einsum("ni, nj -> ij", self.V, self.U)
        s = self.diag_values(VtU)
        v = torch.einsum("nk, ...n -> ...k", self.V, x)  # v = Vᵀx
        sv = s * v  # sv = SVᵀx
        u = torch.einsum("nk, ...k -> ...n", self.U, sv)  # u = USVᵀx
        return x + u

    # @signature("(..., n) -> (..., n)")
    def decode(self, y: Tensor, /) -> Tensor:
        r"""Computes $x = (𝕀 + USVᵀ)⁻¹y = y - U(𝕀ₖ + SVᵀU)⁻¹SVᵀy$."""
        VtU = torch.einsum("ni, nj -> ij", self.V, self.U)
        s = self.diag_values(VtU)
        A = self.eye + s[:, None] * VtU  # A = 𝕀ₖ + SVᵀU
        v = torch.einsum("nk, ...n -> ...k", self.V, y)  # v = Vᵀy
        sv = s * v  # sv = SVᵀy
        z = torch.linalg.solve(A, sv[..., None]).squeeze(-1)  # z = (𝕀ₖ + SVᵀU)⁻¹SVᵀy
        u = torch.einsum("nk, ...k -> ...n", self.U, z)  # u = U(𝕀ₖ + SVᵀU)⁻¹SVᵀy
        return y - u

    # @signature("(..., n) -> [(..., n), (...)]")
    def encode_and_logabsdet(self, x: Tensor, /) -> tuple[Tensor, Tensor]:
        VtU = torch.einsum("ni, nj -> ij", self.V, self.U)
        s = self.diag_values(VtU)
        A = self.eye + s[:, None] * VtU  # A = 𝕀ₖ + SVᵀU
        _, logabsdet = torch.linalg.slogdet(A)
        v = torch.einsum("nk, ...n -> ...k", self.V, x)  # v = Vᵀx
        sv = s * v  # sv = SVᵀx
        u = torch.einsum("nk, ...k -> ...n", self.U, sv)  # u = USVᵀx
        return x + u, logabsdet.expand(x.shape[:-1])

    # @signature("(..., n) -> [(..., n), (...)]")
    def decode_and_logabsdet(self, y: Tensor, /) -> tuple[Tensor, Tensor]:
        VtU = torch.einsum("ni, nj -> ij", self.V, self.U)
        s = self.diag_values(VtU)
        A = self.eye + s[:, None] * VtU  # A = 𝕀ₖ + SVᵀU
        _, logabsdet = torch.linalg.slogdet(A)
        v = torch.einsum("nk, ...n -> ...k", self.V, y)  # v = Vᵀy
        sv = s * v  # sv = SVᵀy
        z = torch.linalg.solve(A, sv[..., None]).squeeze(-1)  # z = (𝕀ₖ + SVᵀU)⁻¹SVᵀy
        u = torch.einsum("nk, ...k -> ...n", self.U, z)  # u = U(𝕀ₖ + SVᵀU)⁻¹SVᵀy
        return y - u, -logabsdet.expand(y.shape[:-1])
