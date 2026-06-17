r"""ProFITi-style forecasting components."""

__all__ = [
    "ProFITi",
    "ProFITiConfig",
    "ProFITiConditioning",
    "ProFITiFlow",
    "ProFITiFlowLayer",
    "Shiesh",
    "TriangularAttention",
    "ProFITiBlock",
]

import math
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import chain
from typing import Any, Final

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .grafiti import Grafiti


# implements Transform protocol
class Shiesh(nn.Module):
    r"""Elementwise Shiesh transform used by ProFITi flows.

    .. math:: fₜ(x) = (1/a)\sinh⁻¹(ℯᵃᵗ\sinh(a⋅x))

    Its inverse is $f₋ₜ$, and its derivative is

    .. math:: fₜ'(x) = \frac{ℯᵃᵗ\cosh(a⋅x)}{\sqrt{1 + ℯ²ᵃᵗ\sinh²(a⋅x)}}

    Shiesh is the solution of the following IVP at time t:

    .. math:: du/dt =\tanh(a⋅u(t)), u(0) = x
    """

    LOG2: Final[float] = math.log(2.0)

    def __init__(self, t: float = 1.0, a: float = 1.0) -> None:
        super().__init__()
        if a <= 0:
            raise ValueError("a must be positive.")

        self.t = nn.Parameter(torch.tensor(t), requires_grad=False)
        self.a = nn.Parameter(torch.tensor(a), requires_grad=False)

    @staticmethod
    def _log_cosh(x: Tensor, /) -> Tensor:
        abs_x = x.abs()
        safe_x = torch.where(abs_x < 20.0, x, 0.0)
        direct = torch.log(torch.cosh(safe_x))
        asymptotic = abs_x - Shiesh.LOG2 + torch.log1p(torch.exp(-2.0 * abs_x))
        return torch.where(abs_x < 20.0, direct, asymptotic)

    @staticmethod
    def _asinh_exp(x: Tensor, /) -> Tensor:
        m = x < 20.0
        # torch.where evaluates both branches eagerly; mask branch inputs before
        # exp to avoid overflow in the inactive branch.
        x0 = torch.where(m, x, 0.0)
        x1 = torch.where(m, 0.0, x)
        y0 = torch.asinh(torch.exp(x0))
        y1 = x1 + torch.log1p(torch.sqrt(1.0 + torch.exp(-2.0 * x1)))
        return torch.where(m, y0, y1)

    def _transform_and_logabsdet(self, x: Tensor, t: Tensor) -> tuple[Tensor, Tensor]:
        a = self.a.to(device=x.device, dtype=x.dtype)
        t = t.to(device=x.device, dtype=x.dtype)
        # u=a⋅x, s=a⋅t, v=log|ℯˢsinh(u)|;
        u = a * x
        s = a * t
        m = (u.abs() < 20.0) & (s.abs() < 80.0)

        # Direct formula for moderate values:
        # z=ℯˢsinh(u), y₀=asinh(z)/a,  j₀=log(ℯˢcosh(u)/√(1+z²)).
        u0 = torch.where(m, u, 0.0)
        z = torch.exp(torch.where(s.abs() < 80.0, s, 0.0)) * torch.sinh(u0)
        y0 = torch.asinh(z) / a
        j0 = s + torch.log(torch.cosh(u0)) - 0.5 * torch.log1p(z.square())

        # Stable formula for large values:
        # v=log|ℯˢsinh(u)|, y₁=sign(u)⋅asinh(ℯᵛ)/a, j₁=s+log(cosh(u))-½log(1+ℯ²ᵛ).
        r = torch.where(m, 20.0, u.abs())
        v = s + r - self.LOG2 + torch.log1p(-torch.exp(-2.0 * r))
        y1 = torch.sign(u) * self._asinh_exp(v) / a
        j1 = s + self._log_cosh(u) - 0.5 * F.softplus(2.0 * v)

        return torch.where(m, y0, y1), torch.where(m, j0, j1)

    def encode_and_logabsdet(self, x: Tensor, /) -> tuple[Tensor, Tensor]:
        return self._transform_and_logabsdet(x, self.t)

    def decode_and_logabsdet(self, y: Tensor, /) -> tuple[Tensor, Tensor]:
        return self._transform_and_logabsdet(y, -self.t)


@dataclass(frozen=True)
class ProFITiConfig:
    r"""Configuration for constructing a ProFITi model."""

    input_dim: int = 41
    num_heads: int = 4
    latent_dim: int = 128
    num_layers: int = 2


class ProFITi(nn.Module):
    r"""Stub for a streamlined ProFITi re-implementation.

    Args:
        conditioning_module: Module that maps context and query data to flow
            conditioning states.
        flow: Conditional normalizing flow over target values.
    """

    conditioning_module: nn.Module
    flow: nn.Module

    @classmethod
    def from_config(
        cls,
        config: ProFITiConfig | Mapping[str, Any],
        /,
    ) -> ProFITi:
        r"""Construct a ProFITi model from a configuration object."""
        if isinstance(config, Mapping):
            config = ProFITiConfig(**config)

        conditioning_module = Grafiti(
            input_dim=config.input_dim,
            hidden_dim=config.latent_dim,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
        )
        return cls(conditioning_module=conditioning_module, flow=nn.Identity())

    def __init__(
        self,
        *,
        conditioning_module: nn.Module,
        flow: nn.Module,
    ) -> None:
        super().__init__()
        self.conditioning_module = conditioning_module
        self.flow = flow

    def sample(
        self,
        context_times: Tensor,
        context_values: Tensor,
        query_times: Tensor,
        *,
        context_mask: Tensor | None = None,
        query_mask: Tensor | None = None,
        num_samples: int = 1,
    ) -> Tensor:
        raise NotImplementedError

    def log_prob(
        self,
        value: Tensor,
        context_times: Tensor,
        context_values: Tensor,
        query_times: Tensor,
        *,
        context_mask: Tensor | None = None,
        query_mask: Tensor | None = None,
    ) -> Tensor:
        raise NotImplementedError


class TriangularAttention(nn.Module):
    r"""Implements (sorted) invertible triangular attention (equation 10).

    .. math:: σ(QKᵀ)V = A(H, H)⋅X
    """

    def __init__(self, *, dim_context: int, dim_hidden: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.q_proj = nn.Linear(dim_context, dim_hidden)
        self.k_proj = nn.Linear(dim_context, dim_hidden)
        self.dim_hidden = dim_hidden
        self.scale = dim_hidden**-0.5
        self.eps = eps

    def _scores(self, context: Tensor) -> tuple[Tensor, Tensor]:
        Q = self.q_proj(context)  # (..., N, L)
        K = self.k_proj(context)  # (..., N, L)

        scores = torch.einsum("...ML, ...NL -> ...MN", Q, K / self.scale)  # (..., N, N)
        diagonal = scores.diagonal(dim1=-2, dim2=-1)  # (..., N)
        diagonal = F.softplus(diagonal) + self.eps
        # overwrite the diagonal with the new values
        scores = torch.diagonal_scatter(scores, diagonal, dim1=-2, dim2=-1)
        scores = scores.tril()

        # for a linear transform x -> Ax, the logabsdet is |A|
        # since A is triangular with positive diagonal, log|A| = sum(log(diag(A)))
        logabsdet = diagonal.log().sum(dim=-1)  # (...)
        return scores, logabsdet

    def encode_and_logabsdet(
        self,
        x: Tensor,  # (..., N, D)
        context: Tensor,  # (..., N, E)
        /,
    ) -> tuple[Tensor, Tensor]:  # (..., N, D), (...)
        scores, logabsdet = self._scores(context)
        y = torch.einsum("...MN, ...NL -> ...ML", scores, x)
        return y, logabsdet

    def decode_and_logabsdet(
        self,
        y: Tensor,  # (..., N, D)
        context: Tensor,  # (..., N, E)
        /,
    ) -> tuple[Tensor, Tensor]:  # (..., N, D), (...)
        scores, logabsdet = self._scores(context)
        # solve Ax = y for x
        x = torch.linalg.solve_triangular(scores, y, upper=False)
        # note: log|det(A⁻¹)| = log|1/det(A)| = -log|det(A)|
        return x, -logabsdet


class ProFITiBlock(nn.Module):
    r"""Implements profiti-block (equation 15).

    .. math:: shiesh(el(sita(y, x), x)
    """

    scale_net: nn.Module
    shift_net: nn.Module
    attention: TriangularAttention
    shiesh: Shiesh

    def __init__(self, *, latent_dim: int, num_layers: int = 2) -> None:
        super().__init__()

        self.attention = TriangularAttention(
            dim_context=latent_dim,
            dim_hidden=latent_dim,
        )
        self.shiesh = Shiesh(t=1.0, a=1.0)

        self.scale_net = nn.Sequential(
            *chain.from_iterable(
                (
                    nn.Linear(latent_dim, latent_dim),
                    nn.ReLU(),
                )
                for _ in range(num_layers)
            ),
            nn.Linear(latent_dim, 1),
            nn.Tanh(),  # always end with tanh
        )

        self.shift_net = nn.Sequential(
            *chain.from_iterable(
                (
                    nn.Linear(latent_dim, latent_dim),
                    nn.ReLU(),
                )
                for _ in range(num_layers)
            ),
            nn.Linear(latent_dim, 1),
        )

    def encode_and_logabsdet(
        self,
        x: Tensor,
        context: Tensor,
        /,
    ) -> tuple[Tensor, Tensor]:
        # 1. compute sita
        y, ldj_sita = self.attention.encode_and_logabsdet(x, context)

        # 2. compute elementwise linear transformation
        scale = self.scale_net(context)
        shift = self.shift_net(context)
        y = scale.exp() * y + shift
        ldj_scale = scale.sum(dim=-1)

        # 3. apply Shiesh
        y, ldj_shiesh = self.shiesh.encode_and_logabsdet(y)

        return y, (ldj_sita + ldj_scale + ldj_shiesh)


class ProFITiConditioning(nn.Module):
    r"""Stub for ProFITi conditioning modules."""

    def forward(
        self,
        context_times: Tensor,
        context_values: Tensor,
        query_times: Tensor,
        *,
        context_mask: Tensor | None = None,
        query_mask: Tensor | None = None,
    ) -> Tensor:
        raise NotImplementedError


class ProFITiFlow(nn.Module):
    r"""Stub for ProFITi conditional normalizing flows."""

    def forward(
        self,
        value: Tensor,
        conditioning: Tensor,
        *,
        mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        raise NotImplementedError

    def inverse(
        self,
        latent: Tensor,
        conditioning: Tensor,
        *,
        mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        raise NotImplementedError


class ProFITiFlowLayer(nn.Module):
    r"""Stub for a single ProFITi flow layer."""

    def forward(
        self,
        value: Tensor,
        conditioning: Tensor,
        *,
        mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        raise NotImplementedError

    def inverse(
        self,
        latent: Tensor,
        conditioning: Tensor,
        *,
        mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        raise NotImplementedError
