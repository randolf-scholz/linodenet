r"""ProFITi-style forecasting components."""

__all__ = [
    "ProFITi",
    "ProFITiConfig",
    "ConditionalFlowSequence",
    "Shiesh",
    "TriangularAttention",
    "ProFITiBlock",
    "Transform",
    "ConditionalTransform",
]

import math
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import chain
from typing import Any, Final, Protocol

import torch
from torch import Tensor, nn
from torch.linalg import solve_triangular
from torch.nn import functional as F

from .grafiti import Grafiti

_LOG2PI = math.log(2.0 * math.pi)


class Transform(Protocol):
    r"""Protocol for invertible transforms."""

    def encode_and_logabsdet(self, x: Tensor, /) -> tuple[Tensor, Tensor]: ...
    def decode_and_logabsdet(self, y: Tensor, /) -> tuple[Tensor, Tensor]: ...


class ConditionalTransform(Protocol):
    r"""Protocol for invertible conditional transforms."""

    def encode_and_logabsdet(
        self, x: Tensor, context: Tensor, /
    ) -> tuple[Tensor, Tensor]: ...
    def decode_and_logabsdet(
        self, y: Tensor, context: Tensor, /
    ) -> tuple[Tensor, Tensor]: ...


# implements Transform protocol
class Shiesh(nn.Module, Transform):
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

    # (..., $K, D) -> (..., $K, $K)
    def _scores(
        self,
        query: Tensor,
        key: Tensor,
        /,
        *,
        valid_mask: Tensor | None = None,  # (..., $K), bool
    ) -> Tensor:
        if valid_mask is not None:
            assert valid_mask.dtype == torch.bool
            assert valid_mask.shape == query.shape[:-1]
            assert valid_mask.shape == key.shape[:-1]
            query = torch.where(valid_mask.unsqueeze(-1), query, 0.0)
            key = torch.where(valid_mask.unsqueeze(-1), key, 0.0)

        Q = self.q_proj(query)  # (..., $K, L)
        K = self.k_proj(key)  # (..., $K, L)

        scores = torch.einsum("...ML, ...NL -> ...MN", Q, K * self.scale)  # (..., K, K)
        diagonal = scores.diagonal(dim1=-2, dim2=-1)  # (..., $K)
        diagonal = F.softplus(diagonal) + self.eps

        if valid_mask is not None:
            active = valid_mask.unsqueeze(-1) & valid_mask.unsqueeze(-2)
            scores = torch.where(active, scores, 0.0)
            diagonal = torch.where(valid_mask, diagonal, 1.0)

        # overwrite the diagonal with the new values
        scores = torch.diagonal_scatter(scores, diagonal, dim1=-2, dim2=-1)
        return scores.tril()

    def encode_and_logabsdet(
        self,
        query: Tensor,  # (..., $K, D)
        key: Tensor,  # (..., $K, D)
        value: Tensor,  # (..., $K, F)
        /,
        *,
        valid_mask: Tensor | None = None,  # (..., $K), bool
    ) -> tuple[Tensor, Tensor]:  # (..., $K, F), (...)
        scores = self._scores(query, key, valid_mask=valid_mask)  # (..., $K, $K)
        if valid_mask is not None:
            value = torch.where(valid_mask.unsqueeze(-1), value, 0.0)
        y = torch.einsum("...MN, ...NF -> ...MF", scores, value)
        # for a linear transform x -> Ax, the logabsdet is |A|
        # since A is triangular with positive diagonal, log|A| = sum(log(diag(A)))
        logabsdet = scores.diagonal(dim1=-2, dim2=-1).log().sum(dim=-1)
        return y, logabsdet

    def decode_and_logabsdet(
        self,
        query: Tensor,  # (..., $K, D)
        key: Tensor,  # (..., $K, D)
        value: Tensor,  # (..., $K, F)
        /,
        *,
        valid_mask: Tensor | None = None,  # (..., $K), bool
    ) -> tuple[Tensor, Tensor]:  # (..., $K, F), (...)
        scores = self._scores(query, key, valid_mask=valid_mask)  # (..., $K, $K)
        if valid_mask is not None:
            value = torch.where(valid_mask.unsqueeze(-1), value, 0.0)
        # solve Ax = y for x
        x = solve_triangular(scores, value, upper=False)
        logabsdet = -scores.diagonal(dim1=-2, dim2=-1).log().sum(dim=-1)
        # note: log|det(A⁻¹)| = log|1/det(A)| = -log|det(A)|
        return x, logabsdet


class ProFITiBlock(nn.Module, ConditionalTransform):
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
            nn.Flatten(start_dim=-2),
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
            nn.Flatten(start_dim=-2),
        )

    def encode_and_logabsdet(
        self,
        x: Tensor,  # (..., K)
        context: Tensor,  # (..., K, D)
        /,
    ) -> tuple[Tensor, Tensor]:  # (..., K), (...)
        # 1. compute sita
        y, ldj_sita = self.attention.encode_and_logabsdet(
            context,
            context,
            x.unsqueeze(-1),
        )
        y = y.squeeze(-1)

        # 2. compute elementwise linear transformation
        scale = self.scale_net(context)
        shift = self.shift_net(context)
        y = scale.exp() * y + shift
        ldj_scale = scale.sum(dim=-1)

        # 3. apply Shiesh
        y, ldj_shiesh = self.shiesh.encode_and_logabsdet(y)
        ldj_shiesh = ldj_shiesh.sum(dim=-1)

        return y, (ldj_sita + ldj_scale + ldj_shiesh)

    def decode_and_logabsdet(
        self,
        y: Tensor,  # (..., K)
        context: Tensor,  # (..., K, D)
        /,
    ) -> tuple[Tensor, Tensor]:  # (..., K), (...)
        # 1. reverse shiesh
        y, ldj_shiesh = self.shiesh.decode_and_logabsdet(y)
        ldj_shiesh = ldj_shiesh.sum(dim=-1)

        # 2. reverse elementwise linear transformation
        scale = self.scale_net(context)
        shift = self.shift_net(context)
        y = (y - shift) / scale.exp()
        ldj_scale = -scale.sum(dim=-1)

        # 3. reverse sita
        y, ldj_sita = self.attention.decode_and_logabsdet(
            context,
            context,
            y.unsqueeze(-1),
        )
        y = y.squeeze(-1)

        return y, (ldj_sita + ldj_scale + ldj_shiesh)


class ConditionalFlowSequence(nn.ModuleList):
    r"""Implements a sequence of flow layers."""

    def __init__(self, layers: list[ConditionalTransform], /) -> None:
        super().__init__(layers)  # type: ignore[arg-type]

    def encode_and_logabsdet(
        self,
        x: Tensor,
        context: Tensor,
        /,
    ) -> tuple[Tensor, Tensor]:
        batch_shape = x.shape[:-1]
        logabsdet = torch.zeros(batch_shape, dtype=x.dtype, device=x.device)
        for layer in self:
            x, ldj = layer.encode_and_logabsdet(x, context)  # type: ignore[arg-type]
            logabsdet = logabsdet + ldj
        return x, logabsdet

    def decode_and_logabsdet(
        self,
        y: Tensor,
        context: Tensor,
        /,
    ) -> tuple[Tensor, Tensor]:
        batch_shape = y.shape[:-1]
        logabsdet = torch.zeros(batch_shape, dtype=y.dtype, device=y.device)
        for layer in reversed(self):
            y, ldj = layer.decode_and_logabsdet(y, context)  # type: ignore[arg-type]
            logabsdet = logabsdet + ldj
        return y, logabsdet


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
        context_embedding: Module that maps context and query data to flow
            conditioning states.
        conditional_flow: Conditional normalizing flow over target values.
    """

    context_embedding: nn.Module
    conditional_flow: ConditionalTransform

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
            latent_dim=config.latent_dim,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            output_mode="embeddings",
        )

        flow = ConditionalFlowSequence(
            [
                ProFITiBlock(latent_dim=config.latent_dim)
                for _ in range(config.num_layers)
            ]
        )

        return cls(
            context_embedding=conditioning_module,
            conditional_flow=flow,
        )

    def __init__(
        self,
        *,
        context_embedding: nn.Module,
        conditional_flow: ConditionalTransform,
    ) -> None:
        super().__init__()
        self.context_embedding = context_embedding
        self.conditional_flow = conditional_flow

    def sample_and_log_prob(
        self,
        size: int | tuple[int, ...] = (),  # *S
        *,
        context_times: Tensor,  # (..., $N)
        context_values: Tensor,  # (..., $N, D)
        query_times: Tensor,  # (..., $K)
        query_mask: Tensor,  # (..., $K, D), bool
    ) -> tuple[Tensor, Tensor]:  # (*S, ..., $K, D), (*S, ...)
        # Note: Shape legend for the dense ProFITi sampling path
        #   *S: sample shape (variadic, dynamic)
        #   $N: context time steps (dynamic)
        #   $K: query time steps (dynamic)
        #   D: channels.
        #   P: selected target values per batch item
        #   H: latent embedding dimension.
        T = context_times
        X = context_values
        Q = query_times
        M = query_mask
        sample_shape = (size,) if isinstance(size, int) else size

        *batch_shape, context_size, context_dim = X.shape  # ..., $N, D
        query_size = Q.shape[-1]  # $K

        assert M.dtype == torch.bool
        assert M.shape == (*batch_shape, query_size, context_dim)

        target_counts = M.sum(dim=(-2, -1))  # (...)
        min_target_count = int(target_counts.min().item())
        max_target_count = int(target_counts.max().item())
        if min_target_count == 0:
            raise ValueError("query_mask must select at least one target value.")

        if min_target_count != max_target_count:
            raise ValueError(
                "sample_and_log_prob requires the same number of target values "
                "for each batch item."
            )

        # (..., $K, D)
        Y = X.new_full((*batch_shape, query_size, context_dim), torch.nan)

        # use grafiti as an encoder (eq 16)
        H = self.context_embedding(
            torch.cat([T, Q], dim=-1),  # (..., $N + $K)
            torch.cat([X, Y], dim=-2),  # (..., $N + $K, D)
            torch.cat(
                [  # (..., $N + $K, D)
                    M.new_zeros((*batch_shape, context_size, context_dim)),
                    M,
                ],
                dim=-2,
            ),  # fmt: skip,
        )  # (..., P, H), P = max_target_count

        # sample from the latent distribution
        Z = torch.randn(
            (*sample_shape, *H.shape[:-1]),
            dtype=H.dtype,
            device=H.device,
        )  # (*S, ..., P)

        # apply the conditional flow
        samples_flat, logabsdet = self.conditional_flow.decode_and_logabsdet(
            Z,
            H.expand(*sample_shape, *H.shape),
        )  # (*S, ..., P), (*S, ...)

        log_prob = -0.5 * (Z.square() + _LOG2PI).sum(dim=-1) - logabsdet  # (*S, ...)

        samples = samples_flat.new_full(
            (*sample_shape, *M.shape),
            torch.nan,
        )  # (*S, ..., $K, D)
        target_counts_by_time = M.sum(dim=-1)  # (..., $K)
        time_offsets = target_counts_by_time.cumsum(dim=-1) - target_counts_by_time
        channel_offsets = M.cumsum(dim=-1) - 1  # (..., $K, D)
        sample_positions = channel_offsets + time_offsets.unsqueeze(dim=-1)
        *batch_indices, time_indices, channel_indices = M.nonzero(as_tuple=True)
        flow_indices = (
            *batch_indices,
            sample_positions[*batch_indices, time_indices, channel_indices],
        )
        samples[..., *batch_indices, time_indices, channel_indices] = samples_flat[
            ..., *flow_indices
        ]

        return samples, log_prob

    def sample(
        self,
        num_samples: int = 1,
        *,
        context_times: Tensor,
        context_values: Tensor,
        query_times: Tensor,
        query_mask: Tensor | None = None,
    ) -> Tensor:
        raise NotImplementedError

    def log_prob(
        self,
        value: Tensor,
        *,
        context_times: Tensor,
        context_values: Tensor,
        query_times: Tensor,
        query_mask: Tensor | None = None,
    ) -> Tensor:
        raise NotImplementedError
