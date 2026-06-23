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
from torch import Tensor, nan, nn
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
    t: Tensor
    a: Tensor

    def __init__(self, t: float = 1.0, a: float = 1.0) -> None:
        super().__init__()
        if a <= 0:
            raise ValueError("a must be positive.")

        self.register_buffer("t", torch.tensor(t))
        self.register_buffer("a", torch.tensor(a))

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
        query: Tensor,  # (..., $K, D), finite
        key: Tensor,  # (..., $K, D), finite
        value: Tensor,  # (..., $K, F), finite
        /,
        *,
        valid_mask: Tensor | None = None,  # (..., $K), bool
    ) -> tuple[Tensor, Tensor]:  # (..., $K, F), (...)
        scores = self._scores(query, key, valid_mask=valid_mask)  # (..., $K, $K)
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

        self.shiesh = Shiesh(t=1.0, a=1.0)

    def encode_and_logabsdet(
        self,
        x: Tensor,  # (..., K)
        context: Tensor,  # (..., K, D)
        /,
    ) -> tuple[Tensor, Tensor]:  # (..., K), (...)
        # zero out NaN values (prevents NaN poisioning)
        # this is OK because the operations are either element-wise, or account for masking
        # nevertheless, in the unit tests we check padding invariance explictly
        valid_mask = context.isfinite().all(dim=-1)  # (..., K), bool
        context = torch.where(valid_mask.unsqueeze(-1), context, 0.0)
        x = torch.where(valid_mask, x, 0.0)

        # 1. compute sita
        y, ldj_sita = self.attention.encode_and_logabsdet(
            context, context, x.unsqueeze(-1), valid_mask=valid_mask
        )
        y = y.squeeze(-1)

        # 2. compute elementwise linear transformation
        scale = self.scale_net(context)
        shift = self.shift_net(context)
        y = scale.exp() * y + shift
        ldj_scale = torch.where(valid_mask, scale, 0.0).sum(dim=-1)

        # 3. apply Shiesh
        y, ldj_shiesh = self.shiesh.encode_and_logabsdet(y)
        ldj_shiesh = torch.where(valid_mask, ldj_shiesh, 0.0).sum(dim=-1)
        y = torch.where(valid_mask, y, nan)

        return y, (ldj_sita + ldj_scale + ldj_shiesh)

    def decode_and_logabsdet(
        self,
        y: Tensor,  # (..., K)
        context: Tensor,  # (..., K, D)
        /,
    ) -> tuple[Tensor, Tensor]:  # (..., K), (...)
        # zero out NaN values (prevents NaN poisioning)
        # this is OK because the operations are either element-wise, or account for masking
        # nevertheless, in the unit tests we check padding invariance explictly
        valid_mask = context.isfinite().all(dim=-1)  # (..., K), bool
        context = torch.where(valid_mask.unsqueeze(-1), context, 0.0)
        y = torch.where(valid_mask, y, 0.0)

        # 1. reverse shiesh
        y, ldj_shiesh = self.shiesh.decode_and_logabsdet(y)
        ldj_shiesh = torch.where(valid_mask, ldj_shiesh, 0.0).sum(dim=-1)

        # 2. reverse elementwise linear transformation
        scale = self.scale_net(context)
        shift = self.shift_net(context)
        y = (y - shift) / scale.exp()
        ldj_scale = -torch.where(valid_mask, scale, 0.0).sum(dim=-1)

        # 3. reverse sita
        y, ldj_sita = self.attention.decode_and_logabsdet(
            context, context, y.unsqueeze(-1), valid_mask=valid_mask
        )
        y = y.squeeze(-1)
        y = torch.where(valid_mask, y, nan)

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
        context_times: Tensor,  # (..., $N), padded NaN
        context_values: Tensor,  # (..., $N, D), padded NaN, sparse
        query_times: Tensor,  # (..., $K), padded NaN
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

        # (..., $K, D)
        Y = X.new_full((*batch_shape, query_size, context_dim), nan)
        time_points = torch.cat([T, Q], dim=-1).nan_to_num(0.0)

        # use grafiti as an encoder (eq 16)
        H = self.context_embedding(
            time_points,  # (..., $N + $K)
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

        # mark samples as NaN if they correspond to non-query values
        valid_mask = H.isfinite().all(dim=-1)  # (..., P)
        Z = torch.where(valid_mask.expand(*Z.shape), Z, nan)
        log_prob = -0.5 * (Z.square() + _LOG2PI).nansum(dim=-1)  # (*S, ...)

        # apply the conditional flow
        samples_flat, logabsdet = self.conditional_flow.decode_and_logabsdet(
            Z,
            H.expand(*sample_shape, *H.shape),
        )  # (*S, ..., P), (*S, ...)

        # reshape sample to (..., $K, D) and fill non-query positions with NaN
        samples = samples_flat.new_full(
            (*sample_shape, *M.shape), nan
        )  # (*S, ..., $K, D)
        samples[..., M] = samples_flat[..., valid_mask]

        return samples, log_prob - logabsdet

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
        values: Tensor,  # (*S, ..., $K, D)
        /,
        *,
        context_times: Tensor,  # (..., $N), padded NaN
        context_values: Tensor,  # (..., $N, D), padded NaN, sparse
        query_times: Tensor,  # (..., $K), padded NaN
    ) -> Tensor:  # (*S, ...)
        r"""Compute the joint log-likelihood of the target values under the model.

        .. math:: \log(p_{Y_{q₁}, ..., Y_{qₖ}}(y_1, ..., y_k ∣ (t₁, x₁), ..., (tₙ, xₙ)))

        The leading ``*S`` dims of ``value`` beyond ``context_values``'s batch
        shape are treated as sample dims; all samples must share the same
        finite/NaN pattern (e.g. samples drawn by :meth:`sample_and_log_prob`).
        """
        # target_shape = sample_shape + batch_shape
        *target_shape, qry_size, _ = values.shape
        *_, ctx_size, ctx_dim = context_values.shape

        T = context_times.broadcast_to(*target_shape, context_times.shape[-1])
        X = context_values.broadcast_to(*target_shape, *context_values.shape[-2:])
        Q = query_times.broadcast_to(*target_shape, query_times.shape[-1])
        M = values.isfinite()
        Y = X.new_full((*target_shape, qry_size, ctx_dim), nan)

        H = self.context_embedding(  # (..., P, H), P = max_target_count
            torch.cat([T, Q], dim=-1).nan_to_num(0.0),
            torch.cat([X, Y], dim=-2),
            torch.cat(
                [
                    M.new_zeros((*target_shape, ctx_size, ctx_dim)),
                    M,
                ],
                dim=-2,
            ),
        )

        target_counts = M.sum(dim=(-2, -1))
        max_target_count = H.shape[-2]  # == target_counts.max()
        target_slots = torch.arange(max_target_count, device=H.device)
        valid_mask = target_slots < target_counts.unsqueeze(dim=-1)
        H = torch.where(valid_mask.unsqueeze(dim=-1), H, nan)

        value_flat = values.new_full(H.shape[:-1], nan)
        value_flat[valid_mask] = values[M]

        Z, logabsdet = self.conditional_flow.encode_and_logabsdet(value_flat, H)
        log_prob = -0.5 * (Z.square() + _LOG2PI).nansum(dim=-1)
        return log_prob + logabsdet
