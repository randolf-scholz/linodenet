r"""ProFITi-style forecasting components."""

__all__ = [
    "ConditionalFlowSequence",
    "ConditionalTransform",
    "ModuleSequence",
    "ProFITi",
    "ProFITiBlock",
    "ProFITiConfig",
    "Shiesh",
    "Transform",
    "TriangularAttention",
]

import math
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from itertools import chain
from typing import TYPE_CHECKING, Any, Final, Protocol, overload

import torch
from torch import Generator, Tensor, nan, nn
from torch.linalg import solve_triangular
from torch.nn import functional as F

from .grafiti import Grafiti
from .utils import EventBatch

_LOG2PI = math.log(2.0 * math.pi)


class ModuleSequence[M: nn.Module](nn.ModuleList, Sequence[M]):
    r"""Wrapper for ModuleList to make it a generic Sequence type."""

    if TYPE_CHECKING:
        _modules: Mapping[str, M]  # type: ignore[assignment]

        # noinspection PyMissingConstructor
        def __init__(self, _: Iterable[M] = (), /) -> None: ...
        def __iter__(self) -> Iterator[M]: ...

    @overload
    def __getitem__(self, index: int, /) -> M: ...  # pyrefly: ignore[bad-override]
    @overload
    def __getitem__(self, index: slice, /) -> ModuleSequence[M]: ...
    def __getitem__(self, index: int | slice, /) -> M | ModuleSequence[M]:  # pyright: ignore[reportIncompatibleMethodOverride]
        if isinstance(index, slice):
            modules = list(self._modules.values())
            selection = modules[index]
            return ModuleSequence(selection)
        return self._modules[self._get_abs_string_index(index)]


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

    def _scores(
        self,
        query: Tensor,  # Float[..., $K, D]
        key: Tensor,  # Float[..., $K, D]
        /,
        *,
        valid_mask: Tensor | None = None,  # Bool[..., $K]
    ) -> Tensor:  # Float[..., $K, $K]
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
        query: Tensor,  # Float[..., $K, D]
        key: Tensor,  # Float[..., $K, D]
        value: Tensor,  # Float[..., $K, F]
        /,
        *,
        valid_mask: Tensor | None = None,  # Bool[..., $K]
    ) -> tuple[Tensor, Tensor]:  # Float[..., $K, F], Float[...]
        scores = self._scores(query, key, valid_mask=valid_mask)  # (..., $K, $K)
        y = torch.einsum("...MN, ...NF -> ...MF", scores, value)
        # for a linear transform x -> Ax, the logabsdet is |A|
        # since A is triangular with positive diagonal, log|A| = sum(log(diag(A)))
        logabsdet = scores.diagonal(dim1=-2, dim2=-1).log().sum(dim=-1)
        return y, logabsdet

    def decode_and_logabsdet(
        self,
        query: Tensor,  # Float[..., $K, D]
        key: Tensor,  # Float[..., $K, D]
        value: Tensor,  # Float[..., $K, F]
        /,
        *,
        valid_mask: Tensor | None = None,  # Bool[..., $K]
    ) -> tuple[Tensor, Tensor]:  # Float[..., $K, F], Float[...]
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
        x: Tensor,  # Float[..., K]
        context: Tensor,  # Float[..., K, D]
        /,
    ) -> tuple[Tensor, Tensor]:  # Float[..., K], Float[...]
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
        y: Tensor,  # Float[..., K]
        context: Tensor,  # Float[..., K, D]
        /,
    ) -> tuple[Tensor, Tensor]:  # Float[..., K], Float[...]
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


class ConditionalFlowSequence(ModuleSequence[ConditionalTransform]):  # type: ignore[type-var]
    r"""Implements a sequence of flow layers."""

    def __init__(self, layers: list[ConditionalTransform], /) -> None:
        super().__init__(layers)

    def encode_and_logabsdet(
        self,
        x: Tensor,
        context: Tensor,
        /,
    ) -> tuple[Tensor, Tensor]:
        batch_shape = x.shape[:-1]
        logabsdet = torch.zeros(batch_shape, dtype=x.dtype, device=x.device)
        for layer in self:
            x, ldj = layer.encode_and_logabsdet(x, context)
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
            y, ldj = layer.decode_and_logabsdet(y, context)
            logabsdet = logabsdet + ldj
        return y, logabsdet


@dataclass(frozen=True)
class ProFITiConfig:
    r"""Configuration for constructing a ProFITi model."""

    input_dim: int = 41
    num_heads: int = 4
    latent_dim: int = 128
    num_layers: int = 2
    num_flow_layers: int = 2


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
            dim_input=config.input_dim,
            dim_latent=config.latent_dim,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            output_mode="embeddings",
        )

        flow = ConditionalFlowSequence(
            [
                ProFITiBlock(latent_dim=config.latent_dim)
                for _ in range(config.num_flow_layers)
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

    def _encode(
        self,
        *,
        query_times: Tensor,  # Float[..., $K], padded NaN, non-decreasing
        query_mask: Tensor,  # Bool[..., $K, F]  padded False
        context_times: Tensor,  # Float[..., $N], padded NaN, non-decreasing
        context_mask: Tensor,  # Bool[..., $N, D], padded False
        context_values: Tensor,  # Float[..., $N, D], padded NaN, sparse
    ) -> tuple[Tensor, Tensor]:  # Float[..., P, latent_dim], Float[..., P]
        request = EventBatch.from_request(
            query_times=query_times,
            query_mask=query_mask,
            context_times=context_times,
            context_mask=context_mask,
            context_values=context_values,
        )

        H = self.context_embedding(  # (..., P, latent_dim)
            # TODO: can we get rid of this nan_to_num?
            timestamps=request.timestamps.nan_to_num(0.0),
            context_values=request.context_values,
            context_mask=request.context_mask,
            query_mask=request.query_mask,
        )

        valid_mask = H.isfinite().all(dim=-1)  # (..., P)
        H = torch.where(valid_mask.unsqueeze(-1), H, nan)
        return H, valid_mask

    def log_prob(
        self,
        values: Tensor,  # Float[*S, ..., $K, F]
        /,
        *,
        query_times: Tensor,  # Float[..., $K], padded NaN, non-decreasing
        query_mask: Tensor,  # Bool[..., $K, F]  padded False
        context_times: Tensor,  # Float[..., $N], padded NaN, non-decreasing
        context_mask: Tensor,  # Bool[..., $N, D], padded False
        context_values: Tensor,  # Float[..., $N, D], padded NaN, sparse
    ) -> Tensor:  # Float[*S, ...]
        r"""Compute the joint log-likelihood of the target values under the model.

        .. math:: \log(p_{Y_{q₁}, ..., Y_{qₖ}}(y_1, ..., y_k ∣ (t₁, x₁), ..., (tₙ, xₙ)))

        The leading ``*S`` dims of ``values`` beyond the batch shape are treated
        as sample dims; all samples must share the same query mask (e.g. samples
        drawn by :meth:`sample_and_log_prob`).
        """
        *shape, qry_size, qry_dim = values.shape  # shape = sample_shape + batch_shape
        *_, ctx_size, ctx_dim = context_values.shape

        M = query_mask.broadcast_to(*shape, qry_size, qry_dim)  # (*S, ..., $K, D)
        H, valid_mask = self._encode(  # (*S, ..., P, latent_dim), (*S, ..., P)
            query_times=query_times.broadcast_to(*shape, qry_size),
            query_mask=query_mask.broadcast_to(*shape, qry_size, qry_dim),
            context_times=context_times.broadcast_to(*shape, ctx_size),
            context_mask=context_mask.broadcast_to(*shape, ctx_size, ctx_dim),
            context_values=context_values.broadcast_to(*shape, ctx_size, ctx_dim),
            # timestamps=timestamps.broadcast_to(*shape, combined_size),
            # context_values=context_values.broadcast_to(*shape, combined_size, D),
            # context_mask=context_mask.broadcast_to(*shape, combined_size, D),
            # query_mask=M,
        )

        value_flat = values.new_full(H.shape[:-1], nan)  # (*S, ..., P)
        value_flat[valid_mask] = values[M]

        Z, logabsdet = self.conditional_flow.encode_and_logabsdet(value_flat, H)
        log_prob = -0.5 * (Z.square() + _LOG2PI).nansum(dim=-1)
        return log_prob + logabsdet

    def sample_and_log_prob(
        self,
        size: int | tuple[int, ...] = (),  # *S
        *,
        query_times: Tensor,  # Float[..., $K], padded NaN, non-decreasing
        query_mask: Tensor,  # Bool[..., $K, F]  padded False
        context_times: Tensor,  # Float[..., $N], padded NaN, non-decreasing
        context_mask: Tensor,  # Bool[..., $N, D], padded False
        context_values: Tensor,  # Float[..., $N, D], padded NaN, sparse
        rng: Generator | None = None,
    ) -> tuple[Tensor, Tensor]:  # Float[*S, ..., $K, D], Float[*S, ...]
        # Shape legend: *S=sample, $T=combined steps, D=channels, P=packed targets
        sample_shape = (size,) if isinstance(size, int) else size

        H, valid_mask = self._encode(  # (..., P, latent_dim), (..., P)
            query_times=query_times,
            query_mask=query_mask,
            context_times=context_times,
            context_mask=context_mask,
            context_values=context_values,
        )

        Z = torch.randn(
            (*sample_shape, *H.shape[:-1]),
            dtype=H.dtype,
            device=H.device,
            generator=rng,
        )
        Z = torch.where(valid_mask.expand_as(Z), Z, nan)
        log_prob = -0.5 * (Z.square() + _LOG2PI).nansum(dim=-1)  # (*S, ...)

        samples_flat, logabsdet = self.conditional_flow.decode_and_logabsdet(
            Z, H.expand(*sample_shape, *H.shape)
        )  # (*S, ..., P), (*S, ...)

        samples = samples_flat.new_full((*sample_shape, *query_mask.shape), nan)
        samples[..., query_mask] = samples_flat[..., valid_mask]  # (*S, ..., $T, D)

        return samples, log_prob - logabsdet

    def sample(
        self,
        size: int | tuple[int, ...] = (),  # *S
        *,
        query_times: Tensor,  # Float[..., $K], padded NaN, non-decreasing
        query_mask: Tensor,  # Bool[..., $K, F]  padded False
        context_times: Tensor,  # Float[..., $N], padded NaN, non-decreasing
        context_mask: Tensor,  # Bool[..., $N, D], padded False
        context_values: Tensor,  # Float[..., $N, D], padded NaN, sparse
        rng: Generator | None = None,
    ) -> Tensor:  # Float[*S, ..., $K, D]
        return self.sample_and_log_prob(
            size=size,
            query_times=query_times,
            query_mask=query_mask,
            context_times=context_times,
            context_mask=context_mask,
            context_values=context_values,
            rng=rng,
        )[0]
