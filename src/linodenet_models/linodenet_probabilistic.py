r"""Probabilistic Linodenet model."""

__all__ = [
    "update_masked",
    "linear_gaussian_flow",
    "LinodenetProbabilistic",
    "LinearGaussianFlow",
    "KoopmanFilter",
    "make_koopman_filter",
    "make_linodenet_prob",
    "make_linodenet",
]

import warnings
from collections.abc import Callable, Mapping
from typing import Any, Final, Optional

import torch
from torch import Generator, Tensor, einsum, nan, nn
from torch.linalg import cholesky, solve_triangular

from linodenet.state_update import GaussianForwardUpdater, GaussianReverseUpdater

from .decoders import LowRankTransform, Transform, TransformSequence
from .kalman_filter import (
    marginal_gaussian_log_prob,
    marginal_gaussian_sample,
)
from .parametrizations import PositiveDefinite, ReZero, SkewSymmetric, Symmetric
from .profiti import Shiesh
from .utils import EventBatch


def update_masked[R: Tensor | tuple[Tensor, ...]](
    target: R,  # (*(..., *eᵢ),)
    fn: Callable[..., R],  # [*(..., *dᵢ)] -> (*(..., *eᵢ),)
    /,
    *,
    args: tuple[Tensor, ...],
    batch_mask: Tensor,  # Bool[...]
) -> R:  # (*(..., *eᵢ),)
    r"""Update ``target`` with ``fn`` applied to selected batch elements."""
    assert batch_mask.dtype == torch.bool

    ys = fn(*(x[batch_mask] for x in args))
    if isinstance(ys, Tensor):
        assert isinstance(target, Tensor)
        return target.masked_scatter(  # pyrefly: ignore[bad-return]
            batch_mask.reshape(
                *batch_mask.shape, *(1,) * (target.ndim - batch_mask.ndim)
            ),
            ys,
        )

    assert isinstance(target, tuple)
    return tuple(  # type: ignore[return-value]
        t.masked_scatter(
            batch_mask.reshape(*batch_mask.shape, *(1,) * (t.ndim - batch_mask.ndim)), y
        )
        for t, y in zip(target, ys, strict=True)
    )


def linear_gaussian_flow(
    delta_t: Tensor,  # Float[..., $n]
    z0: tuple[Tensor, Tensor],  # Float[..., d], Float[..., d, d]
    A: Tensor,  # Float[d, d]
    Q: Tensor,  # Float[d, d]
    b: Optional[Tensor] = None,  # Float[d]
    /,
) -> tuple[Tensor, Tensor]:  # Float[...., $n, d], Float[..., $n, d, d]
    r"""Propagate a linear-gaussian system.

    .. math:: dZₜ = AZₜdt + bdt + C dWₜ

    Given Z₀∼𝓝(μ₀, Σ₀), then Zₜ∼𝓝(μₜ, Σₜ) for all $t$, with

    .. math::
        μₜ &= eᴬᵗμ₀ + φ₁(At)bt \\
        Σₜ &= eᴬᵗΣ₀eᴬᵀᵗ + ??

    Args:
        delta_t: The time-delta(s) to propagate for, of shape (..., $n).
        z0: initial state given as a pair of mean and covariance matrix,
            of shapes (..., d) and (..., d, d) respectively.
        A: The system matrix of the linear ODE component, of shape (d, d).
        Q: The diffusion matrix of the linear SDE component, of shape (d, d). Must be symmetric positive definite.
        b: Optional affine bias of the linear ODE component, of shape (d,). If None, then no bias is applied.
    """
    mu_0, sigma_0 = z0
    n = A.shape[-1]

    if b is None:
        # [[A, Q], [0, -Aᵀ]]
        # -> [[F, C], [0, F⁻ᵀ]]
        M = torch.cat(
            [
                torch.cat([A, Q], dim=-1),
                torch.cat([torch.zeros_like(A), -A.mT], dim=-1),
            ],
            dim=0,
        )

    else:
        # use augmented block matrix
        # [[A, Q, b], [0, -Aᵀ, 0], [0, 0, 0]]
        # -> [[F, C, r], [0, F⁻ᵀ, 0], [0, 0, 1]]
        b = b.unsqueeze(-1)
        M = torch.cat(
            [
                torch.cat([A, Q, b], dim=-1),
                torch.cat([torch.zeros_like(A), -A.mT, torch.zeros_like(b)], dim=-1),
                torch.zeros((1, 2 * n + 1), dtype=A.dtype, device=A.device),
            ],
            dim=0,
        )

    # exp(M∆t) is a block matrix
    Mdt = einsum("..., kl -> ...kl", delta_t, M)
    P = torch.linalg.matrix_exp(Mdt)
    F = P[..., :n, :n]  # top left block
    C = P[..., :n, n : 2 * n]  # top center block
    r = P[..., :n, -1] if b is not None else 0.0  # top right block
    mu_t = einsum("...nkl, ...l -> ...nk", F, mu_0) + r  # eᴬᵗμ₀ + φ₁(At)bt
    sigma_t = F @ sigma_0.unsqueeze(-3) @ F.mT + C @ F.mT  # eᴬᵗΣ₀eᴬᵀᵗ + CFᵀ

    return mu_t, sigma_t


class LinearGaussianFlow(nn.Module):
    r"""Implements the propagation of a Normal distribution under linear ODE/SDE.

    That is, $z₀∼𝓝(μ₀, Σ₀)$ is propagated under the linear ODE/SDE

    .. math:: dZₜ = AZₜdt + bdt + C dWₜ

    Then, at time $t$ the solution is $zₜ∼𝓝(μₜ, Σₜ)$, where

    .. math::
        μₜ &= eᴬᵗμ₀ + φ₁(At)bt  \\
        Σₜ &= eᴬᵗΣ₀e^{Aᵀt} + ∫₀ᵗ eᴬ⁽ᵗ⁻ˢ⁾ Q e^{Aᵀ(t-s)} ds

    The last integral can be computed in closed form using a block matrix exponential.

    References:
        - | Computing integrals involving the matrix exponential
          | Van Loan
          | IEEE Transactions on Automatic Control, 1978
    """

    input_size: Final[int]
    use_rezero: Final[bool]
    use_bias: Final[bool]

    kernel: Tensor
    bias: Tensor | None
    noise: Tensor  # C = √Q

    kernel_initialization: nn.Module
    r"""MODULE: Optional Initialization of the drift kernel."""
    kernel_parametrization: nn.Module
    r"""MODULE: Optional parametrization of the drift kernel."""

    def __init__(
        self,
        input_size: int,
        *,
        kernel_initialization: str | Tensor | nn.Module = "skew-symmetric",
        kernel_parametrization: Optional[str | nn.Module] = None,
        use_rezero: bool = True,
        use_bias: bool = False,
    ) -> None:
        super().__init__()
        warnings.warn(
            "Using inefficient research implementation without parametrization caching.",
            stacklevel=2,
        )
        self.input_size = input_size
        self.use_rezero = use_rezero
        self.use_bias = use_bias

        self.kernel_weight = nn.Parameter(self._init_kernel(kernel_initialization))
        self.kernel_parametrization = self._get_parametrization(kernel_parametrization)
        self.register_buffer("kernel", self.kernel_parametrization(self.kernel_weight))

        self.register_parameter(
            "bias",
            nn.Parameter(torch.zeros(input_size)) if self.use_bias else None,
        )

        self.noise_weight = nn.Parameter(torch.randn(input_size, input_size))
        self.noise_parametrization = self._get_parametrization("positive-definite")
        self.register_buffer("noise", self.noise_parametrization(self.noise_weight))

    def _init_kernel(self, init: str | Tensor | nn.Module, /) -> Tensor:
        match init:
            case None:
                return torch.randn(self.input_size, self.input_size)
            case "zero":
                return torch.zeros(self.input_size, self.input_size)
            case "skew-symmetric":
                kernel = torch.randn(self.input_size, self.input_size)
                return (kernel - kernel.mT) / 2
            case "symmetric":
                kernel = torch.randn(self.input_size, self.input_size)
                return (kernel + kernel.mT) / 2
            case _:
                raise ValueError

    def _get_parametrization(self, param: str | nn.Module | None, /) -> nn.Module:
        match param:
            case None | "identity":
                parametrization = nn.Identity()
            case "symmetric":
                parametrization = Symmetric()
            case "skew-symmetric":
                parametrization = SkewSymmetric()
            case "positive-definite":
                parametrization = PositiveDefinite()
            case _:
                raise NotImplementedError

        return nn.Sequential(
            parametrization,
            ReZero() if self.use_rezero else nn.Identity(),
        )

    def propagate(
        self, delta_t: Tensor, z_0: tuple[Tensor, Tensor]
    ) -> tuple[Tensor, Tensor]:
        r"""Propagate the linear-Gaussian system for one or more time-deltas."""
        self.kernel = self.kernel_parametrization(self.kernel_weight)
        self.noise = self.noise_parametrization(self.noise_weight)
        return linear_gaussian_flow(delta_t, z_0, self.kernel, self.noise, self.bias)

    def forward(
        self, delta_t: Tensor, z_0: tuple[Tensor, Tensor], /
    ) -> tuple[Tensor, Tensor]:
        r"""Propagate for a single step."""
        next_mean, next_cov = self.propagate(delta_t.unsqueeze(-1), z_0)
        return next_mean.squeeze(-2), next_cov.squeeze(-3)


class LinodenetProbabilistic(nn.Module):
    r"""Latent Gaussian-linear ODE Network.

    - latent linear gaussian system
    - Normalizing flow decoder
    - special update rule for the latent parameters.

    This version does not support missing values.
    """

    decoder: Transform  # normalizing flow

    input_size: Final[int]
    latent_size: Final[int]

    initial_mean: Tensor
    initial_cov: Tensor
    initial_cov_parametrization: nn.Module

    # BUFFERS
    prior_means: Tensor
    prior_covs: Tensor
    posterior_means: Tensor
    posterior_covs: Tensor

    def __init__(
        self,
        input_size: int,
        *,
        decoder: nn.Module,
        state_updater: nn.Module,
        state_propagator: nn.Module,
        batch_first: bool = True,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.latent_size = input_size  # hard modelling constraint
        self.batch_first = batch_first

        self.decoder = decoder  # type: ignore[assignment]
        self.state_updater = state_updater
        self.state_propagator = state_propagator

        self.initial_mean = nn.Parameter(torch.zeros(self.latent_size))
        self.initial_cov = nn.Parameter(torch.zeros(self.latent_size, self.latent_size))
        self.initial_cov_parametrization = PositiveDefinite()

        self.register_buffer("prior_means", None, persistent=False)
        self.register_buffer("prior_covs", None, persistent=False)
        self.register_buffer("posterior_means", None, persistent=False)
        self.register_buffer("posterior_covs", None, persistent=False)

    def forward(
        self,
        *,
        timestamps: Tensor,  # Float[..., $T], padded NaN, non-decreasing
        query_mask: Tensor,  # Bool[..., $T, F], padded False
        context_values: Tensor,  # Float[..., $T, D], padded Nan, sparse
        context_mask: Tensor,  # Bool[..., $T, D], padded False
        initial_state: tuple[Tensor, Tensor] | None = None,  # Float[..., L]
        initial_time: Tensor | None = None,  # t₀, () or (...)
    ) -> tuple[Tensor, Tensor]:  # Float[..., $T, L], Float[..., $T, L, L]
        r"""Compute the posterior latent states.

        Args:
            timestamps: The timestamps of the observations, of shape (..., $T).
                Must be padded with NaN for missing values and must be non-decreasing.
            query_mask: The mask indicating which timestamps are valid queries,
                of shape (..., $T, F). Must be padded with False for missing values.
            context_values: The observed context values, of shape (..., $T, D).
                Must be padded with NaN for missing values and must be sparse.
            context_mask: The mask indicating which context values are valid,
                of shape (..., $T, D). Must be padded with False for missing values.
            initial_state: Optional initial state of the latent ODE,
                of shape (..., L). If None, then the initial state is inferred from the context.
            initial_time: Optional initial time of the latent ODE,
                of shape () or (...). If None, then the initial time is inferred from the timestamps.

        Returns:
            A tuple of the predicted latent states and their covariances,
        """
        # does not support missing values.
        assert torch.equal(context_mask.any(dim=-1), context_mask.all(dim=-1))
        seq_dim = -2 if self.batch_first else -1
        T = timestamps[..., None].movedim(seq_dim, 0).squeeze(-1)  # ($N, ...)
        X = context_values.movedim(seq_dim, 0)  # ($N, ..., D)
        Q = query_mask.movedim(seq_dim, 0)
        M = context_mask.movedim(seq_dim, 0)
        T0 = (
            T[[0]]
            if initial_time is None
            else initial_time.expand_as(T[0]).unsqueeze(0)
        )
        DT = T.diff(dim=0, prepend=T0)
        valid_steps = (M | Q).any(dim=-1)
        _, *batch_shape = T.shape

        prior_means: list[Tensor] = []
        prior_covs: list[Tensor] = []
        posterior_means: list[Tensor] = []
        posterior_covs: list[Tensor] = []

        post_state = (
            initial_state
            if initial_state is not None
            else (
                self.initial_mean.expand(*batch_shape, self.latent_size),
                self.initial_cov_parametrization(self.initial_cov).expand(
                    *batch_shape, self.latent_size, self.latent_size
                ),
            )
        )

        for delta_t, x_obs, obs_mask, active in zip(DT, X, M, valid_steps, strict=True):
            prior_state = update_masked(
                post_state,
                lambda dt, mean, cov: self.state_propagator(dt, (mean, cov)),
                args=(delta_t, *post_state),
                batch_mask=active,
            )

            post_state = update_masked(
                prior_state,
                lambda y, mean, cov: self.state_updater(y, (mean, cov)),
                args=(x_obs, *prior_state),
                batch_mask=obs_mask.any(dim=-1),
            )

            prior_means.append(prior_state[0])
            prior_covs.append(prior_state[1])
            posterior_means.append(post_state[0])
            posterior_covs.append(post_state[1])

        stack_dim_mean = -2 if self.batch_first else 0
        stack_dim_cov = -3 if self.batch_first else 0
        self.prior_means = torch.stack(prior_means, dim=stack_dim_mean)
        self.prior_covs = torch.stack(prior_covs, dim=stack_dim_cov)
        self.posterior_means = torch.stack(posterior_means, dim=stack_dim_mean)
        self.posterior_covs = torch.stack(posterior_covs, dim=stack_dim_cov)

        return self.posterior_means, self.posterior_covs

    def predict(
        self,
        *,
        query_times: Tensor,  # Float[..., $K], padded NaN, non-decreasing
        query_mask: Tensor,  # Bool[..., $K, F]  padded False
        context_times: Tensor,  # Float[..., $N], padded NaN, non-decreasing
        context_mask: Tensor,  # Bool[..., $N, D], padded False
        context_values: Tensor,  # Float[..., $N, D], padded NaN, sparse
        initial_state: tuple[Tensor, Tensor] | None = None,  # (..., 2d), (..., d, 3)
        initial_time: Tensor | None = None,  # t₀, () or (...)
    ) -> tuple[Tensor, Tensor]:  # Float[..., $K, D], Float[..., $K, D, D]
        r"""Compute the posterior latent states at the query times."""
        combined = EventBatch.from_request(
            context_times=context_times,
            context_values=context_values,
            context_mask=context_mask,
            query_times=query_times,
            query_mask=query_mask,
            batch_first=self.batch_first,
        )
        post_means, post_covs = self.forward(
            timestamps=combined.timestamps,  # Float[..., $T], padded NaN, non-decreasing
            context_values=combined.context_values,  # Float[..., $T, D], padded NaN, sparse
            context_mask=combined.context_mask,  # Bool[..., $T, D], padded False
            query_mask=combined.query_mask,  # Bool[..., $T, F], padded False
            initial_state=initial_state,
            initial_time=initial_time,
        )
        post_means = post_means[..., *combined.query_indices, :]
        post_covs = post_covs[..., *combined.query_indices, :, :]
        return post_means, post_covs

    def log_prob(
        self,
        values: Tensor,  # Float[..., $K, F]
        /,
        *,
        query_times: Tensor,  # Float[..., $K], padded NaN, non-decreasing
        query_mask: Tensor,  # Bool[..., $K, F]  padded False
        context_times: Tensor,  # Float[..., $N], padded NaN, non-decreasing
        context_values: Tensor,  # Float[..., $N, D], padded NaN, sparse
        context_mask: Tensor,  # Bool[..., $N, D], padded False
        initial_state: tuple[Tensor, Tensor] | None = None,
        initial_time: Tensor | None = None,
    ) -> Tensor:  # Float[..., $K]
        r"""Compute the time-marginal log-likelihood of the model.

        .. math:: pₖ = p_{Y_{qₖ}}(yₖ | (t₁, y₁), ..., (tₙ, yₙ))
        """
        mean, cov = self.predict(
            query_times=query_times,
            query_mask=query_mask,
            context_times=context_times,
            context_values=context_values,
            context_mask=context_mask,
            initial_state=initial_state,
            initial_time=initial_time,
        )
        mask = query_mask.expand(*values.shape)
        assert torch.equal(mask.any(dim=-1), mask.all(dim=-1)), (
            "LinodenetProbabilistic requires all query features present or none."
        )
        safe_values = torch.where(mask, values, torch.zeros_like(values))
        z, ldj = self.decoder.encode_and_logabsdet(safe_values)
        base_log_prob = marginal_gaussian_log_prob(
            z,
            mean=mean.expand(*values.shape),
            cov=cov.expand(*values.shape[:-1], *cov.shape[-2:]),
            mask=mask,
        )
        return torch.where(mask.any(dim=-1), base_log_prob + ldj, 0.0)

    def sample(
        self,
        size: int | tuple[int, ...] = (),  # *S
        *,
        query_times: Tensor,  # Float[..., $K], padded NaN, non-decreasing
        query_mask: Tensor,  # Bool[..., $K, F]  padded False
        context_times: Tensor,  # Float[..., $N], padded NaN, non-decreasing
        context_values: Tensor,  # Float[..., $N, D], padded NaN, sparse
        context_mask: Tensor,  # Bool[..., $N, D], padded False
        initial_state: tuple[Tensor, Tensor] | None = None,
        initial_time: Tensor | None = None,
        rng: Generator | None = None,
    ) -> Tensor:  # Float[*S, ..., $K, F]
        r"""Sample from the time-marginal predictive distribution."""
        sample_shape = (size,) if isinstance(size, int) else size
        mean, cov = self.predict(
            query_times=query_times,
            query_mask=query_mask,
            context_times=context_times,
            context_values=context_values,
            context_mask=context_mask,
            initial_state=initial_state,
            initial_time=initial_time,
        )
        assert torch.equal(query_mask.any(dim=-1), query_mask.all(dim=-1)), (
            "LinodenetProbabilistic requires all query features present or none."
        )
        z = marginal_gaussian_sample(
            sample_shape, mean=mean, cov=cov, mask=query_mask, rng=rng
        )
        y, _ = self.decoder.decode_and_logabsdet(z.nan_to_num(0.0))
        return y.masked_fill(~query_mask.expand(*sample_shape, *query_mask.shape), nan)

    def sample_and_log_prob(
        self,
        size: int | tuple[int, ...] = (),  # *S
        *,
        query_times: Tensor,  # Float[..., $K], padded NaN, non-decreasing
        query_mask: Tensor,  # Bool[..., $K, F]  padded False
        context_times: Tensor,  # Float[..., $N], padded NaN, non-decreasing
        context_values: Tensor,  # Float[..., $N, D], padded NaN, sparse
        context_mask: Tensor,  # Bool[..., $N, D], padded False
        initial_state: tuple[Tensor, Tensor] | None = None,
        initial_time: Tensor | None = None,
        rng: Generator | None = None,
    ) -> tuple[Tensor, Tensor]:  # Float[*S, ..., $K, F], Float[*S, ..., $K]
        r"""Sample from the model and compute sample log-probabilities."""
        samples = self.sample(
            size,
            query_times=query_times,
            query_mask=query_mask,
            context_times=context_times,
            context_values=context_values,
            context_mask=context_mask,
            initial_state=initial_state,
            initial_time=initial_time,
            rng=rng,
        )
        return samples, self.log_prob(
            samples,
            query_times=query_times,
            query_mask=query_mask,
            context_times=context_times,
            context_values=context_values,
            context_mask=context_mask,
            initial_state=initial_state,
            initial_time=initial_time,
        )


def make_linodenet(
    *,
    input_size: int,
    state_updater: str = "forward",
    retention: Tensor | float | tuple[Tensor | float, Tensor | float] = 0.5,
    retention_learnable: bool = True,
    decoder: str | TransformSequence = "shiesh",
    low_rank: int | None = None,
    batch_first: bool = True,
    state_propagator: Mapping[str, Any] | None = None,
) -> LinodenetProbabilistic:
    r"""Backward-compatible alias for :func:`make_linodenet_prob`."""
    return make_linodenet_prob(
        input_size=input_size,
        state_update=state_updater,
        retention=retention,
        retention_learnable=retention_learnable,
        decoder=decoder,
        low_rank=low_rank,
        batch_first=batch_first,
        state_propagator=state_propagator,
    )


def make_linodenet_prob(
    *,
    input_size: int,
    state_update: str = "forward",
    retention: Tensor | float | tuple[Tensor | float, Tensor | float] = 0.5,
    retention_learnable: bool = True,
    decoder: str | TransformSequence = "shiesh",
    low_rank: int | None = None,
    batch_first: bool = True,
    state_propagator: Mapping[str, Any] | None = None,
) -> LinodenetProbabilistic:
    r"""Instantiate the probabilistic Linodenet demo model."""
    if isinstance(decoder, TransformSequence):
        decoder_module = decoder
    else:
        match decoder:
            case "shiesh":
                decoder_module = TransformSequence([Shiesh(t=1.0, a=1.0)])
            case "shiesh-lowrank-shiesh":
                rank = 1 if low_rank is None else low_rank
                decoder_module = TransformSequence(
                    [
                        Shiesh(t=1.0, a=1.0),
                        LowRankTransform(input_size, rank=rank),
                        Shiesh(t=1.0, a=1.0),
                    ]
                )
            case _:
                raise ValueError(f"Unknown decoder: {decoder!r}.")

    updater_kwargs = {
        "decoder": decoder_module,
        "parametrization": "covariance",
        "retention": retention,
        "retention_learnable": retention_learnable,
    }
    match state_update:
        case "forward":
            updater = GaussianForwardUpdater(**updater_kwargs)
        case "reverse":
            updater = GaussianReverseUpdater(**updater_kwargs)
        case _:
            raise ValueError(f"Unknown state update method: {state_update!r}.")

    propagator_kwargs = {
        "input_size": input_size,
        "kernel_initialization": "zero",
        "kernel_parametrization": "identity",
        "use_rezero": False,
        "use_bias": False,
    }
    if state_propagator is not None:
        propagator_kwargs.update(state_propagator)
    propagator = LinearGaussianFlow(**propagator_kwargs)

    return LinodenetProbabilistic(
        input_size=input_size,
        decoder=decoder_module,
        state_updater=updater,
        state_propagator=propagator,
        batch_first=batch_first,
    )


class Augment(nn.Module):
    def __init__(
        self, input_size: int, latent_size: int, *, encoder: nn.Module
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.latent_size = latent_size
        self.encoder = encoder

    def encode_and_logabsdet(self, x: Tensor, /) -> Tensor:
        e, ldj = self.encoder.sample_and_log_prob(x.shape[:-1])
        return torch.cat([x, e], dim=self.dim), ldj

    def decode(self, z: Tensor, /) -> Tensor:
        return z[..., : self.input_size]  # only keep the first n values


class KoopmanFilter(nn.Module):
    r"""Working title.

    - high dimensional latent space ℝᴺ, low dimensional data space ℝⁿ
    - latent linear gaussian system.
    - non-linear decoder g: ℝᴺ -> ℝᴺ (possibly normalizing flow)
    - y = π(g(u)) + ε, π projection that drops last N-n coordinates, ε∼𝓝(0, R) noise

    Bound on the log-likelihood of partial observations:

    .. math:: log p(y_obs) ≥ 𝐄_{z∼q(z)} [ log 𝓝(y_obs ∣ μ_obs(z), R_obs) ]
        - KL(q(z) || 𝓝(μₜ⁻, Σₜ⁻))
    """

    input_size: Final[int]
    latent_size: Final[int]

    initial_mean: Tensor
    initial_cov: Tensor
    initial_cov_parametrization: nn.Module

    # BUFFERS
    prior_means: Tensor
    prior_covs: Tensor
    posterior_means: Tensor
    posterior_covs: Tensor

    def __init__(
        self,
        input_size: int,
        latent_size: int,
        *,
        decoder: nn.Module,
        state_propagator: nn.Module,
        observation_noise: float = 0.5,
        n_iter: int = 1,
        max_num_samples: int = 256,
        batch_first: bool = True,
    ) -> None:
        super().__init__()
        if input_size <= 0 or latent_size <= 0 or latent_size < input_size:
            raise ValueError(f"Invalid {input_size=}, {latent_size=}")
        if observation_noise <= 0:
            raise ValueError(
                f"Expected observation_noise > 0, got {observation_noise=}."
            )
        if n_iter < 1:
            raise ValueError(f"Expected n_iter >= 1, got {n_iter=}.")
        if max_num_samples < 1:
            raise ValueError(f"Expected max_num_samples >= 1, got {max_num_samples=}.")

        self.input_size = input_size
        self.latent_size = latent_size
        self.decoder = decoder
        self.state_propagator = state_propagator
        self.n_iter = n_iter
        self.max_num_samples = max_num_samples
        self.batch_first = batch_first

        self.initial_mean = nn.Parameter(torch.zeros(self.latent_size))
        self.initial_cov = nn.Parameter(torch.zeros(self.latent_size, self.latent_size))
        self.initial_cov_parametrization = PositiveDefinite()
        self.observation_log_variance = nn.Parameter(
            torch.full((input_size,), 2.0 * torch.log(torch.tensor(observation_noise)))
        )

        # Common random numbers make Monte Carlo bounds reproducible across
        # equivalent padded and unpadded batches.
        self.register_buffer(
            "mc_noise",
            torch.randn(max_num_samples, latent_size),
            persistent=False,
        )

        self.register_buffer("prior_means", None, persistent=False)
        self.register_buffer("prior_covs", None, persistent=False)
        self.register_buffer("posterior_means", None, persistent=False)
        self.register_buffer("posterior_covs", None, persistent=False)

    @property
    def observation_covariance(self) -> Tensor:
        r"""Return the diagonal observation-noise covariance."""
        return (
            self.observation_log_variance.exp()
            + torch.finfo(self.initial_mean.dtype).eps
        ).diag_embed()

    def observation(self, z: Tensor, /) -> Tensor:
        r"""Decode latent states and retain the observed slice."""
        match self.decoder:
            case TransformSequence() as decoder:
                value = z
                for layer in reversed(decoder):
                    value = layer.decode(value)  # type: ignore[attr-defined]
            case decoder if hasattr(decoder, "decode"):
                value = decoder.decode(z)  # type: ignore[attr-defined]
            case decoder:
                value = decoder(z)
        return value[..., : self.input_size]

    def _kl_gaussian(
        self,
        posterior: tuple[Tensor, Tensor],
        prior: tuple[Tensor, Tensor],
        /,
    ) -> Tensor:
        r"""Compute ``KL(posterior || prior)`` for batched dense Gaussians."""
        mean, cov = posterior
        prior_mean, prior_cov = prior
        prior_scale = cholesky(prior_cov)
        posterior_scale = cholesky(cov)
        delta = mean - prior_mean
        trace = (
            torch.cholesky_solve(cov, prior_scale)
            .diagonal(dim1=-2, dim2=-1)
            .sum(dim=-1)
        )
        quadratic = (
            delta * torch.cholesky_solve(delta.unsqueeze(-1), prior_scale).squeeze(-1)
        ).sum(dim=-1)
        logdet_prior = 2.0 * prior_scale.diagonal(dim1=-2, dim2=-1).log().sum(dim=-1)
        logdet_posterior = 2.0 * posterior_scale.diagonal(dim1=-2, dim2=-1).log().sum(
            dim=-1
        )
        return 0.5 * (
            trace + quadratic - self.latent_size + logdet_prior - logdet_posterior
        )

    def _observation_log_prob(
        self, values: Tensor, means: Tensor, mask: Tensor, /
    ) -> Tensor:
        r"""Evaluate diagonal Normal observation densities with arbitrary masks."""
        variance = self.observation_covariance.diagonal(dim1=-2, dim2=-1)
        residual = torch.where(mask, values - means, torch.zeros_like(means))
        log_prob = -0.5 * (
            residual.square() / variance
            + variance.log()
            + torch.log(
                torch.tensor(2.0 * torch.pi, dtype=values.dtype, device=values.device)
            )
        )
        return torch.where(mask, log_prob, torch.zeros_like(log_prob)).sum(dim=-1)

    def update_iekf(
        self,
        y_obs: Tensor,
        state: tuple[Tensor, Tensor],
        /,
        *,
        mask: Tensor | None = None,
        n_iter: int = 3,
    ) -> tuple[Tensor, Tensor]:
        r"""Update a Gaussian state using a masked iterated extended Kalman filter.

        The update applies Gauss--Newton to the MAP objective induced by the
        nonlinear Normal observation model. It is a Laplace approximation, not
        the exact minimizer of the Gaussian ELBO returned by :meth:`log_prob`.

        The iteration is:

            # r = y - h(μ⁽ⁱ⁻¹⁾) - 𝐃h(μ⁽ⁱ⁻¹⁾)(μ₋ - μ⁽ⁱ⁻¹⁾)
            # S = 𝐃h(μ⁽ⁱ⁻¹⁾) Σ₋ 𝐃h(μ⁽ⁱ⁻¹⁾)ᵀ + R
            # K = Σ₋ 𝐃h(μ⁽ⁱ⁻¹⁾)ᵀ S⁻¹
            # μ⁽ⁱ⁾ = μ₋ + K r
            # Σ⁽ⁱ⁾ = Σ₋ - K S Kᵀ

        References:
            - | Algorithm 7.9, Bayesian Filtering and Smoothing, 2nd Edition
              | Simo Särkkä and Lennart Svensson
        """
        assert n_iter >= 1
        assert mask is None or (mask.shape == y_obs.shape and mask.dtype == torch.bool)

        batch_shape = y_obs.shape[:-1]

        μ_prior, Σ_prior = state
        μ = μ_prior

        R = self.observation_covariance.expand(
            *batch_shape, self.input_size, self.input_size
        )
        L = cholesky(Σ_prior)

        if mask is not None:
            y_obs = torch.where(mask, y_obs, 0.0)
            cov_mask = mask.unsqueeze(-1) & mask.unsqueeze(-2)
            R = torch.where(cov_mask, R, 0.0) + (~mask).to(dtype=R.dtype).diag_embed()

        for _ in range(n_iter):
            # Pack the N Cholesky columns as a batch of tangent directions.
            # This computes H L without materializing the full H (..., n, N).
            h, HL_transposed = torch.func.jvp(
                self.observation,
                (μ.unsqueeze(-2).expand_as(L.mT).clone(),),
                (L.mT,),
            )
            HL = HL_transposed.mT
            h = h[..., 0, :]
            δ = solve_triangular(L, (μ_prior - μ).unsqueeze(-1), upper=False).squeeze(
                -1
            )
            r = y_obs - h - einsum("...ij, ...j -> ...i", HL, δ)
            if mask is not None:
                HL = torch.where(mask[..., None], HL, 0.0)
                r = torch.where(mask, r, 0.0)

            ΣHt = L @ HL.mT
            K = torch.linalg.solve(HL @ HL.mT + R, ΣHt.mT).mT
            μ = μ_prior + einsum("...ij, ...j -> ...i", K, r)

        Σ = Σ_prior - ΣHt @ K.mT  # type: ignore[unbound-name]
        Σ = (Σ + Σ.mT) / 2  # ensure symmetry.
        return μ, Σ

    def update(
        self,
        y_obs: Tensor,
        state: tuple[Tensor, Tensor],
        /,
        *,
        mask: Tensor | None = None,
        n_iter: int = 3,
    ) -> tuple[Tensor, Tensor]:
        r"""Update the state given an observation."""
        return self.update_iekf(y_obs, state, mask=mask, n_iter=n_iter)

    def forward(
        self,
        *,
        timestamps: Tensor,  # Float[..., $T], padded NaN, non-decreasing
        query_mask: Tensor,  # Bool[..., $T, F], padded False
        context_values: Tensor,  # Float[..., $T, D], padded Nan, sparse
        context_mask: Tensor,  # Bool[..., $T, D], padded False
        initial_state: tuple[Tensor, Tensor] | None = None,  # Float[..., L]
        initial_time: Tensor | None = None,  # t₀, () or (...)
    ) -> tuple[Tensor, Tensor]:  # Float[..., $T, L], Float[..., $T, L, L]
        r"""Compute the posterior latent states."""
        seq_dim = -2 if self.batch_first else -1
        T = timestamps[..., None].movedim(seq_dim, 0).squeeze(-1)  # ($N, ...)
        X = context_values.movedim(seq_dim, 0)  # ($N, ..., D)
        Q = query_mask.movedim(seq_dim, 0)
        M = context_mask.movedim(seq_dim, 0)
        T0 = (
            T[[0]]
            if initial_time is None
            else initial_time.expand_as(T[0]).unsqueeze(0)
        )
        DT = T.diff(dim=0, prepend=T0)
        valid_steps = (M | Q).any(dim=-1)
        _, *batch_shape = T.shape

        prior_means: list[Tensor] = []
        prior_covs: list[Tensor] = []
        posterior_means: list[Tensor] = []
        posterior_covs: list[Tensor] = []

        post_state = (
            initial_state
            if initial_state is not None
            else (
                self.initial_mean.expand(*batch_shape, self.latent_size),
                self.initial_cov_parametrization(self.initial_cov).expand(
                    *batch_shape, self.latent_size, self.latent_size
                ),
            )
        )

        for delta_t, x_obs, obs_mask, active in zip(DT, X, M, valid_steps, strict=True):
            prior_state = update_masked(
                post_state,
                lambda dt, mean, cov: self.state_propagator(dt, (mean, cov)),
                args=(delta_t, *post_state),
                batch_mask=active,
            )

            post_state = update_masked(
                prior_state,
                lambda y, mean, cov, mask: self.update(
                    y, (mean, cov), mask=mask, n_iter=self.n_iter
                ),
                args=(x_obs, *prior_state, obs_mask),
                batch_mask=obs_mask.any(dim=-1),
            )

            prior_means.append(prior_state[0])
            prior_covs.append(prior_state[1])
            posterior_means.append(post_state[0])
            posterior_covs.append(post_state[1])

        stack_dim_mean = -2 if self.batch_first else 0
        stack_dim_cov = -3 if self.batch_first else 0
        self.prior_means = torch.stack(prior_means, dim=stack_dim_mean)
        self.prior_covs = torch.stack(prior_covs, dim=stack_dim_cov)
        self.posterior_means = torch.stack(posterior_means, dim=stack_dim_mean)
        self.posterior_covs = torch.stack(posterior_covs, dim=stack_dim_cov)

        return self.posterior_means, self.posterior_covs

    def predict(
        self,
        *,
        query_times: Tensor,  # Float[..., $K], padded NaN, non-decreasing
        query_mask: Tensor,  # Bool[..., $K, F]  padded False
        context_times: Tensor,  # Float[..., $N], padded NaN, non-decreasing
        context_mask: Tensor,  # Bool[..., $N, D], padded False
        context_values: Tensor,  # Float[..., $N, D], padded NaN, sparse
        initial_state: tuple[Tensor, Tensor] | None = None,  # (..., 2d), (..., d, 3)
        initial_time: Tensor | None = None,  # t₀, () or (...)
    ) -> tuple[Tensor, Tensor]:  # Float[..., $K, D], Float[..., $K, D, D]
        r"""Compute the posterior latent states at the query times."""
        combined = EventBatch.from_request(
            context_times=context_times,
            context_values=context_values,
            context_mask=context_mask,
            query_times=query_times,
            query_mask=query_mask,
            batch_first=self.batch_first,
        )
        post_means, post_covs = self.forward(
            timestamps=combined.timestamps,  # Float[..., $T], padded NaN, non-decreasing
            context_values=combined.context_values,  # Float[..., $T, D], padded NaN, sparse
            context_mask=combined.context_mask,  # Bool[..., $T, D], padded False
            query_mask=combined.query_mask,  # Bool[..., $T, F], padded False
            initial_state=initial_state,
            initial_time=initial_time,
        )
        post_means = post_means[..., *combined.query_indices, :]
        post_covs = post_covs[..., *combined.query_indices, :, :]
        return post_means, post_covs

    def log_prob(
        self,
        values: Tensor,  # Float[..., $K, F]
        /,
        *,
        query_times: Tensor,  # Float[..., $K], padded NaN, non-decreasing
        query_mask: Tensor,  # Bool[..., $K, F]  padded False
        context_times: Tensor,  # Float[..., $N], padded NaN, non-decreasing
        context_values: Tensor,  # Float[..., $N, D], padded NaN, sparse
        context_mask: Tensor,  # Bool[..., $N, D], padded False
        initial_state: tuple[Tensor, Tensor] | None = None,
        initial_time: Tensor | None = None,
        num_samples: int = 16,
    ) -> Tensor:  # Float[..., $K]
        r"""Estimate a per-timestamp ELBO for noisy marginal observations.

        The nonlinear decoder makes the predictive density intractable in
        general. This therefore returns a Monte Carlo estimate of a *lower
        bound* on ``log p(values | context)`` rather than an exact log-density.
        ``num_samples`` controls the reparameterized expectation under the
        Gaussian iEKF posterior; it must not exceed ``max_num_samples``.
        """
        return self.log_prob_bound(
            values,
            query_times=query_times,
            query_mask=query_mask,
            context_times=context_times,
            context_values=context_values,
            context_mask=context_mask,
            initial_state=initial_state,
            initial_time=initial_time,
            num_samples=num_samples,
        )

    def log_prob_bound(
        self,
        values: Tensor,
        /,
        *,
        query_times: Tensor,
        query_mask: Tensor,
        context_times: Tensor,
        context_values: Tensor,
        context_mask: Tensor,
        initial_state: tuple[Tensor, Tensor] | None = None,
        initial_time: Tensor | None = None,
        num_samples: int = 16,
    ) -> Tensor:
        r"""Return the Monte Carlo ELBO used by :meth:`log_prob`.

        This is an estimated lower bound, not an exact marginal log-likelihood.
        Common random numbers are used only to make equivalent batch layouts
        produce identical estimates.
        """
        if not 1 <= num_samples <= self.max_num_samples:
            raise ValueError(f"Expected 1 <= num_samples <= {self.max_num_samples}.")

        mean_prior, cov_prior = self.predict(
            query_times=query_times,
            query_mask=query_mask,
            context_times=context_times,
            context_values=context_values,
            context_mask=context_mask,
            initial_state=initial_state,
            initial_time=initial_time,
        )
        mask = query_mask.expand_as(values)
        posterior = self.update(
            values,
            (mean_prior, cov_prior),
            mask=mask,
            n_iter=self.n_iter,
        )
        mean, cov = posterior
        scale = cholesky(cov)
        noise = self.mc_noise[:num_samples].to(dtype=mean.dtype, device=mean.device)
        noise = noise.reshape(num_samples, *(1,) * (mean.ndim - 1), self.latent_size)
        samples = mean.unsqueeze(0) + (
            scale.unsqueeze(0) @ noise.unsqueeze(-1)
        ).squeeze(-1)
        log_likelihood = self._observation_log_prob(
            values.unsqueeze(0),
            self.observation(samples),
            mask.unsqueeze(0),
        ).mean(dim=0)
        bound = log_likelihood - self._kl_gaussian(posterior, (mean_prior, cov_prior))
        return torch.where(mask.any(dim=-1), bound, torch.zeros_like(bound))

    def sample(
        self,
        size: int | tuple[int, ...] = (),  # *S
        *,
        query_times: Tensor,  # Float[..., $K], padded NaN, non-decreasing
        query_mask: Tensor,  # Bool[..., $K, F]  padded False
        context_times: Tensor,  # Float[..., $N], padded NaN, non-decreasing
        context_values: Tensor,  # Float[..., $N, D], padded NaN, sparse
        context_mask: Tensor,  # Bool[..., $N, D], padded False
        initial_state: tuple[Tensor, Tensor] | None = None,
        initial_time: Tensor | None = None,
        rng: Generator | None = None,
    ) -> Tensor:  # Float[*S, ..., $K, F]
        r"""Sample from the time-marginal predictive distribution."""
        sample_shape = (size,) if isinstance(size, int) else size
        mean, cov = self.predict(
            query_times=query_times,
            query_mask=query_mask,
            context_times=context_times,
            context_values=context_values,
            context_mask=context_mask,
            initial_state=initial_state,
            initial_time=initial_time,
        )
        latent_mask = torch.ones_like(mean, dtype=torch.bool)
        z = marginal_gaussian_sample(
            sample_shape, mean=mean, cov=cov, mask=latent_mask, rng=rng
        )
        observations = self.observation(z)
        noise = (
            torch.randn(
                observations.shape,
                dtype=observations.dtype,
                device=observations.device,
                generator=rng,
            )
            * self.observation_covariance.diagonal(dim1=-2, dim2=-1).sqrt()
        )
        mask = query_mask.expand(*sample_shape, *query_mask.shape)
        return (observations + noise).masked_fill(~mask, nan)

    def sample_and_log_prob(
        self,
        size: int | tuple[int, ...] = (),  # *S
        *,
        query_times: Tensor,  # Float[..., $K], padded NaN, non-decreasing
        query_mask: Tensor,  # Bool[..., $K, F]  padded False
        context_times: Tensor,  # Float[..., $N], padded NaN, non-decreasing
        context_values: Tensor,  # Float[..., $N, D], padded NaN, sparse
        context_mask: Tensor,  # Bool[..., $N, D], padded False
        initial_state: tuple[Tensor, Tensor] | None = None,
        initial_time: Tensor | None = None,
        rng: Generator | None = None,
    ) -> tuple[Tensor, Tensor]:  # Float[*S, ..., $K, F], Float[*S, ..., $K]
        r"""Sample from the model and compute sample log-probabilities."""
        samples = self.sample(
            size,
            query_times=query_times,
            query_mask=query_mask,
            context_times=context_times,
            context_values=context_values,
            context_mask=context_mask,
            initial_state=initial_state,
            initial_time=initial_time,
            rng=rng,
        )
        return samples, self.log_prob(
            samples,
            query_times=query_times,
            query_mask=query_mask,
            context_times=context_times,
            context_values=context_values,
            context_mask=context_mask,
            initial_state=initial_state,
            initial_time=initial_time,
        )


def make_koopman_filter(
    *,
    input_size: int,
    latent_size: int | None = None,
    decoder: str | nn.Module = "shiesh-lowrank-shiesh",
    low_rank: int | None = None,
    observation_noise: float = 0.5,
    n_iter: int = 1,
    batch_first: bool = True,
    state_propagator: Mapping[str, Any] | None = None,
) -> KoopmanFilter:
    r"""Instantiate the high-dimensional noisy flow-observation filter."""
    latent_size = 2 * input_size if latent_size is None else latent_size
    if isinstance(decoder, nn.Module):
        decoder_module = decoder
    else:
        match decoder:
            case "identity":
                decoder_module = nn.Identity()
            case "shiesh":
                decoder_module = TransformSequence([Shiesh(t=1.0, a=1.0)])
            case "lowrank":
                rank = input_size if low_rank is None else low_rank
                layer = LowRankTransform(latent_size, rank=rank)
                nn.init.constant_(layer.theta, 0.1)
                decoder_module = TransformSequence([layer])
            case "shiesh-lowrank-shiesh":
                rank = input_size if low_rank is None else low_rank
                layer = LowRankTransform(latent_size, rank=rank)
                nn.init.constant_(layer.theta, 0.1)
                decoder_module = TransformSequence(
                    [
                        Shiesh(t=1.0, a=1.0),
                        layer,
                        Shiesh(t=1.0, a=1.0),
                    ]
                )
            case _:
                raise ValueError(f"Unknown decoder: {decoder!r}.")

    propagator_kwargs = {
        "input_size": latent_size,
        "kernel_initialization": "zero",
        "kernel_parametrization": "identity",
        "use_rezero": False,
        "use_bias": False,
    }
    if state_propagator is not None:
        propagator_kwargs.update(state_propagator)

    return KoopmanFilter(
        input_size=input_size,
        latent_size=latent_size,
        decoder=decoder_module,
        state_propagator=LinearGaussianFlow(**propagator_kwargs),
        observation_noise=observation_noise,
        n_iter=n_iter,
        batch_first=batch_first,
    )
