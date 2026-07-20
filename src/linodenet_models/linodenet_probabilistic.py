r"""Probabilistic Linodenet model."""

__all__ = [
    "update_masked",
    "linear_gaussian_flow",
    "LinodenetProbabilistic",
    "LinearGaussianFlow",
    "make_linodenet_prob",
    "make_linodenet",
]

import warnings
from collections.abc import Callable, Mapping
from typing import Any, Final, Optional

import torch
from torch import Generator, Tensor, nan, nn

from linodenet.state_update import GaussianForwardUpdater, GaussianReverseUpdater

from .decoders import LowRankTransform, TransformSequence
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
    batch_mask: Tensor,  # (...)
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
    delta_t: Tensor,  # (..., $n)
    z0: tuple[Tensor, Tensor],  # (..., d), (..., d, d)
    A: Tensor,  # (d, d)
    Q: Tensor,  # (d, d)
    b: Optional[Tensor] = None,  # (d,) | None
    /,
) -> tuple[Tensor, Tensor]:  # (...., $n, d), (..., $n, d, d)
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
    Mdt = torch.einsum("..., kl -> ...kl", delta_t, M)
    P = torch.linalg.matrix_exp(Mdt)
    F = P[..., :n, :n]  # top left block
    C = P[..., :n, n : 2 * n]  # top center block
    r = P[..., :n, -1] if b is not None else 0.0  # top right block
    mu_t = torch.einsum("...nkl, ...l -> ...nk", F, mu_0) + r  # eᴬᵗμ₀ + φ₁(At)bt
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

    decoder: TransformSequence  # normalizing flow

    input_size: Final[int]
    latent_size: Final[int]

    initial_mean: Tensor
    initial_cov: Tensor
    initial_cov_parametrization: nn.Module

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
        r"""Forward pass of the probabilistic Linodenet model.

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

        posterior_state = (
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
                posterior_state,
                lambda dt, mean, cov: self.state_propagator(dt, (mean, cov)),
                args=(delta_t, *posterior_state),
                batch_mask=active,
            )

            posterior_state = update_masked(
                prior_state,
                lambda y, mean, cov: self.state_updater(y, (mean, cov)),
                args=(x_obs, *prior_state),
                batch_mask=obs_mask.any(dim=-1),
            )

            prior_means.append(prior_state[0])
            prior_covs.append(prior_state[1])
            posterior_means.append(posterior_state[0])
            posterior_covs.append(posterior_state[1])

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
        query_times: Tensor,  # Float[(..., K)], padded NaN, strictly increasing
        query_mask: Tensor,  # Bool[(..., K, F)]  padded False
        context_times: Tensor,  # Float[(..., N)], padded NaN, non-decreasing
        context_mask: Tensor,  # Bool[(..., N, D)], padded False
        context_values: Tensor,  # Float[(..., N, D)], padded NaN, sparse
        initial_state: tuple[Tensor, Tensor] | None = None,  # (..., 2d), (..., d, 3)
        initial_time: Tensor | None = None,  # t₀, () or (...)
    ) -> tuple[Tensor, Tensor]:  # (..., $K, D), (..., $K, D, D)
        combined = EventBatch.from_request(
            context_times=context_times,
            context_values=context_values,
            context_mask=context_mask,
            query_times=query_times,
            query_mask=query_mask,
            batch_first=self.batch_first,
        )
        post_means, post_covs = self.forward(
            timestamps=combined.timestamps,  # (..., $T), padded NaN, non-decreasing
            context_values=combined.context_values,  # (..., $T, D), padded NaN, sparse
            context_mask=combined.context_mask,  # Bool[(..., $T, D)], padded False
            query_mask=combined.query_mask,  # Bool[(..., $T, F)], padded False
            initial_state=initial_state,
            initial_time=initial_time,
        )
        post_means = post_means[combined.query_indices]
        post_covs = post_covs[combined.query_indices]
        return post_means, post_covs

    def log_prob(
        self,
        values: Tensor,  # (..., $K, F)
        /,
        *,
        query_times: Tensor,  # Float[(..., K)], padded NaN, strictly increasing
        query_mask: Tensor,  # Bool[(..., K, F)]  padded False
        context_times: Tensor,  # Float[(..., N)], padded NaN, non-decreasing
        context_values: Tensor,  # Float[(..., N, D)], padded NaN, sparse
        context_mask: Tensor,  # Bool[(..., N, D)], padded False
        initial_state: tuple[Tensor, Tensor] | None = None,
        initial_time: Tensor | None = None,
    ) -> Tensor:  # (..., $K)
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
        size: int | tuple[int, ...] = (),
        *,
        query_times: Tensor,  # Float[(..., K)], padded NaN, strictly increasing
        query_mask: Tensor,  # Bool[(..., K, F)]  padded False
        context_times: Tensor,  # Float[(..., N)], padded NaN, non-decreasing
        context_values: Tensor,  # Float[(..., N, D)], padded NaN, sparse
        context_mask: Tensor,  # Bool[(..., N, D)], padded False
        initial_state: tuple[Tensor, Tensor] | None = None,
        initial_time: Tensor | None = None,
        rng: Generator | None = None,
    ) -> Tensor:
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
        size: int | tuple[int, ...] = (),
        *,
        query_times: Tensor,  # Float[(..., K)], padded NaN, strictly increasing
        query_mask: Tensor,  # Bool[(..., K, F)]  padded False
        context_times: Tensor,  # Float[(..., N)], padded NaN, non-decreasing
        context_values: Tensor,  # Float[(..., N, D)], padded NaN, sparse
        context_mask: Tensor,  # Bool[(..., N, D)], padded False
        initial_state: tuple[Tensor, Tensor] | None = None,
        initial_time: Tensor | None = None,
        rng: Generator | None = None,
    ) -> tuple[Tensor, Tensor]:
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


class MarODE(nn.Module):
    r"""Working title.

    Marginalizable variant similar to moses; supports missing values.

    1. Keep track of multiple latent gaussian distributions $(μ⁽ʰ⁾, Σ⁽ʰ⁾)$.
    2. Decoder is a separable normalizing flow.
    3. State propagator is a linear gaussian flow.
    4. State update is KL-regularized proximal step:
       $θ₊ = \argmin_θ -\log q^\mar(y^\obs∣θ) + λ⋅\KL(p_θ, p_{θ₋})$
    5. mixture weights are learned.
    """

    def __init__(self) -> None:
        super().__init__()
        raise NotImplementedError


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
