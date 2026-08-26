r"""Reimplementation of the Continuous Recurrent Unit (CRU)."""

__all__ = [
    "CRU",
    "CRUConfig",
    "DecoderConfig",
    "EncoderConfig",
    "Decoder",
    "Encoder",
    # config dicts
    "EncoderConfigDict",
    "CRUConfigDict",
    "DecoderConfigDict",
    # functions
    "build_cru",
    "update_masked",
    "new_activation",
]

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Final, NotRequired, TypedDict, cast

import torch
from torch import Generator, Tensor, nan, nn

from .utils import EventBatch

_LOG2PI = math.log(2.0 * math.pi)


class _ELUP1(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return torch.where(x < 0.0, x.exp(), x + 1.0)


class _Exp(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.exp()


class _Square(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.square()


class _Abs(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.abs()


def new_activation(name: str) -> nn.Module:
    match name:
        case "relu":
            return nn.ReLU()
        case "tanh":
            return nn.Tanh()
        case "elup1":
            return _ELUP1()
        case "exp":
            return _Exp()
        case "square":
            return _Square()
        case "abs":
            return _Abs()
        case _:
            raise NotImplementedError


class Encoder(nn.Module):
    r"""Returns $yₜ, σₜ^{obs} = f_θ(xₜ)$."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: int,
        *,
        num_hidden_layers: int = 2,
        activation_function: str = "relu",
        variance_activation: str = "elup1",
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.feature_extractor = self._build_hidden_layers(
            num_hidden_layers,
            activation_function,
        )
        self.mean_model = nn.Linear(hidden_size, output_size)
        self.variance_model = nn.Sequential(
            nn.Linear(hidden_size, output_size),
            new_activation(variance_activation),
        )

    def _build_hidden_layers(
        self, num_layers: int, activation_name: str, /
    ) -> nn.Module:
        hidden_layers = []
        for _ in range(num_layers):
            hidden_layers.extend([
                nn.Linear(self.hidden_size, self.hidden_size),
                new_activation(activation_name),
                nn.LayerNorm(self.hidden_size),
            ])  # fmt: skip

        return nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            new_activation(activation_name),
            nn.LayerNorm(self.hidden_size),
            *hidden_layers,
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        h = self.feature_extractor(x)
        h = nn.functional.normalize(h, p=2, dim=-1, eps=1e-8)
        return self.mean_model(h), self.variance_model(h)


class Decoder(nn.Module):
    r"""Returns $oₜ, σₜ^{out} = g_ϕ(μₜ⁺, Σₜ⁺)$."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: int,
        *,
        num_hidden_mean_model_layers: int = 2,
        num_hidden_variance_model_layers: int = 0,
        activation_function: str = "relu",
        variance_activation: str = "elup1",
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.mean_model = self._build_mean_model(
            num_hidden_mean_model_layers,
            activation_function,
        )
        self.variance_model = self._build_variance_model(
            num_hidden_variance_model_layers,
            activation_function,
            variance_activation,
        )

    def _build_mean_model(self, num_layers: int, activation_name: str, /) -> nn.Module:
        hidden_layers = []
        for _ in range(num_layers):
            hidden_layers.extend([
                nn.Linear(self.hidden_size, self.hidden_size),
                new_activation(activation_name),
                nn.LayerNorm(self.hidden_size),
            ])  # fmt: skip

        return nn.Sequential(
            nn.Linear(2 * self.input_size, self.hidden_size),
            new_activation(activation_name),
            nn.LayerNorm(self.hidden_size),
            *hidden_layers,
            nn.Linear(self.hidden_size, self.output_size),
        )

    def _build_variance_model(
        self,
        num_layers: int,
        activation_name: str,
        variance_activation_name: str,
        /,
    ) -> nn.Module:
        hidden_layers = []
        for _ in range(num_layers):
            hidden_layers.extend([
                nn.Linear(self.hidden_size, self.hidden_size),
                new_activation(activation_name),
                nn.LayerNorm(self.hidden_size),
            ])  # fmt: skip

        return nn.Sequential(
            nn.Linear(3 * self.input_size, self.hidden_size),
            new_activation(activation_name),
            nn.LayerNorm(self.hidden_size),
            *hidden_layers,
            nn.Linear(self.hidden_size, self.output_size),
            new_activation(variance_activation_name),
        )

    def forward(
        self,
        mean: Tensor,  # Float[..., 2d]
        covariance: Tensor,  # Float[..., d, 3]
        /,
    ) -> tuple[Tensor, Tensor]:  # (..., d),
        cov = torch.cat(covariance.unbind(dim=-1), dim=-1)
        return self.mean_model(mean), self.variance_model(cov)


@dataclass(frozen=True, slots=True)
class EncoderConfig:
    r"""Configuration for the CRU observation encoder."""

    input_size: int
    output_size: int
    hidden_size: int
    num_hidden_layers: int = 2
    activation_function: str = "relu"
    variance_activation: str = "elup1"


@dataclass(frozen=True, slots=True)
class DecoderConfig:
    r"""Configuration for the CRU output decoder."""

    input_size: int
    output_size: int
    hidden_size: int
    num_hidden_mean_model_layers: int = 2
    num_hidden_variance_model_layers: int = 0
    activation_function: str = "relu"
    variance_activation: str = "elup1"


@dataclass(frozen=True, slots=True, kw_only=True)
class CRUConfig:
    r"""Hierarchical configuration for constructing a CRU."""

    input_size: int
    latent_size: int
    encoder: EncoderConfig
    decoder: DecoderConfig
    output_size: int
    num_basis: int = 15
    bandwidth: int = 3
    variance_activation: str = "elup1"
    initial_variance: float = 10.0
    validate_args: bool = False
    batch_first: bool = True


class CRUConfigDict(TypedDict):
    r"""Mapping form of ``CRUConfig``."""

    input_size: int
    latent_size: int
    encoder: EncoderConfigLike
    decoder: DecoderConfigLike
    output_size: int
    num_basis: NotRequired[int]
    bandwidth: NotRequired[int]
    variance_activation: NotRequired[str]
    initial_variance: NotRequired[float]
    validate_args: NotRequired[bool]
    batch_first: NotRequired[bool]


class EncoderConfigDict(TypedDict):
    r"""Mapping form of ``EncoderConfig``."""

    input_size: int
    output_size: int
    hidden_size: int
    num_hidden_layers: NotRequired[int]
    activation_function: NotRequired[str]
    variance_activation: NotRequired[str]


class DecoderConfigDict(TypedDict):
    r"""Mapping form of ``DecoderConfig``."""

    input_size: int
    output_size: int
    hidden_size: int
    num_hidden_mean_model_layers: NotRequired[int]
    num_hidden_variance_model_layers: NotRequired[int]
    activation_function: NotRequired[str]
    variance_activation: NotRequired[str]


type EncoderConfigLike = EncoderConfig | EncoderConfigDict
type DecoderConfigLike = DecoderConfig | DecoderConfigDict


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


def _marginal_var_gaussian_log_prob(
    values: Tensor,
    *,
    mean: Tensor,
    var: Tensor,
    mask: Tensor,
) -> Tensor:
    r"""Compute log-likelihoods of masked diagonal Gaussian marginals (variance parameterization)."""
    assert values.shape == mean.shape == var.shape
    assert mask.shape == values.shape
    assert mask.dtype == torch.bool

    centered = torch.where(mask, values - mean, torch.zeros_like(values))
    safe_var = torch.where(mask, var, torch.ones_like(var))
    log_prob = -0.5 * (centered.square() / safe_var + torch.log(safe_var) + _LOG2PI)
    return torch.where(mask, log_prob, torch.zeros_like(log_prob)).sum(dim=-1)


def _marginal_var_gaussian_sample(
    size: int | tuple[int, ...] = (),
    *,
    mean: Tensor,
    var: Tensor,
    mask: Tensor,
    rng: Generator | None = None,
) -> Tensor:
    r"""Sample from masked diagonal Gaussian marginals (variance parameterization)."""
    assert mean.shape == var.shape
    assert mask.shape == mean.shape
    assert mask.dtype == torch.bool

    sample_shape = (size,) if isinstance(size, int) else size
    safe_mean = torch.where(mask, mean, torch.zeros_like(mean))
    safe_std = torch.where(mask, var.sqrt(), torch.zeros_like(var))
    noise = torch.randn(
        (*sample_shape, *mean.shape),
        dtype=mean.dtype,
        device=mean.device,
        generator=rng,
    )
    samples = (
        safe_mean.expand(*sample_shape, *mean.shape)
        + safe_std.expand(*sample_shape, *mean.shape) * noise
    )
    return samples.masked_fill(~mask.expand(*sample_shape, *mask.shape), nan)


def _marginal_var_gaussian_sample_and_log_prob(
    size: int | tuple[int, ...] = (),
    *,
    mean: Tensor,
    var: Tensor,
    mask: Tensor,
    rng: Generator | None = None,
) -> tuple[Tensor, Tensor]:
    r"""Sample from masked diagonal Gaussian marginals and score the samples."""
    samples = _marginal_var_gaussian_sample(
        size, mean=mean, var=var, mask=mask, rng=rng
    )
    return samples, _marginal_var_gaussian_log_prob(
        samples,
        mean=mean.expand(*samples.shape),
        var=var.expand(*samples.shape),
        mask=mask.expand(*samples.shape),
    )


class CRU(nn.Module):
    r"""Continuous Recurrent Unit for probabilistic forecasting.

    The basic setup is a latent linear SDE with Gaussian observations.

    .. math:: dz = Azdt + Gdβ   \qquad   yₜ∼𝓝(Hzₜ, σₜ²𝕀)

    This is combined with an encoder decoder setup:

    .. code-block:: text

        μ_{t₀}⁺, Σ_{t₀}⁺ ← 0, σ₀⋅𝕀
        for t ∈ {t₁, …, tₙ}:
            yₜ, σₜ^{obs} ← f_θ(xₜ)
            μₜ⁻, Σₜ⁻ ← predict(μₛ⁺, Σₛ⁺, t - s)
            μₜ⁺, Σₜ⁺ ← update(μₜ⁻, Σₜ⁻, yₜ, σₜ^{obs})
            oₜ, σₜ^{out} ← g_ϕ(μₜ⁺, Σₜ⁺)
            s ← t

    Where:
        - f_θ, the encoder, is a neural network
        - g_ϕ, the decoder, is a neural network
        - predict is performed via matrix exponential solution to the linear SDE
        - update is the regular kalman update

    Additionally, CRU makes the following simplifying assumptions:
        - latent space is twice the dimension of the latent observation space
        - trivial observation model: H = [𝕀_d, 𝟎_d]
        - block-wise diagonal covariance matrix: Σₜ = [[Σₜᵘ, Σₜˢ], [Σₜˢ, Σₜˡ]],
          where each block is a diagonal matrix Σₜᵘ=σₜᵘ𝕀, Σₜˢ=σₜˢ𝕀, Σₜˡ=σₜˡ𝕀
        - Hence, the kalman gain takes the form Kₜ=[Kₜᵘ; Kₜˡ], where
          - Kₜᵘ = diag(σₜᵘ / (σₜᵘ + σₜ^{obs}))
          - Kₜˡ = diag(σₜˢ / (σₜᵘ + σₜ^{obs}))

    Note: Differences to the reference implementation.
        - We do not make the initial covariance trainable, as this is not
          mentioned in the paper.
        - ``latent_size`` corresponds to ``latent_observation_size``.
        - ``bandwidth`` must be in the range ``[0, latent_size - 1]``.
        - Batch-first layout is used by default.

    Note:
        CRU does not support input missingness. In their experiments, for instance
        on PhysioNet, missing features are simply imputed with zeros. The mask
        is not forwarded to the model, so it cannot distinguish between
        observed zeros and missing values.
    """

    # Constants
    input_size: Final[int]
    r"""CONST: Dimensionality of observed context values."""
    output_size: Final[int]
    r"""CONST: Dimensionality of forecast targets."""
    latent_size: Final[int]
    r"""CONST: Dimensionality of encoded latent observations."""
    validate_args: Final[bool]
    r"""CONST: Whether forward inputs should be validated before computation."""
    batch_first: Final[bool]
    r"""CONST: If True, time axis is the second-to-last dimension of inputs."""

    # Submodules
    encoder: nn.Module
    r"""MODULE: Maps observations to latent-observation Gaussian parameters."""
    decoder: nn.Module
    r"""MODULE: Maps latent Gaussian states to predictive distributions."""
    transition_coefficient_model: nn.Module
    r"""MODULE: Maps μₜ to the coefficient vector α(μₜ)."""

    # Buffers
    block_banded_mask: Tensor
    r"""BUFFER: Block banded mask for transition matrix."""
    initial_mean: Tensor
    r"""BUFFER: Initial mean."""
    initial_covariance: Tensor
    r"""BUFFER: Initial covariance."""
    prior_means: Tensor
    r"""BUFFER: Prior mean trajectory from the last forward pass."""
    prior_variances: Tensor
    r"""BUFFER: Prior covariance trajectory from the last forward pass."""
    posterior_means: Tensor
    r"""BUFFER: Posterior mean trajectory from the last forward pass."""
    posterior_variances: Tensor
    r"""BUFFER: Posterior covariance trajectory from the last forward pass."""
    pred_means: Tensor
    r"""BUFFER: Predicted means from the last forward pass."""
    pred_variances: Tensor
    r"""BUFFER: Predicted variances from the last forward pass."""

    @property
    def config(self) -> dict[str, object]:
        r"""Return constructor-relevant configuration."""
        return {
            "input_size": self.input_size,
            "output_size": self.output_size,
            "latent_size": self.latent_size,
            "validate_args": self.validate_args,
            "initial_variance": self.initial_variance,
        }

    def __init__(
        self,
        input_size: int,
        latent_size: int,  # corresponds to latent_observation_size in CRU reference impl.
        *,
        output_size: int | None = None,
        encoder: nn.Module,
        decoder: nn.Module,
        num_basis: int = 15,  # number of basis matrices for the transition model
        bandwidth: int = 3,  # bandwidth of the blocks of the transition matrix
        initial_variance: float = 10.0,
        variance_activation: str = "elup1",
        validate_args: bool = False,
        batch_first: bool = True,
    ) -> None:
        super().__init__()
        if initial_variance <= 0:
            raise ValueError("initial_variance must be positive.")

        self.input_size = input_size
        if output_size is None:
            output_size = getattr(decoder, "output_size", None)
        if not isinstance(output_size, int):
            raise TypeError(
                "output_size must be provided if decoder has no output_size."
            )
        self.output_size = output_size
        self.latent_size = latent_size
        self.num_basis = num_basis
        self.validate_args = validate_args
        self.initial_variance = initial_variance
        self.batch_first = batch_first

        self.variance_activation = new_activation(variance_activation)

        self.encoder = encoder
        self.decoder = decoder

        # The transition matrix A is parametrized as a linear combination ∑ₖαₖ(μₜ)Aₖ
        # where Aₖ = [[B₁₁, B₁₂], [B₂₁, B₂₂]] and each block is a d×d banded matrix of bandwidth b.
        # The number of parameters of a banded d×d matrix with bandwidth b is:
        #   d + 2*(T_{d-1} - T_{d-b-1}), where Tₙ is the n-th triangle number
        # note: reference allows bandwidth=latent_size, which is a bug.
        assert bandwidth >= 0, "bandwidth must be non-negative"
        assert bandwidth < latent_size, "bandwidth must be smaller than latent_size."
        T = lambda n: n * (n + 1) // 2
        num_params: int = latent_size + 2 * (
            T(latent_size - 1) - T(latent_size - bandwidth - 1)
        )
        self.transition_matrix_parameters = nn.Parameter(
            torch.zeros(num_basis, 2, 2, num_params)
        )

        # create a mask for the transition matrix model
        band_mask = (
            torch.ones((latent_size, latent_size), dtype=torch.bool)
            .triu(-bandwidth)
            .tril(bandwidth)
        )
        self.register_buffer(
            "block_banded_mask",
            band_mask.expand(2, 2, latent_size, latent_size),
        )

        # "For all experiments, we used a transition net with one linear layer and
        # softmax output."
        self.transition_coefficient_model = nn.Sequential(
            nn.Linear(2 * self.latent_size, self.num_basis),
            nn.Softmax(dim=-1),
        )

        # NOTE: The reference implementation makes the initial variance trainable.
        # this however is not mentioned in the paper. We use fixed buffers instead.
        self.register_buffer("initial_mean", torch.zeros(2 * self.latent_size))
        self.register_buffer(
            "initial_covariance", initial_variance * torch.eye(2 * self.latent_size)
        )
        self.register_buffer("q", torch.ones(2 * self.latent_size))
        self.register_buffer("prior_means", torch.empty(0), persistent=False)
        self.register_buffer("prior_variances", torch.empty(0), persistent=False)
        self.register_buffer("posterior_means", torch.empty(0), persistent=False)
        self.register_buffer("posterior_variances", torch.empty(0), persistent=False)
        self.register_buffer("pred_means", torch.empty(0), persistent=False)
        self.register_buffer("pred_variances", torch.empty(0), persistent=False)

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
        combined = EventBatch.from_request(
            context_times=context_times,
            context_values=context_values,
            context_mask=context_mask,
            query_times=query_times,
            query_mask=query_mask,
            batch_first=self.batch_first,
        )
        post_means, post_logvars = self.forward(
            timestamps=combined.timestamps,  # Float[..., $T], padded NaN, non-decreasing
            context_values=combined.context_values,  # Float[..., $T, D], padded NaN, sparse
            context_mask=combined.context_mask,  # Bool[..., $T, D], padded False
            query_mask=combined.query_mask,  # Bool[..., $T, F], padded False
            initial_state=initial_state,
            initial_time=initial_time,
        )

        self.pred_means = post_means[..., *combined.query_indices, :]
        self.pred_variances = post_logvars[..., *combined.query_indices, :]
        return self.pred_means, self.pred_variances

    def forward(
        self,
        *,
        timestamps: Tensor,  # Float[..., $T], padded NaN
        query_mask: Tensor,  # Bool[..., $T, D], padded False
        context_values: Tensor,  # Float[..., $T, D], padded NaN, sparse
        context_mask: Tensor,  # Bool[..., $T, D], padded False
        initial_state: tuple[Tensor, Tensor] | None = None,  # (..., 2d), (..., d, 3)
        initial_time: Tensor | None = None,  # t₀, () or (...)
    ) -> tuple[Tensor, Tensor]:  # Float[..., $K, F], Float[..., $K, F]
        r"""Filter and forecast over combined context/query time points.

        Context and query masks explicitly select valid feature-level entries.
        Context values outside ``context_mask`` are ignored. CRU does not
        support feature-level missingness: each time step must be either fully
        observed or fully missing (all context features present or none).

        Args:
            timestamps: Float tensor of shape ``(..., $T)`` containing the combined context and
            query_mask: Boolean mask selecting requested forecast entries.
            context_values: Sparse observations at context/query time points.
            context_mask: Boolean mask selecting observed context entries.
            initial_state: Optional initial latent state ``(mean, cov)`` where
                ``mean`` has shape ``(..., 2d)`` and ``cov`` has shape
                ``(..., d, 3)``.
            initial_time: Optional initial time ``t₀``.

        Returns:
            pred_means: Posterior predicted means at all time points, shape
                ``(..., $N + $K, F)``. NaN at non-query positions.
            pred_vars: Posterior predicted variances, same shape. NaN at
                non-query positions.
        """
        d = self.latent_size

        # Step-level masks derived from feature-level masks.
        has_context = context_mask.any(dim=-1)  # (..., $N+$K)
        has_query = query_mask.any(dim=-1)  # (..., $N+$K)
        valid_steps = timestamps.isfinite() & (has_context | has_query)
        has_query = has_query.unsqueeze(-1)  # (..., $N+$K, 1)

        # CRU does not support feature-level missingness.
        assert torch.equal(has_context, context_mask.all(dim=-1)), (
            "CRU requires all context features present or none per step."
        )
        # Observations must align with finite time stamps.
        assert (~has_context | timestamps.isfinite()).all(), (
            "context_mask is True at a time step with non-finite time."
        )

        *batch_shape, _ = timestamps.shape

        y_means = context_values.new_full((*context_values.shape[:-1], d), nan)
        y_variances = context_values.new_full((*context_values.shape[:-1], d), nan)
        y_means, y_variances = update_masked(  # (..., $N+$K, d) each
            (y_means, y_variances),
            self.encoder,
            args=(context_values,),
            batch_mask=has_context,
        )

        if self.batch_first:
            timestamps = timestamps.movedim(-1, 0)
            y_means = y_means.movedim(-2, 0)
            y_variances = y_variances.movedim(-2, 0)
            has_context = has_context.movedim(-1, 0)
            valid_steps = valid_steps.movedim(-1, 0)

        # Initialize state (mean: (..., 2d), cov: (..., d, 3)).
        t = timestamps[0] if initial_time is None else initial_time
        if initial_state is None:
            cov_u = self.initial_covariance[:d, :d].diagonal()
            cov_l = self.initial_covariance[d:, d:].diagonal()
            cov_s = self.initial_covariance[:d, d:].diagonal()
            post_mean = self.initial_mean.expand(*batch_shape, 2 * d)
            post_cov = torch.stack(
                [cov_u, cov_l, cov_s],
                dim=-1,
            ).expand(*batch_shape, d, 3)
        else:
            post_mean, post_cov = initial_state
            post_mean = post_mean.expand(*batch_shape, 2 * d)
            post_cov = post_cov.expand(*batch_shape, d, 3)

        prior_means_list: list[Tensor] = []
        prior_vars_list: list[Tensor] = []
        post_means_list: list[Tensor] = []
        post_vars_list: list[Tensor] = []
        pred_means_list: list[Tensor] = []
        pred_vars_list: list[Tensor] = []

        for t_obs, y, y_var, ctx_mask, active in zip(
            timestamps,
            y_means,
            y_variances,
            has_context,
            valid_steps,
            strict=True,
        ):
            delta = torch.where(active, t_obs - t, torch.zeros_like(t_obs))
            t = torch.where(active, t_obs, t)

            # Propagate only for active batch elements; restore old state for inactive.
            prior_mean, prior_cov = update_masked(
                (post_mean, post_cov),
                self.propagate_state,
                args=(delta, post_mean, post_cov),
                batch_mask=active,
            )

            # Update only for batch elements that have context at this step.
            post_mean, post_cov = update_masked(
                (prior_mean, prior_cov),
                self.update_state,
                args=(y, y_var, ctx_mask, prior_mean, prior_cov),
                batch_mask=active & ctx_mask,
            )

            pred_mean, pred_var = self.decoder(post_mean, post_cov)

            prior_means_list.append(prior_mean)
            prior_vars_list.append(prior_cov)
            post_means_list.append(post_mean)
            post_vars_list.append(post_cov)
            pred_means_list.append(pred_mean)
            pred_vars_list.append(pred_var)

        stack_dim_mean = -2 if self.batch_first else 0
        stack_dim_cov = -3 if self.batch_first else 0

        self.prior_means = torch.stack(prior_means_list, dim=stack_dim_mean)
        self.prior_variances = torch.stack(prior_vars_list, dim=stack_dim_cov)
        self.posterior_means = torch.stack(post_means_list, dim=stack_dim_mean)
        self.posterior_variances = torch.stack(post_vars_list, dim=stack_dim_cov)

        pred_means = torch.stack(pred_means_list, dim=stack_dim_mean)
        pred_vars = torch.stack(pred_vars_list, dim=stack_dim_mean)
        pred_means = pred_means.masked_fill(~has_query, nan)
        pred_vars = pred_vars.masked_fill(~has_query, nan)

        return pred_means, pred_vars

    def log_prob(
        self,
        values: Tensor,  # Float[..., $K, F]
        /,
        *,
        query_times: Tensor,  # Float[..., K], padded NaN, non-decreasing
        query_mask: Tensor,  # Bool[..., K, F]  padded False
        context_times: Tensor,  # Float[..., N], padded NaN, non-decreasing
        context_values: Tensor,  # Float[..., N, D], padded NaN, sparse
        context_mask: Tensor,  # Bool[..., N, D], padded False
        initial_state: tuple[Tensor, Tensor] | None = None,
        initial_time: Tensor | None = None,
    ) -> Tensor:  # Float[..., $K]
        r"""Compute the time-marginal log-likelihood of the model.

        .. math:: pₖ = p_{Y_{qₖ}}(yₖ | (t₁, y₁), ..., (tₙ, yₙ))
        """
        mean, var = self.predict(
            query_times=query_times,
            query_mask=query_mask,
            context_times=context_times,
            context_values=context_values,
            context_mask=context_mask,
            initial_state=initial_state,
            initial_time=initial_time,
        )
        return _marginal_var_gaussian_log_prob(
            values,
            mean=mean.expand(*values.shape),
            var=var.expand(*values.shape),
            mask=query_mask.expand(*values.shape),
        )

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
        r"""Sample from the time-marginal distribution.

        .. math:: pₖ = p_{Y_{qₖ}}(yₖ | (t₁, y₁), ..., (tₙ, yₙ))
        """
        sample_shape = (size,) if isinstance(size, int) else size
        mean, var = self.predict(
            query_times=query_times,
            query_mask=query_mask,
            context_times=context_times,
            context_values=context_values,
            context_mask=context_mask,
            initial_state=initial_state,
            initial_time=initial_time,
        )
        return _marginal_var_gaussian_sample(
            sample_shape, mean=mean, var=var, mask=query_mask, rng=rng
        )

    def sample_and_log_prob(
        self,
        size: int | tuple[int, ...] = (),  # *S
        *,
        query_times: Tensor,  # Float[..., K], padded NaN, non-decreasing
        query_mask: Tensor,  # Bool[..., K, F]  padded False
        context_times: Tensor,  # Float[..., N], padded NaN, non-decreasing
        context_values: Tensor,  # Float[..., N, D], padded NaN, sparse
        context_mask: Tensor,  # Bool[..., N, D], padded False
        initial_state: tuple[Tensor, Tensor] | None = None,
        initial_time: Tensor | None = None,
        rng: Generator | None = None,
    ) -> tuple[Tensor, Tensor]:  # Float[*S, ..., $K, F], Float[*S, ..., $K]
        r"""Sample from the time-marginal distribution and yield log-probabilities.

        .. math:: pₖ = p_{Y_{qₖ}}(yₖ | (t₁, y₁), ..., (tₙ, yₙ))
        """
        sample_shape = (size,) if isinstance(size, int) else size
        mean, var = self.predict(
            query_times=query_times,
            query_mask=query_mask,
            context_times=context_times,
            context_values=context_values,
            context_mask=context_mask,
            initial_state=initial_state,
            initial_time=initial_time,
        )
        return _marginal_var_gaussian_sample_and_log_prob(
            sample_shape, mean=mean, var=var, mask=query_mask, rng=rng
        )

    def transition_matrix_model(self, mean: Tensor) -> Tensor:
        """Locally linear transition model.

        Aₜ = ∑ₖαₖ(t)Aₖ, where αₖ(t) = w_ψ(μₜ⁺)
        Here Aₖ = [Aₖ¹¹, Aₖ¹²; Aₖ²¹, Aₖ²²],
        where each block is a band-matrix of bandwidth b.
        """
        *batch_shape, _ = mean.shape
        d = self.latent_size
        alpha = self.transition_coefficient_model(mean)  # (..., k)
        weighted = torch.einsum(
            "...k, kijp -> ...ijp", alpha, self.transition_matrix_parameters
        )  # (..., 2, 2, p(b))

        mask = self.block_banded_mask.expand(*batch_shape, 2, 2, d, d)
        blocks = weighted.new_zeros(*batch_shape, 2, 2, d, d)
        blocks = blocks.masked_scatter(mask, weighted)
        return torch.cat([
            torch.cat([blocks[..., 0, 0, :, :], blocks[..., 0, 1, :, :]], dim=-1),
            torch.cat([blocks[..., 1, 0, :, :], blocks[..., 1, 1, :, :]], dim=-1),
        ], dim=-2)  # fmt: skip

    def propagate_state(
        self,
        delta_time: Tensor,  # Float[...]
        posterior_mean: Tensor,  # Float[..., 2d]
        posterior_variance: Tensor,  # Float[..., d, 3]
        /,
    ) -> tuple[Tensor, Tensor]:
        r"""Propagate a latent posterior through the continuous transition model."""
        # reconstruct Σ from σᵤ, σᵥ, σₛ
        var_u, var_l, var_s = posterior_variance.unbind(dim=-1)
        cov_u = torch.diag_embed(var_u)
        cov_l = torch.diag_embed(var_l)
        cov_s = torch.diag_embed(var_s)
        cov = torch.cat([
            torch.cat([cov_u, cov_s], dim=-1),
            torch.cat([cov_s, cov_l], dim=-1),
        ], dim=-2)  # fmt: skip

        A = self.transition_matrix_model(posterior_mean)
        Q = torch.diag_embed(self.variance_activation(self.q)).expand_as(A)

        # compute van Loan matrix exponential
        n = posterior_mean.shape[-1]
        zero = torch.zeros_like(A)
        M = torch.cat([
            torch.cat([A, Q], dim=-1),
            torch.cat([zero, -A.mT], dim=-1),
        ], dim=-2)  # fmt: skip
        # eᴹᵗ = [[F, C], [0, -Fᵀ]]
        exp_Mt = torch.linalg.matrix_exp(M * delta_time[..., None, None])
        exp_At = exp_Mt[..., :n, :n]  # upper left block
        C = exp_Mt[..., :n, n:]  # upper right block

        # μₜ = eᴬᵗμ₀
        prior_mean = torch.einsum("...mn, ...n -> ...m", exp_At, posterior_mean)
        # Σₜ = eᴬᵗΣ₀eᴬᵀᵗ + Ceᴬᵀᵗ
        prior_cov = (exp_At @ cov + C) @ exp_At.mT  # [Σᵤ, Σₛ; Σₛ, Σˡ]

        # Note: If X is block-wise diagonal, then exp(X) is also block-wise diagonal.
        d = var_u.shape[-1]
        prior_var_u = prior_cov[..., :d, :d].diagonal(dim1=-2, dim2=-1)
        prior_var_s = prior_cov[..., :d, d:].diagonal(dim1=-2, dim2=-1)
        prior_var_l = prior_cov[..., d:, d:].diagonal(dim1=-2, dim2=-1)

        return prior_mean, torch.stack([prior_var_u, prior_var_l, prior_var_s], dim=-1)

    def update_state(
        self,
        observation_mean: Tensor,  # Float[..., d]
        observation_variance: Tensor,  # Float[..., d]
        observation_mask: Tensor,  # Float[...,]
        prior_mean: Tensor,  # Float[..., 2d]
        prior_variance: Tensor,  # Float[..., d, 3]
    ) -> tuple[Tensor, Tensor]:  # Float[..., 2d], Float[..., d, 3]
        r"""Apply the CRU/Kalman measurement update for one time step."""
        # assumptions:
        # H = [𝕀_d, 0_d]
        # Σₜ = [[Σₜᵘ, Σₜˢ], [Σₜˢ, Σₜˡ]], where
        #   Σₜᵘ=diag(σₜᵘ), Σₜˢ=diag(σₜˢ), Σₜˡ=diag(σₜˡ)
        # Note: the paper explicitly uses σ for the variance rather than stdv.
        # kalman gain: Kₜ = Σₜ⁻Hᵀ(HΣₜ⁻Hᵀ + Rₜ)⁻¹
        # compute the simplified kalman gain Kₜ=[Kₜᵘ; Kₜˡ], where
        #   - Kₜᵘ = diag(σₜᵘ / (σₜᵘ + σₜ^{obs}))
        #   - Kₜˡ = diag(σₜˢ / (σₜᵘ + σₜ^{obs}))
        d = observation_mean.shape[-1]
        mask = observation_mask.unsqueeze(-1)  # (..., 1)
        var_u, var_l, var_s = prior_variance.unbind(dim=-1)
        denominator = var_u + observation_variance
        gain_u = var_u / denominator
        gain_l = var_s / denominator
        # μₜ⁺ = μₜ⁻ + Kₜ(yₜ - Hμₜ⁻), using H = [𝕀_d, 𝟎_d]
        residual = observation_mean - prior_mean[..., :d]
        post_mean = prior_mean + torch.where(
            mask,
            torch.cat([gain_u * residual, gain_l * residual], dim=-1),
            0.0,
        )
        # Σₜ⁺ = (I - KₜH)Σₜ⁻
        post_cov = torch.stack([
            var_u - torch.where(mask, gain_u * var_u, 0.0),  # (1-Kᵘ)σᵘ
            var_l - torch.where(mask, gain_l * var_s, 0.0),  # σˡ - Kˡσˢ
            var_s - torch.where(mask, gain_u * var_s, 0.0),  # (1-Kᵘ)σˢ
        ], dim=-1)  # fmt: skip

        # validation that resulting covariance is positive definite
        # if __debug__:
        #     assert (var_u > 0).all()
        #     assert (var_l > 0).all()
        #     assert (var_s >= 0).all()
        #     assert (var_u * var_l > var_s**2).all()

        return post_mean, post_cov


def build_cru(config: CRUConfig | CRUConfigDict | Mapping[str, Any], /) -> CRU:
    r"""Construct a CRU from a hierarchical configuration object."""
    if not isinstance(config, CRUConfig):
        encoder = config["encoder"]
        decoder = config["decoder"]
        config = CRUConfig(
            input_size=config["input_size"],
            latent_size=config["latent_size"],
            encoder=(
                encoder
                if isinstance(encoder, EncoderConfig)
                else EncoderConfig(**cast("EncoderConfigDict", encoder))
            ),
            decoder=(
                decoder
                if isinstance(decoder, DecoderConfig)
                else DecoderConfig(**cast("DecoderConfigDict", decoder))
            ),
            output_size=config["output_size"],
            num_basis=config.get("num_basis", 15),
            bandwidth=config.get("bandwidth", 3),
            variance_activation=config.get("variance_activation", "elup1"),
            initial_variance=config.get("initial_variance", 10.0),
            validate_args=config.get("validate_args", False),
            batch_first=config.get("batch_first", True),
        )

    encoder = Encoder(
        config.encoder.input_size,
        config.encoder.output_size,
        config.encoder.hidden_size,
        num_hidden_layers=config.encoder.num_hidden_layers,
        activation_function=config.encoder.activation_function,
        variance_activation=config.encoder.variance_activation,
    )
    decoder = Decoder(
        config.decoder.input_size,
        config.decoder.output_size,
        config.decoder.hidden_size,
        num_hidden_mean_model_layers=config.decoder.num_hidden_mean_model_layers,
        num_hidden_variance_model_layers=(
            config.decoder.num_hidden_variance_model_layers
        ),
        activation_function=config.decoder.activation_function,
        variance_activation=config.decoder.variance_activation,
    )

    return CRU(
        config.input_size,
        config.latent_size,
        output_size=config.output_size,
        encoder=encoder,
        decoder=decoder,
        num_basis=config.num_basis,
        bandwidth=config.bandwidth,
        initial_variance=config.initial_variance,
        variance_activation=config.variance_activation,
        validate_args=config.validate_args,
        batch_first=config.batch_first,
    )
