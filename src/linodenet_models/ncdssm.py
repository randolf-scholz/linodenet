r"""Neural Continuous-Discrete State Space Models (NCDSSM).

The continuous latent state is a linear Gaussian SDE. Neural networks connect
sparse observations to its auxiliary observation space, following Ansari et al.
(2023). Unlike the reference training implementation, this model exposes a
forecasting distribution and never stores training objectives.
"""

__all__ = [
    "AuxiliaryInference",
    "ContinuousLinearSDE",
    "EmissionNetwork",
    "NCDSSM",
    "NCDSSMConfig",
]

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Final

import torch
from torch import Generator, Tensor, nan, nn
from torch.nn import functional

from .utils import EventBatch

_LOG2PI = math.log(2.0 * math.pi)


class AuxiliaryInference(nn.Module):
    r"""Infer $q_ϕ(aₜ\mid xₜ,mₜ)$, a diagonal Gaussian auxiliary observation."""

    input_size: Final[int]
    auxiliary_size: Final[int]
    network: nn.Sequential

    def __init__(
        self,
        input_size: int,
        auxiliary_size: int,
        hidden_size: int,
        *,
        min_variance: float = 1e-4,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.auxiliary_size = auxiliary_size
        self.min_variance = min_variance
        self.network = nn.Sequential(
            nn.Linear(2 * input_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * auxiliary_size),
        )

    def forward(
        self,
        values: Tensor,  # Float[..., D], sparse
        mask: Tensor,  # Bool[..., D]
        /,
    ) -> tuple[Tensor, Tensor]:  # Float[..., A], Float[..., A]
        r"""Return auxiliary mean and diagonal variance for one sparse event."""
        assert values.shape == mask.shape
        assert values.shape[-1] == self.input_size
        # Values are zero-filled only after concatenating the feature mask, so an
        # observed zero remains distinct from a missing feature.
        safe_values = torch.where(mask, values, torch.zeros_like(values))
        mean, raw_variance = self.network(
            torch.cat([safe_values, mask.to(values.dtype)], dim=-1)
        ).chunk(2, dim=-1)
        return mean, functional.softplus(raw_variance) + self.min_variance


class EmissionNetwork(nn.Module):
    r"""Moment-match auxiliary moments to diagonal output Gaussian parameters.

    The paper decodes $p_θ(xₜ\mid aₜ)$. Exact marginalization over nonlinear
    decoders is intractable, so the network consumes both moments of $aₜ$ and
    parameterizes the predictive diagonal Gaussian directly.
    """

    auxiliary_size: Final[int]
    output_size: Final[int]
    network: nn.Sequential

    def __init__(
        self,
        auxiliary_size: int,
        output_size: int,
        hidden_size: int,
        *,
        min_variance: float = 1e-4,
    ) -> None:
        super().__init__()
        self.auxiliary_size = auxiliary_size
        self.output_size = output_size
        self.min_variance = min_variance
        self.network = nn.Sequential(
            nn.Linear(2 * auxiliary_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * output_size),
        )

    def forward(
        self,
        auxiliary_mean: Tensor,  # Float[..., A]
        auxiliary_variance: Tensor,  # Float[..., A]
        /,
    ) -> tuple[Tensor, Tensor]:  # Float[..., F], Float[..., F]
        r"""Return predictive output mean and diagonal variance."""
        assert auxiliary_mean.shape == auxiliary_variance.shape
        assert auxiliary_mean.shape[-1] == self.auxiliary_size
        mean, raw_variance = self.network(
            torch.cat([auxiliary_mean, auxiliary_variance.log()], dim=-1)
        ).chunk(2, dim=-1)
        return mean, functional.softplus(raw_variance) + self.min_variance


class ContinuousLinearSDE(nn.Module):
    r"""Propagate $dzₜ=Fzₜdt+Ldβₜ$ exactly with a Van Loan exponential."""

    latent_size: Final[int]

    def __init__(self, latent_size: int, *, min_variance: float = 1e-4) -> None:
        super().__init__()
        self.latent_size = latent_size
        self.min_variance = min_variance
        self.drift = nn.Parameter(
            -0.1 * torch.eye(latent_size) + 0.05 * torch.randn(latent_size, latent_size)
        )
        self.process_log_variance = nn.Parameter(torch.full((latent_size,), -2.0))

    @property
    def process_covariance(self) -> Tensor:
        r"""Return $Q=LLᵀ$, constrained to be positive diagonal."""
        return torch.diag(
            functional.softplus(self.process_log_variance) + self.min_variance
        )

    def forward(
        self,
        mean: Tensor,  # Float[..., L]
        covariance: Tensor,  # Float[..., L, L]
        delta_time: Tensor,  # Float[...]
        /,
    ) -> tuple[Tensor, Tensor]:  # Float[..., L], Float[..., L, L]
        r"""Predict state moments after a non-negative time interval."""
        assert mean.shape[:-1] == covariance.shape[:-2] == delta_time.shape
        assert (delta_time >= 0).all(), "delta_time must be non-negative."
        drift = self.drift
        zeros = torch.zeros_like(drift)
        # exp([[F,Q],[0,-Fᵀ]]Δt) = [[Φ,C],[0,Φ⁻ᵀ]], so ∫ΦQΦᵀ=CΦᵀ.
        van_loan = torch.cat(
            [
                torch.cat([drift, self.process_covariance], dim=-1),
                torch.cat([zeros, -drift.mT], dim=-1),
            ],
            dim=-2,
        )
        exponential = torch.linalg.matrix_exp(delta_time[..., None, None] * van_loan)
        transition = exponential[..., : self.latent_size, : self.latent_size]
        integral = (
            exponential[..., : self.latent_size, self.latent_size :] @ transition.mT
        )
        next_mean = (transition @ mean.unsqueeze(-1)).squeeze(-1)
        next_covariance = transition @ covariance @ transition.mT + integral
        return next_mean, 0.5 * (next_covariance + next_covariance.mT)


@dataclass(frozen=True, slots=True, kw_only=True)
class NCDSSMConfig:
    r"""Configuration used by :meth:`NCDSSM.from_config`."""

    input_size: int
    output_size: int
    latent_size: int
    auxiliary_size: int
    encoder_hidden_size: int
    decoder_hidden_size: int
    initial_variance: float = 1.0
    min_variance: float = 1e-4
    batch_first: bool = True
    validate_args: bool = True


class NCDSSM(nn.Module):
    r"""Neural continuous-discrete state-space model for sparse forecasting.

    .. math::

        dzₜ &= Fzₜdt + Ldβₜ, \\
        aₜ\mid zₜ &∼ 𝓝(Hzₜ,R), \\
        q_ϕ(aₜ\mid xₜ,mₜ) &= 𝓝(μₐ,Rₐ), \\
        xₜ\mid𝒟 &≈ 𝓝(μₓ,\operatorname{diag}(σₓ²)).

    The first two equations are Equations (3)--(4) of Ansari et al. (2023).
    Context is encoded into auxiliary observations and filtered with a Kalman
    update; the emission network moment-matches the output distribution.
    """

    input_size: Final[int]
    output_size: Final[int]
    latent_size: Final[int]
    auxiliary_size: Final[int]
    batch_first: Final[bool]
    validate_args: Final[bool]

    pred_means: Tensor
    r"""BUFFER: Predictive means from the latest :meth:`predict` call."""
    pred_variances: Tensor
    r"""BUFFER: Predictive variances from the latest :meth:`predict` call."""
    pred_logvars: Tensor
    r"""BUFFER: Log variances from the latest :meth:`predict` call."""

    @classmethod
    def from_config(cls, config: NCDSSMConfig | Mapping[str, Any], /) -> NCDSSM:
        r"""Build an NCDSSM and its required neural submodules."""
        if isinstance(config, Mapping):
            config = NCDSSMConfig(**config)
        encoder = AuxiliaryInference(
            config.input_size,
            config.auxiliary_size,
            config.encoder_hidden_size,
            min_variance=config.min_variance,
        )
        emission = EmissionNetwork(
            config.auxiliary_size,
            config.output_size,
            config.decoder_hidden_size,
            min_variance=config.min_variance,
        )
        return cls(
            config.input_size,
            config.output_size,
            config.latent_size,
            config.auxiliary_size,
            encoder=encoder,
            emission=emission,
            initial_variance=config.initial_variance,
            min_variance=config.min_variance,
            batch_first=config.batch_first,
            validate_args=config.validate_args,
        )

    @classmethod
    def from_parameters(
        cls,
        *,
        input_size: int,
        output_size: int,
        latent_size: int,
        auxiliary_size: int,
        encoder_hidden_size: int,
        decoder_hidden_size: int,
        initial_variance: float = 1.0,
        min_variance: float = 1e-4,
        batch_first: bool = True,
        validate_args: bool = True,
    ) -> NCDSSM:
        r"""Build an NCDSSM directly from hyperparameters."""
        return cls.from_config(
            NCDSSMConfig(
                input_size=input_size,
                output_size=output_size,
                latent_size=latent_size,
                auxiliary_size=auxiliary_size,
                encoder_hidden_size=encoder_hidden_size,
                decoder_hidden_size=decoder_hidden_size,
                initial_variance=initial_variance,
                min_variance=min_variance,
                batch_first=batch_first,
                validate_args=validate_args,
            )
        )

    def __init__(
        self,
        input_size: int,
        output_size: int,
        latent_size: int,
        auxiliary_size: int,
        *,
        encoder: AuxiliaryInference,
        emission: EmissionNetwork,
        dynamics: ContinuousLinearSDE | None = None,
        initial_variance: float = 1.0,
        min_variance: float = 1e-4,
        batch_first: bool = True,
        validate_args: bool = True,
    ) -> None:
        super().__init__()
        if min(input_size, output_size, latent_size, auxiliary_size) < 1:
            raise ValueError("All model dimensions must be positive.")
        if initial_variance <= 0 or min_variance <= 0:
            raise ValueError("Variances must be positive.")
        self.input_size = input_size
        self.output_size = output_size
        self.latent_size = latent_size
        self.auxiliary_size = auxiliary_size
        self.batch_first = batch_first
        self.validate_args = validate_args
        self.min_variance = min_variance
        self.encoder = encoder
        self.emission = emission
        self.dynamics = dynamics or ContinuousLinearSDE(
            latent_size, min_variance=min_variance
        )

        # $z₀∼𝓝(μ₀,Σ₀)$ and $aₜ\mid zₜ∼𝓝(Hzₜ,R)$.
        self.initial_mean = nn.Parameter(torch.zeros(latent_size))
        raw_initial_variance = math.log(
            math.expm1(max(initial_variance - min_variance, 1e-8))
        )
        self.initial_log_variance = nn.Parameter(
            torch.full((latent_size,), raw_initial_variance)
        )
        self.observation_matrix = nn.Parameter(
            nn.init.xavier_uniform_(torch.empty(auxiliary_size, latent_size))
        )
        self.observation_log_variance = nn.Parameter(
            torch.full((auxiliary_size,), -2.0)
        )
        self.register_buffer("identity", torch.eye(latent_size))
        self.register_buffer("pred_means", torch.empty(0), persistent=False)
        self.register_buffer("pred_variances", torch.empty(0), persistent=False)
        self.register_buffer("pred_logvars", torch.empty(0), persistent=False)

    @property
    def initial_covariance(self) -> Tensor:
        r"""Positive diagonal initial covariance $Σ₀$."""
        return torch.diag(
            functional.softplus(self.initial_log_variance) + self.min_variance
        )

    @property
    def observation_covariance(self) -> Tensor:
        r"""Positive diagonal auxiliary observation covariance $R$."""
        return torch.diag(
            functional.softplus(self.observation_log_variance) + self.min_variance
        )

    def _update(
        self,
        prior_mean: Tensor,  # Float[..., L]
        prior_covariance: Tensor,  # Float[..., L, L]
        auxiliary_mean: Tensor,  # Float[..., A]
        auxiliary_variance: Tensor,  # Float[..., A]
        /,
    ) -> tuple[Tensor, Tensor]:  # Float[..., L], Float[..., L, L]
        r"""Apply the Equation (4) auxiliary-observation Kalman update."""
        observation_matrix = self.observation_matrix  # Float[A, L]
        observation_covariance = torch.diag_embed(auxiliary_variance)  # (..., A, A)
        cross_covariance = prior_covariance @ observation_matrix.mT  # (..., L, A)
        innovation_covariance = (  # (..., A, A)
            observation_matrix @ cross_covariance + observation_covariance
        )
        gain = torch.linalg.solve(innovation_covariance, cross_covariance.mT).mT
        innovation = auxiliary_mean - prior_mean @ observation_matrix.mT  # (..., A)
        posterior_mean = prior_mean + (gain @ innovation.unsqueeze(-1)).squeeze(-1)
        correction = self.identity - gain @ observation_matrix  # (..., L, L)
        # Joseph form: (I-KH)P(I-KH)ᵀ + KRₐKᵀ preserves positive semidefiniteness.
        posterior_covariance = (
            correction @ prior_covariance @ correction.mT
            + gain @ observation_covariance @ gain.mT
        )
        return posterior_mean, 0.5 * (posterior_covariance + posterior_covariance.mT)

    def _emit(
        self,
        state_mean: Tensor,  # Float[..., L]
        state_covariance: Tensor,  # Float[..., L, L]
        /,
    ) -> tuple[Tensor, Tensor]:  # Float[..., F], Float[..., F]
        r"""Moment-match $p_θ(xₜ\mid aₜ)$ given predictive state moments."""
        observation_matrix = self.observation_matrix  # Float[A, L]
        auxiliary_mean = state_mean @ observation_matrix.mT  # (..., A)
        auxiliary_covariance = (  # (..., A, A)
            observation_matrix @ state_covariance @ observation_matrix.mT
            + self.observation_covariance
        )
        auxiliary_variance = torch.diagonal(
            auxiliary_covariance, dim1=-2, dim2=-1
        ).clamp_min(self.min_variance)  # (..., A)
        return self.emission(auxiliary_mean, auxiliary_variance)

    def forward(
        self,
        *,
        timestamps: Tensor,  # Float[..., T], padded NaN, non-decreasing
        query_mask: Tensor,  # Bool[..., T, F], padded False
        context_values: Tensor,  # Float[..., T, D], padded NaN, sparse
        context_mask: Tensor,  # Bool[..., T, D], padded False
        initial_state: tuple[Tensor, Tensor] | None = None,  # (..., L), (..., L, L)
        initial_time: Tensor | None = None,  # Float[] or Float[...]
    ) -> tuple[Tensor, Tensor]:  # Float[..., T, F], Float[..., T, F]
        r"""Filter and forecast over an ordered, combined event request.

        Args:
            timestamps: Combined context and query timestamps.
            query_mask: Feature-level mask selecting requested output values.
            context_values: Sparse context values at combined event times.
            context_mask: Feature-level mask selecting observed context values.
            initial_state: Optional initial latent Gaussian moments $(μ₀, Σ₀)$.
            initial_time: Optional time associated with ``initial_state``.

        Returns:
            predicted_means: Output means for the combined event sequence. Values
                outside ``query_mask`` are NaN.
            predicted_variances: Output variances for the combined event sequence.
                Values outside ``query_mask`` are NaN.
        """
        has_context = context_mask.any(dim=-1)  # (..., T)
        has_query = query_mask.any(dim=-1)  # (..., T)
        valid_steps = timestamps.isfinite() & (has_context | has_query)  # (..., T)

        if self.validate_args:
            assert (
                context_values.shape[:-1] == context_mask.shape[:-1] == timestamps.shape
            )
            assert context_values.shape[-1] == context_mask.shape[-1] == self.input_size
            assert query_mask.shape[:-1] == timestamps.shape
            assert query_mask.shape[-1] == self.output_size
            assert torch.equal(context_values.isfinite(), context_mask)
            assert torch.equal(timestamps.isfinite(), valid_steps)

        output_mask = query_mask
        if self.batch_first:
            timestamps = timestamps.movedim(-1, 0)  # (T, ...)
            context_values = context_values.movedim(-2, 0)  # (T, ..., D)
            context_mask = context_mask.movedim(-2, 0)  # (T, ..., D)
            has_context = has_context.movedim(-1, 0)  # (T, ...)
            valid_steps = valid_steps.movedim(-1, 0)  # (T, ...)

        num_steps, *batch_shape = timestamps.shape
        assert context_values.shape == (num_steps, *batch_shape, self.input_size)
        assert context_mask.shape == context_values.shape
        assert has_context.shape == valid_steps.shape == (num_steps, *batch_shape)

        state_mean: Tensor  # Float[..., L]
        state_covariance: Tensor  # Float[..., L, L]
        if initial_state is None:
            state_mean = self.initial_mean.expand(*batch_shape, self.latent_size)
            state_covariance = self.initial_covariance.expand(
                *batch_shape, self.latent_size, self.latent_size
            )
        else:
            state_mean, state_covariance = initial_state
            state_mean = state_mean.expand(*batch_shape, self.latent_size)
            state_covariance = state_covariance.expand(
                *batch_shape, self.latent_size, self.latent_size
            )
        state_time = timestamps[0] if initial_time is None else initial_time  # (...,)

        predicted_means: list[Tensor] = []
        predicted_variances: list[Tensor] = []
        for timestamp, values, feature_mask, is_context, active in zip(
            timestamps,
            context_values,
            context_mask,
            has_context,
            valid_steps,
            strict=True,
        ):
            # Equation (3): propagate $p(zₛ\mid𝒟)$ to $p(zₜ\mid𝒟)$.
            delta_time = torch.where(
                active, timestamp - state_time, torch.zeros_like(timestamp)
            )
            prior_mean, prior_covariance = self.dynamics(
                state_mean, state_covariance, delta_time
            )
            state_mean = torch.where(active[..., None], prior_mean, state_mean)
            state_covariance = torch.where(
                active[..., None, None], prior_covariance, state_covariance
            )

            # Equation (6): infer qϕ(aₜ|xₜ,mₜ), then update Equation (4).
            auxiliary_mean, auxiliary_variance = self.encoder(values, feature_mask)
            posterior_mean, posterior_covariance = self._update(
                state_mean,
                state_covariance,
                auxiliary_mean,
                auxiliary_variance,
            )
            update = active & is_context
            state_mean = torch.where(update[..., None], posterior_mean, state_mean)
            state_covariance = torch.where(
                update[..., None, None], posterior_covariance, state_covariance
            )
            state_time = torch.where(active, timestamp, state_time)

            output_mean, output_variance = self._emit(state_mean, state_covariance)
            predicted_means.append(output_mean)
            predicted_variances.append(output_variance)

        stack_dim = -2 if self.batch_first else 0
        mean = torch.stack(predicted_means, dim=stack_dim)  # (..., T, F)
        variance = torch.stack(predicted_variances, dim=stack_dim)  # (..., T, F)
        return mean.masked_fill(~output_mask, nan), variance.masked_fill(
            ~output_mask, nan
        )

    def predict(
        self,
        *,
        query_times: Tensor,  # Float[..., K], padded NaN, non-decreasing
        query_mask: Tensor,  # Bool[..., K, F], padded False
        context_times: Tensor,  # Float[..., N], padded NaN, non-decreasing
        context_values: Tensor,  # Float[..., N, D], padded NaN, sparse
        context_mask: Tensor,  # Bool[..., N, D], padded False
        initial_state: tuple[Tensor, Tensor] | None = None,  # (..., L), (..., L, L)
        initial_time: Tensor | None = None,  # Float[] or Float[...]
    ) -> tuple[Tensor, Tensor]:  # Float[..., K, F], Float[..., K, F]
        r"""Return predictive output moments at the requested query events."""
        combined = EventBatch.from_request(
            query_times=query_times,
            query_mask=query_mask,
            context_times=context_times,
            context_values=context_values,
            context_mask=context_mask,
            batch_first=self.batch_first,
        )
        if self.validate_args:
            combined.validate()
        means, variances = self(
            timestamps=combined.timestamps,
            query_mask=combined.query_mask,
            context_values=combined.context_values,
            context_mask=combined.context_mask,
            initial_state=initial_state,
            initial_time=initial_time,
        )
        self.pred_means = means[..., *combined.query_indices, :]
        self.pred_variances = variances[..., *combined.query_indices, :]
        self.pred_logvars = self.pred_variances.log()
        return self.pred_means, self.pred_variances

    def log_prob(
        self,
        samples: Tensor,  # Float[*S, ..., K, F]
        /,
        *,
        query_times: Tensor,  # Float[..., K], padded NaN, non-decreasing
        query_mask: Tensor,  # Bool[..., K, F], padded False
        context_times: Tensor,  # Float[..., N], padded NaN, non-decreasing
        context_values: Tensor,  # Float[..., N, D], padded NaN, sparse
        context_mask: Tensor,  # Bool[..., N, D], padded False
    ) -> Tensor:  # Float[*S, ..., K]
        r"""Compute time-marginal predictive log-likelihoods of ``samples``."""
        mean, variance = self.predict(
            query_times=query_times,
            query_mask=query_mask,
            context_times=context_times,
            context_values=context_values,
            context_mask=context_mask,
        )

        mean = mean.expand_as(samples)  # Float[*S, ..., K, F]
        variance = variance.expand_as(samples)  # Float[*S, ..., K, F]
        mask = query_mask.expand_as(samples)  # Bool[*S, ..., K, F]

        safe_values = torch.where(mask, samples, 0.0)
        safe_mean = torch.where(mask, mean, 0.0)
        safe_variance = torch.where(mask, variance, 1.0)
        log_prob = -0.5 * (
            (safe_values - safe_mean).square() / safe_variance
            + safe_variance.log()
            + _LOG2PI
        )
        return torch.where(mask, log_prob, 0.0).sum(dim=-1)

    def sample(
        self,
        size: int | tuple[int, ...] = (),  # *S
        *,
        query_times: Tensor,  # Float[..., K], padded NaN, non-decreasing
        query_mask: Tensor,  # Bool[..., K, F], padded False
        context_times: Tensor,  # Float[..., N], padded NaN, non-decreasing
        context_values: Tensor,  # Float[..., N, D], padded NaN, sparse
        context_mask: Tensor,  # Bool[..., N, D], padded False
        rng: Generator | None = None,
    ) -> Tensor:  # Float[*S, ..., K, F]
        r"""Draw independent samples from the predictive time marginals."""
        mean, variance = self.predict(
            query_times=query_times,
            query_mask=query_mask,
            context_times=context_times,
            context_values=context_values,
            context_mask=context_mask,
        )
        sample_shape = (size,) if isinstance(size, int) else size
        noise = torch.randn(
            *sample_shape,
            *mean.shape,
            dtype=mean.dtype,
            device=mean.device,
            generator=rng,
        )
        samples = mean.expand_as(noise) + variance.sqrt().expand_as(noise) * noise
        samples = samples.masked_fill(~query_mask.expand_as(samples), nan)
        return samples

    def sample_and_log_prob(
        self,
        size: int | tuple[int, ...] = (),  # *S
        *,
        query_times: Tensor,  # Float[..., K], padded NaN, non-decreasing
        query_mask: Tensor,  # Bool[..., K, F], padded False
        context_times: Tensor,  # Float[..., N], padded NaN, non-decreasing
        context_values: Tensor,  # Float[..., N, D], padded NaN, sparse
        context_mask: Tensor,  # Bool[..., N, D], padded False
        rng: Generator | None = None,
    ) -> tuple[Tensor, Tensor]:  # Float[*S, ..., K, F], Float[*S, ..., K]
        r"""Draw samples and score those same predictive marginals."""
        samples = self.sample(
            size,
            query_times=query_times,
            query_mask=query_mask,
            context_times=context_times,
            context_values=context_values,
            context_mask=context_mask,
            rng=rng,
        )
        log_prob = self.log_prob(
            samples,
            query_times=query_times,
            query_mask=query_mask,
            context_times=context_times,
            context_values=context_values,
            context_mask=context_mask,
        )
        return samples, log_prob
