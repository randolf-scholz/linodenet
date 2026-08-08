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

    def forward(self, values: Tensor, mask: Tensor, /) -> tuple[Tensor, Tensor]:
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
        self, auxiliary_mean: Tensor, auxiliary_variance: Tensor, /
    ) -> tuple[Tensor, Tensor]:
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
    ) -> tuple[Tensor, Tensor]:
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

    def _canonical_request(
        self,
        *,
        query_times: Tensor,
        query_mask: Tensor,
        context_times: Tensor,
        context_values: Tensor,
        context_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        r"""Convert a request to ``(..., time, feature)`` and validate shapes."""
        if not self.batch_first:
            query_times = query_times.movedim(0, -1)
            query_mask = query_mask.movedim(0, -2)
            context_times = context_times.movedim(0, -1)
            context_values = context_values.movedim(0, -2)
            context_mask = context_mask.movedim(0, -2)
        batch_shape = context_times.shape[:-1]
        if context_values.shape != (
            *batch_shape,
            context_times.shape[-1],
            self.input_size,
        ):
            raise ValueError("context_values has incompatible shape.")
        if (
            context_mask.shape != context_values.shape
            or context_mask.dtype != torch.bool
        ):
            raise ValueError("context_mask must be boolean and match context_values.")
        if query_times.shape[:-1] != batch_shape:
            raise ValueError("Context and query batch shapes must match.")
        if query_mask.shape != (*batch_shape, query_times.shape[-1], self.output_size):
            raise ValueError("query_mask has incompatible shape.")
        if query_mask.dtype != torch.bool:
            raise ValueError("query_mask must be boolean.")
        if self.validate_args:
            context_active = context_mask.any(dim=-1)
            query_active = query_mask.any(dim=-1)
            if not torch.equal(context_values.isfinite(), context_mask):
                raise ValueError("context_values must be finite exactly where masked.")
            if not torch.equal(context_times.isfinite(), context_active):
                raise ValueError(
                    "context_times must be finite exactly at context events."
                )
            if not torch.equal(query_times.isfinite(), query_active):
                raise ValueError("query_times must be finite exactly at query events.")
        return query_times, query_mask, context_times, context_values, context_mask

    def _update(
        self,
        prior_mean: Tensor,
        prior_covariance: Tensor,
        auxiliary_mean: Tensor,
        auxiliary_variance: Tensor,
        /,
    ) -> tuple[Tensor, Tensor]:
        r"""Apply the Equation (4) auxiliary-observation Kalman update."""
        observation_matrix = self.observation_matrix
        observation_covariance = torch.diag_embed(auxiliary_variance)
        cross_covariance = prior_covariance @ observation_matrix.mT
        innovation_covariance = (
            observation_matrix @ cross_covariance + observation_covariance
        )
        gain = torch.linalg.solve(innovation_covariance, cross_covariance.mT).mT
        innovation = auxiliary_mean - prior_mean @ observation_matrix.mT
        posterior_mean = prior_mean + (gain @ innovation.unsqueeze(-1)).squeeze(-1)
        correction = self.identity - gain @ observation_matrix
        # Joseph form: (I-KH)P(I-KH)ᵀ + KRₐKᵀ preserves positive semidefiniteness.
        posterior_covariance = (
            correction @ prior_covariance @ correction.mT
            + gain @ observation_covariance @ gain.mT
        )
        return posterior_mean, 0.5 * (posterior_covariance + posterior_covariance.mT)

    def _emit(
        self, state_mean: Tensor, state_covariance: Tensor, /
    ) -> tuple[Tensor, Tensor]:
        r"""Moment-match $p_θ(xₜ\mid aₜ)$ given predictive state moments."""
        observation_matrix = self.observation_matrix
        auxiliary_mean = state_mean @ observation_matrix.mT
        auxiliary_covariance = (
            observation_matrix @ state_covariance @ observation_matrix.mT
            + self.observation_covariance
        )
        auxiliary_variance = torch.diagonal(
            auxiliary_covariance, dim1=-2, dim2=-1
        ).clamp_min(self.min_variance)
        return self.emission(auxiliary_mean, auxiliary_variance)

    def predict(
        self,
        *,
        query_times: Tensor,
        query_mask: Tensor,
        context_times: Tensor,
        context_values: Tensor,
        context_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        r"""Return predictive mean and diagonal variance at all query times.

        The state-space loop filters the context, then propagates the posterior
        through each query time without conditioning on unknown query values.
        Padded events leave both the state and state time intact.
        """
        query_times, query_mask, context_times, context_values, context_mask = (
            self._canonical_request(
                query_times=query_times,
                query_mask=query_mask,
                context_times=context_times,
                context_values=context_values,
                context_mask=context_mask,
            )
        )
        batch_shape = context_times.shape[:-1]
        batch_size = math.prod(batch_shape) if batch_shape else 1
        num_context, num_queries = context_times.shape[-1], query_times.shape[-1]
        context_times = context_times.reshape(batch_size, num_context)
        context_values = context_values.reshape(
            batch_size, num_context, self.input_size
        )
        context_mask = context_mask.reshape(batch_size, num_context, self.input_size)
        query_times = query_times.reshape(batch_size, num_queries)
        query_mask = query_mask.reshape(batch_size, num_queries, self.output_size)

        state_mean = self.initial_mean.expand(batch_size, -1)
        state_covariance = self.initial_covariance.expand(batch_size, -1, -1)
        state_time = context_times.new_zeros(batch_size)
        for index in range(num_context):
            feature_mask = context_mask[:, index]
            active = feature_mask.any(dim=-1) & context_times[:, index].isfinite()
            event_time = torch.where(active, context_times[:, index], state_time)
            prior_mean, prior_covariance = self.dynamics(
                state_mean, state_covariance, event_time - state_time
            )
            auxiliary_mean, auxiliary_variance = self.encoder(
                context_values[:, index], feature_mask
            )
            posterior_mean, posterior_covariance = self._update(
                prior_mean, prior_covariance, auxiliary_mean, auxiliary_variance
            )
            state_mean = torch.where(active[:, None], posterior_mean, state_mean)
            state_covariance = torch.where(
                active[:, None, None], posterior_covariance, state_covariance
            )
            state_time = torch.where(active, event_time, state_time)

        means: list[Tensor] = []
        variances: list[Tensor] = []
        for index in range(num_queries):
            active = query_mask[:, index].any(dim=-1) & query_times[:, index].isfinite()
            event_time = torch.where(active, query_times[:, index], state_time)
            next_mean, next_covariance = self.dynamics(
                state_mean, state_covariance, event_time - state_time
            )
            state_mean = torch.where(active[:, None], next_mean, state_mean)
            state_covariance = torch.where(
                active[:, None, None], next_covariance, state_covariance
            )
            state_time = torch.where(active, event_time, state_time)
            output_mean, output_variance = self._emit(state_mean, state_covariance)
            means.append(output_mean)
            variances.append(output_variance)

        mean = torch.stack(means, dim=-2).reshape(
            *batch_shape, num_queries, self.output_size
        )
        variance = torch.stack(variances, dim=-2).reshape(
            *batch_shape, num_queries, self.output_size
        )
        if not self.batch_first:
            mean, variance = mean.movedim(-2, 0), variance.movedim(-2, 0)
        self.pred_means, self.pred_variances = mean, variance
        self.pred_logvars = variance.log()
        return mean, variance

    @staticmethod
    def _log_prob(
        values: Tensor, *, mean: Tensor, variance: Tensor, mask: Tensor
    ) -> Tensor:
        r"""Compute masked time-marginal diagonal Gaussian log-probabilities."""
        safe_values = torch.where(mask, values, torch.zeros_like(values))
        safe_mean = torch.where(mask, mean, torch.zeros_like(mean))
        safe_variance = torch.where(mask, variance, torch.ones_like(variance))
        log_prob = -0.5 * (
            (safe_values - safe_mean).square() / safe_variance
            + safe_variance.log()
            + _LOG2PI
        )
        return torch.where(mask, log_prob, torch.zeros_like(log_prob)).sum(dim=-1)

    def log_prob(
        self,
        samples: Tensor,
        /,
        *,
        query_times: Tensor,
        query_mask: Tensor,
        context_times: Tensor,
        context_values: Tensor,
        context_mask: Tensor,
    ) -> Tensor:
        r"""Compute time-marginal predictive log-likelihoods of ``samples``."""
        mean, variance = self.predict(
            query_times=query_times,
            query_mask=query_mask,
            context_times=context_times,
            context_values=context_values,
            context_mask=context_mask,
        )
        sample_dims = samples.ndim - query_mask.ndim
        if sample_dims < 0:
            raise ValueError("samples has fewer dimensions than query_mask.")
        if not self.batch_first:
            samples = samples.movedim(sample_dims, -2)
            query_mask = query_mask.movedim(0, -2)
            mean = mean.movedim(0, -2)
            variance = variance.movedim(0, -2)
        if samples.shape[-2:] != mean.shape[-2:]:
            raise ValueError("samples must end in query-time and output dimensions.")
        log_prob = self._log_prob(
            samples,
            mean=mean.expand_as(samples),
            variance=variance.expand_as(samples),
            mask=query_mask.expand_as(samples),
        )
        return log_prob.movedim(-1, sample_dims) if not self.batch_first else log_prob

    def sample(
        self,
        size: int | tuple[int, ...] = (),
        *,
        query_times: Tensor,
        query_mask: Tensor,
        context_times: Tensor,
        context_values: Tensor,
        context_mask: Tensor,
        rng: Generator | None = None,
    ) -> Tensor:
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
        size: int | tuple[int, ...] = (),
        *,
        query_times: Tensor,
        query_mask: Tensor,
        context_times: Tensor,
        context_values: Tensor,
        context_mask: Tensor,
        rng: Generator | None = None,
    ) -> tuple[Tensor, Tensor]:
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
