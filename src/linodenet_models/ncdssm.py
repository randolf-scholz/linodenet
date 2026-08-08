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
from torch.nn.functional import softplus

from .utils import EventBatch

_LOG2PI = math.log(2.0 * math.pi)


class AuxiliaryInference(nn.Module):
    r"""Infer diagonal Gaussian $q_ϕ(aₖ∣yₖ,Mₖ)$ from sparse observations.

    This extends the per-timestep recognition distribution in Equation (14) of
    Ansari et al. (2023) with the feature mask $Mₖ$.
    """

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
        /,
        mask: Tensor,  # Bool[..., D]
    ) -> tuple[Tensor, Tensor]:  # Float[..., A], Float[..., A]
        r"""Return auxiliary mean and diagonal variance for one sparse event."""
        assert values.shape == mask.shape
        assert values.shape[-1] == self.input_size
        # Values are zero-filled only after concatenating the feature mask, so an
        # observed zero remains distinct from a missing feature.
        y = torch.where(mask, values, 0.0)
        μ_a, raw_Σ_a = self.network(
            torch.cat([y, mask.to(values.dtype)], dim=-1)
        ).chunk(2, dim=-1)
        Σ_a = softplus(raw_Σ_a) + self.min_variance
        return μ_a, Σ_a


class EmissionNetwork(nn.Module):
    r"""Moment-match auxiliary moments to diagonal output Gaussian parameters.

    The paper decodes $p_θ(yₖ∣aₖ)$ in Equation (9). Exact marginalization over nonlinear
    decoders is intractable, so the network consumes both moments of $aₖ$ and
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
        mean: Tensor,  # Float[..., A]
        variance: Tensor,  # Float[..., A]
        /,
    ) -> tuple[Tensor, Tensor]:  # Float[..., F], Float[..., F]
        r"""Return predictive output mean and diagonal variance."""
        assert mean.shape == variance.shape
        assert mean.shape[-1] == self.auxiliary_size
        μ_y, raw_σ_y = self.network(torch.cat([mean, variance.log()], dim=-1)).chunk(
            2, dim=-1
        )
        σ_y = softplus(raw_σ_y) + self.min_variance
        return μ_y, σ_y


class ContinuousLinearSDE(nn.Module):
    r"""Propagate the Equation (10) LTI dynamics exactly with Van Loan's method.

    The SDE is $dzₜ=Fzₜdt+Ldβₜ$, where $Q=LLᵀ$ is the process covariance.
    """

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
        return torch.diag(softplus(self.process_log_variance) + self.min_variance)

    def forward(
        self,
        mean: Tensor,  # Float[..., L]
        cov: Tensor,  # Float[..., L, L]
        delta_t: Tensor,  # Float[...]
        /,
    ) -> tuple[Tensor, Tensor]:  # Float[..., L], Float[..., L, L]
        r"""Predict state moments after a non-negative time interval."""
        assert mean.shape[:-1] == cov.shape[:-2] == delta_t.shape
        assert (delta_t >= 0).all(), "Δt must be non-negative."
        # exp([[F,Q],[0,-Fᵀ]]Δt) = [[Φ,C],[0,Φ⁻ᵀ]], so ∫ΦQΦᵀ=CΦᵀ.
        zeros = torch.zeros_like(self.drift)
        van_loan = torch.cat(
            [
                torch.cat([self.drift, self.process_covariance], dim=-1),
                torch.cat([zeros, -self.drift.mT], dim=-1),
            ],
            dim=-2,
        )
        E = torch.linalg.matrix_exp(delta_t[..., None, None] * van_loan)
        Φ = E[..., : self.latent_size, : self.latent_size]
        C = E[..., : self.latent_size, self.latent_size :] @ Φ.mT
        mean = (Φ @ mean.unsqueeze(-1)).squeeze(-1)
        cov = Φ @ cov @ Φ.mT + C
        return mean, 0.5 * (cov + cov.mT)


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
        aₖ∣zₖ &∼ 𝓝(Hzₖ,R), \\
        q_ϕ(aₖ∣yₖ,Mₖ) &= 𝓝(μₐ,diag(σₐ)), \\
        yₖ∣𝒟 &≈ 𝓝(μᵧ,diag(Σᵧ)).

    This is the linear time-invariant specialization in Equation (10) of
    Ansari et al. (2023), with the initial and emission distributions from
    Equations (7)–(9). The recognition distribution follows Equation (14),
    extended to account for a feature-level missingness mask $Mₖ$. Although
    Equation (14) denotes its second parameter as a covariance matrix, the
    $2h$ auxiliary-inference-network output specified in Appendix C.3 (p. 18)
    implies $h$ independent variances, represented here by $diag(σₐ)$. Context
    is filtered through pseudo-observations; the emission network
    moment-matches the output distribution.
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
        self.dynamics = (
            dynamics
            if dynamics is not None
            else ContinuousLinearSDE(latent_size, min_variance=min_variance)
        )

        # Equations (7)–(8): $z₀∼𝓝(μ₀,Σ₀)$ and $aₖ∣zₖ∼𝓝(Hzₖ,R)$.
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
        return torch.diag(softplus(self.initial_log_variance) + self.min_variance)

    @property
    def observation_covariance(self) -> Tensor:
        r"""Positive diagonal auxiliary observation covariance $R$."""
        return torch.diag(softplus(self.observation_log_variance) + self.min_variance)

    def _update(
        self,
        mean: Tensor,  # Float[..., L]
        cov: Tensor,  # Float[..., L, L]
        μ_a: Tensor,  # Float[..., A]
        σ_a: Tensor,  # Float[..., A]
        /,
    ) -> tuple[Tensor, Tensor]:  # Float[..., L], Float[..., L, L]
        r"""Apply Equation (6) to the recognition pseudo-observation moments.

        The update uses $μₐ$ as the pseudo-observation and diagonal $Σₐ$ as
        its covariance, rather than sampling $aₖ$ from Equation (14).
        """
        m = mean
        P = cov
        H = self.observation_matrix  # Float[A, L]
        σ_a = torch.diag_embed(σ_a)  # (..., A, A)
        PHt = P @ H.mT  # (..., L, A)
        S = H @ PHt + σ_a  # (..., A, A), Equation (6a)
        K = torch.linalg.solve(S, PHt.mT).mT  # (..., L, A), Equation (6b)
        innovation = μ_a - m @ H.mT  # (..., A)
        m = m + (K @ innovation.unsqueeze(-1)).squeeze(-1)  # Equation (6c)
        I_KH = self.identity - K @ H  # (..., L, L)
        # Joseph form: (I-KH)P(I-KH)ᵀ + KΣₐKᵀ preserves positive semidefiniteness.
        P = I_KH @ P @ I_KH.mT + K @ σ_a @ K.mT
        return m, 0.5 * (P + P.mT)

    def _emit(
        self,
        m: Tensor,  # Float[..., L]
        P: Tensor,  # Float[..., L, L]
        /,
    ) -> tuple[Tensor, Tensor]:  # Float[..., F], Float[..., F]
        r"""Moment-match Equation (9) over predictive auxiliary moments."""
        H = self.observation_matrix  # Float[A, L]
        μ_a = m @ H.mT  # (..., A)
        Σ_a = H @ P @ H.mT + self.observation_covariance  # (..., A, A)
        σ_a = torch.diagonal(Σ_a, dim1=-2, dim2=-1).clamp_min(self.min_variance)
        return self.emission(μ_a, σ_a)

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

        m: Tensor  # Float[..., L]
        P: Tensor  # Float[..., L, L]
        if initial_state is None:
            m = self.initial_mean.expand(*batch_shape, self.latent_size)
            P = self.initial_covariance.expand(
                *batch_shape, self.latent_size, self.latent_size
            )
        else:
            m, P = initial_state
            m = m.expand(*batch_shape, self.latent_size)
            P = P.expand(*batch_shape, self.latent_size, self.latent_size)
        t = timestamps[0] if initial_time is None else initial_time  # (...,)

        μ_ys: list[Tensor] = []
        σ_ys: list[Tensor] = []
        for t_next, y, M, is_context, active in zip(
            timestamps,
            context_values,
            context_mask,
            has_context,
            valid_steps,
            strict=True,
        ):
            # Equation (5) predicts between context events; Equation (10) is exact.
            Δt = torch.where(active, t_next - t, 0.0)
            m_prior, P_prior = self.dynamics(m, P, Δt)
            m = torch.where(active[..., None], m_prior, m)
            P = torch.where(active[..., None, None], P_prior, P)

            # Equation (14) infers $q_ϕ(aₖ∣yₖ,Mₖ)$; Equation (6) updates $m,P$.
            μ_a, σ_a = self.encoder(y, M)
            m_posterior, P_posterior = self._update(m, P, μ_a, σ_a)
            has_update = active & is_context
            m = torch.where(has_update[..., None], m_posterior, m)
            P = torch.where(has_update[..., None, None], P_posterior, P)
            t = torch.where(active, t_next, t)

            μ_y, σ_y = self._emit(m, P)
            μ_ys.append(μ_y)
            σ_ys.append(σ_y)

        dim = -2 if self.batch_first else 0
        μ_y = torch.stack(μ_ys, dim=dim)  # (..., T, F)
        σ_y = torch.stack(σ_ys, dim=dim)  # (..., T, F)
        return (
            μ_y.masked_fill(~output_mask, nan),
            σ_y.masked_fill(~output_mask, nan),
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
