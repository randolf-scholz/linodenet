r"""Models of the LinODE-Net package."""

__all__ = [
    "PointForecastingModel",
    "ProbabilisticForecastingModel",
    "PathForecastingModel",
    "ProbabilisticLSSM",
]

from abc import abstractmethod
from typing import Protocol

from torch import Generator, Tensor


class PointForecastingModel(Protocol):
    r"""Protocol for (point)-forecasting models.

    A point-forecasting model is a model that predicts a single value for a future
    time step, given the past observations.

    .. math::   model(t, S) = ŷ(t，S)

    The inputs to the model are:

      - Query $q = (t₁, t₂, …, tₙ)$ are the time indices we want to predict at
      - Context $S = ((τ₁, x₁), (τ₂, x₂), …, (τₘ, xₘ))$ are the observations and covariates.

    The general forecasting model must allow arbitrary number of observations and
    target time steps.
    """

    @abstractmethod
    def __call__(self, query: Tensor, context: Tensor, /) -> Tensor:
        r"""Forward pass of the model.

        .. math::
            model: \Seq(𝓣)×\Seq(𝓣×𝓧) ⟶ \Seq(𝓧),
            ((t₁, …, tₙ), S) ⟼ (ŷ_{t₁}, ŷ_{t₂}, …, ŷ_{tₙ})

        Args:
            query: $q = (t₁, t₂, …, tₙ)$ are the time indices we want to predict at
            context: $S = ((τ₁, x₁), (τ₂, x₂), …, (τₘ, xₘ))$ are the observations and covariates.

        Returns:
            prediction: The predicted values at the query time steps.
        """
        ...


class ProbabilisticForecastingModel(Protocol):
    r"""Protocol for probabilistic forecasting models.

    A probabilistic forecasting model is a model that predicts the conditional
    distribution of the future time series, given the past observations.

    .. math::
        model: \Seq(𝓣)×\Seq(𝓣×𝓧) ⟶ \Seq(𝓟(𝓧)),
        ((t₁, …, tₙ), S, M) ⟼ (p̂(y_{t₁} | t₁, S, M), …, p̂(y_{tₙ} | tₙ, S, M))

    We distinguish between 2 types of probabilistic forecasting models:

    - fully joint-distribution models predict the joint distribution
      $p(𝐲_𝐭 | 𝐭, S, M) = p(y_{t₁}, …, y_{tₙ} | (t₁, ..., tₙ), S, M)$
    - per-time-step models predict the marginal distribution
      $p(y_{tₖ} | tₖ, S, M)$ for each time step $tₖ ∈ T$.
    """

    @abstractmethod
    def log_prob(
        self,
        samples: Tensor,  # Float[*S, ..., $K, F]
        /,
        *,
        query_times: Tensor,  # Float[..., $K], padded NaN, non-decreasing
        query_mask: Tensor,  # Bool[..., $K, F], padded False
        context_times: Tensor,  # Float[..., $N], padded NaN, non-decreasing
        context_values: Tensor,  # Float[..., $N, D], padded NaN, sparse
        context_mask: Tensor,  # Bool[..., $N, D], padded False
    ) -> Tensor:  # Float[*S, ..., $K]
        r"""Compute the log-likelihood of the samples.

        Args:
            samples: The samples to compute the log-likelihood of.
            query_times: $q = (t₁, t₂, …, tₖ)$ are the time indices we want to predict at
            query_mask: $c = (c₁, c₂, …, cₖ)$ indicate channels to be predicted at query time
            context_times: $τ = (τ₁, τ₂, …, τₙ)$ are the time indices of the observations
            context_values: $x = (x₁, x₂, …, xₙ)$ are the values of the observations
            context_mask: $m = (m₁, m₂, …, mₙ)$ indicate valid observations (at feature level)

        Returns:
            log_probs: the time-marginal log-likelihoods of the samples.
        """

    @abstractmethod
    def sample(
        self,
        size: int | tuple[int, ...] = (),  # *S
        *,
        query_times: Tensor,  # Float[..., $K], padded NaN, non-decreasing
        query_mask: Tensor,  # Bool[..., $K, F], padded False
        context_times: Tensor,  # Float[..., $N], padded NaN, non-decreasing
        context_values: Tensor,  # Float[..., $N, D], padded NaN, sparse
        context_mask: Tensor,  # Bool[..., $N, D], padded False
        rng: Generator | None = None,
    ) -> Tensor:  # (*S, ..., $K, F)
        r"""Sample from the predictive distribution of the model.

        Args:
            size: The number of samples to draw from the predictive distribution.
            query_times: $q = (t₁, t₂, …, tₖ)$ are the time indices we want to predict at
            query_mask: $c = (c₁, c₂, …, cₖ)$ indicate channels to be predicted at query time
            context_times: $τ = (τ₁, τ₂, …, τₙ)$ are the time indices of the observations
            context_values: $x = (x₁, x₂, …, xₙ)$ are the values of the observations
            context_mask: $m = (m₁, m₂, …, mₙ)$ indicate valid observations (at feature level)
            rng: The random number generator to use for sampling.

        Returns:
            samples: The sampled values from the predictive distribution.
        """

    @abstractmethod
    def sample_and_log_prob(
        self,
        size: int | tuple[int, ...] = (),  # *S
        *,
        query_times: Tensor,  # Float[..., $K], padded NaN, non-decreasing
        query_mask: Tensor,  # Bool[..., $K, F], padded False
        context_times: Tensor,  # Float[..., $N], padded NaN, non-decreasing
        context_values: Tensor,  # Float[..., $N, D], padded NaN, sparse
        context_mask: Tensor,  # Bool[..., $N, D], padded False
        rng: Generator | None = None,
    ) -> tuple[Tensor, Tensor]:  # (*S, ..., $K, F), (*S, ..., $K)
        r"""Sample from the predictive distribution of the model."""


class PathForecastingModel(Protocol):
    r"""Protocol for path-forecasting models.

    A path-forecasting model is a model that predicts the joint distribution across
    multiple future time steps.

    .. math::
            model: \Seq(𝓣)×\Seq(𝓣×𝓧) ⟶ 𝓟(\Seq(𝓧)),
            ((t₁, …, tₙ), S, M) ⟼ p̂(y_{t₁}, y_{t₂}, …, y_{tₙ} | (t₁, …, tₙ), S, M)
    """

    @abstractmethod
    def log_prob(
        self,
        values: Tensor,  # Float[*S, ..., $K, F]
        /,
        *,
        query_times: Tensor,  # Float[..., $K], padded NaN, non-decreasing
        query_mask: Tensor,  # Bool[..., $K, F], padded False
        context_times: Tensor,  # Float[..., $N], padded NaN, non-decreasing
        context_values: Tensor,  # Float[..., $N, D], padded NaN, sparse
        context_mask: Tensor,  # Bool[..., $N, D], padded False
    ) -> Tensor:  # Float[*S, ...]
        r"""Compute the log-likelihood of the samples.

        Args:
            values: The samples to compute the log-likelihood of.
            query_times: $q = (t₁, t₂, …, tₖ)$ are the time indices we want to predict at
            query_mask: $c = (c₁, c₂, …, cₖ)$ indicate channels to be predicted at query time
            context_times: $τ = (τ₁, τ₂, …, τₙ)$ are the time indices of the observations
            context_values: $x = (x₁, x₂, …, xₙ)$ are the values of the observations
            context_mask: $m = (m₁, m₂, …, mₙ)$ indicate valid observations (at feature level)

        Returns:
            log_probs: the joint log-likelihoods of the samples across all time steps.
        """

    @abstractmethod
    def sample(
        self,
        size: int | tuple[int, ...] = (),  # *S
        *,
        query_times: Tensor,  # Float[..., $K], padded NaN, non-decreasing
        query_mask: Tensor,  # Bool[..., $K, F], padded False
        context_times: Tensor,  # Float[..., $N], padded NaN, non-decreasing
        context_values: Tensor,  # Float[..., $N, D], padded NaN, sparse
        context_mask: Tensor,  # Bool[..., $N, D], padded False
        rng: Generator | None = None,
    ) -> Tensor:  # Float[*S, ..., $K, F]
        r"""Sample from the predictive distribution of the model.

        Args:
            size: The number of samples to draw from the predictive distribution.
            query_times: $q = (t₁, t₂, …, tₖ)$ are the time indices we want to predict at
            query_mask: $c = (c₁, c₂, …, cₖ)$ indicate channels to be predicted at query time
            context_times: $τ = (τ₁, τ₂, …, τₙ)$ are the time indices of the observations
            context_values: $x = (x₁, x₂, …, xₙ)$ are the values of the observations
            context_mask: $m = (m₁, m₂, …, mₙ)$ indicate valid observations (at feature level)
            rng: The random number generator to use for sampling.

        Returns:
            samples: The sampled values from the predictive distribution.
        """

    @abstractmethod
    def sample_and_log_prob(
        self,
        size: int | tuple[int, ...] = (),  # *S
        *,
        query_times: Tensor,  # Float[..., $K], padded NaN, non-decreasing
        query_mask: Tensor,  # Bool[..., $K, F], padded False
        context_times: Tensor,  # Float[..., $N], padded NaN, non-decreasing
        context_values: Tensor,  # Float[..., $N, D], padded NaN, sparse
        context_mask: Tensor,  # Bool[..., $N, D], padded False
        rng: Generator | None = None,
    ) -> tuple[Tensor, Tensor]:  # (*S, ..., $K, F), (*S, ...)
        r"""Sample from the predictive distribution of the model."""


class ProbabilisticLSSM(Protocol):
    r"""Protocol for probabilistic latent state-space models.

    Latent distribution at time t: $p(x∣θₜ)$
    Predictive distribution at time t:
        a. $q(y∣ωₜ=ϕ(θₜ))$  (decoder in parameter space)
        b. $q(y)=p(ϕ⁻¹(y)∣θₜ)|det 𝐃ϕ⁻¹(y)|$ (decoder in data space)

    State update:
        a. $ωₜ' = f(ωₜ, y_obs)$, $θₜ' = ϕ⁻¹(ωₜ')$ (update in observation space)
        b. $θₜ' = g(θₜ, ϕ, y_obs)$ (update in latent space)

    Idea:
        The theoretically correct update is a Bayesian one.
        However, this is generally intractable.
        Instead, we could do a gradient step along a variational loss.

        Loss: d_KL(p(x∣θₜ) ∥ p(x∣θₜ')) - E_{x∼p(x∣θ)}[log p(y_obs∣x)]

        y = g(x) + ε, with ε ∼ N(0, R). Then log p(y_obs ∣ x) = const - 1/2 (y_obs - g(x))ᵀ R⁻¹ (y_obs - g(x)).
    """
