r"""Protocols for point, probabilistic, and path forecasting models."""

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

    $$ model(t, S) = ŷ(t, S) $$

    The inputs to the model are:

    - Query $q = (t₁, t₂, …, tₙ)$ contains the time indices to predict at.
    - Context $S = ((τ₁, x₁), (τ₂, x₂), …, (τₘ, xₘ))$ contains observations and covariates.

    The general forecasting model must allow an arbitrary number of observations and
    target time steps.
    """

    @abstractmethod
    def __call__(self, query: Tensor, context: Tensor, /) -> Tensor:
        r"""Forward pass of the model.

        $$\begin{aligned}
            model &: \Seq(𝓣)×\Seq(𝓣×𝓧) ⟶ \Seq(𝓧) \\
            ((t₁, …, tₙ), S) &⟼ (ŷₜ₁, ŷₜ₂, …, ŷₜₙ)
        \end{aligned}$$

        Args:
            query: Time indices $q = (t₁, t₂, …, tₙ)$ to predict at.
            context: Observations and covariates $S = ((τ₁, x₁), (τ₂, x₂), …, (τₘ, xₘ))$.

        Returns:
            prediction: The predicted values at the query time steps.
        """
        ...


class ProbabilisticForecastingModel(Protocol):
    r"""Protocol for probabilistic forecasting models.

    A probabilistic forecasting model is a model that predicts the conditional
    distribution of the future time series, given the past observations.

    $$\begin{aligned}
        model &: \Seq(𝓣)×\Seq(𝓣×𝓧) ⟶ \Seq(𝓟(𝓧)) \\
        ((t₁, …, tₙ), S) &⟼ (p̂(yₜ₁ ∣ t₁, S), …, p̂(yₜₙ ∣ tₙ, S))
    \end{aligned}$$

    Here $S$ includes observed values and their feature masks.

    We distinguish between 2 types of probabilistic forecasting models:

    - Joint-distribution models predict $p(yₜ₁, …, yₜₙ ∣ (t₁, …, tₙ), S)$.
    - Per-time-step models predict $p(yₜₖ ∣ tₖ, S)$ for each query time $tₖ$.
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
            query_times: Time indices $q = (t₁, t₂, …, tₖ)$ to predict at.
            query_mask: Feature masks $c = (c₁, c₂, …, cₖ)$ selecting channels to predict at each query time.
            context_times: Observation times $τ = (τ₁, τ₂, …, τₙ)$.
            context_values: Observed values $x = (x₁, x₂, …, xₙ)$.
            context_mask: Feature masks $m = (m₁, m₂, …, mₙ)$ selecting valid observations.

        Returns:
            log_probs: The time-marginal log-likelihoods of the samples.
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
            query_times: Time indices $q = (t₁, t₂, …, tₖ)$ to predict at.
            query_mask: Feature masks $c = (c₁, c₂, …, cₖ)$ selecting channels to predict at each query time.
            context_times: Observation times $τ = (τ₁, τ₂, …, τₙ)$.
            context_values: Observed values $x = (x₁, x₂, …, xₙ)$.
            context_mask: Feature masks $m = (m₁, m₂, …, mₙ)$ selecting valid observations.
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

    $$\begin{aligned}
        model &: \Seq(𝓣)×\Seq(𝓣×𝓧) ⟶ 𝓟(\Seq(𝓧)) \\
        ((t₁, …, tₙ), S) &⟼ p̂(yₜ₁, yₜ₂, …, yₜₙ ∣ (t₁, …, tₙ), S)
    \end{aligned}$$

    Here $S$ includes observed values and their feature masks.
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
            query_times: Time indices $q = (t₁, t₂, …, tₖ)$ to predict at.
            query_mask: Feature masks $c = (c₁, c₂, …, cₖ)$ selecting channels to predict at each query time.
            context_times: Observation times $τ = (τ₁, τ₂, …, τₙ)$.
            context_values: Observed values $x = (x₁, x₂, …, xₙ)$.
            context_mask: Feature masks $m = (m₁, m₂, …, mₙ)$ selecting valid observations.

        Returns:
            log_probs: The joint log-likelihoods of the samples across all time steps.
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
            query_times: Time indices $q = (t₁, t₂, …, tₖ)$ to predict at.
            query_mask: Feature masks $c = (c₁, c₂, …, cₖ)$ selecting channels to predict at each query time.
            context_times: Observation times $τ = (τ₁, τ₂, …, τₙ)$.
            context_values: Observed values $x = (x₁, x₂, …, xₙ)$.
            context_mask: Feature masks $m = (m₁, m₂, …, mₙ)$ selecting valid observations.
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

    The latent distribution at time $t$ is $p(x∣θₜ)$. Two possible decoders are:

    1. $q(y∣ωₜ)$ with $ωₜ=ϕ(θₜ)$, decoding in parameter space.
    2. $q(y)=p(ϕ⁻¹(y)∣θₜ)|\det 𝐃ϕ⁻¹(y)|$, decoding through an invertible data map.

    The corresponding state updates can operate in observation or latent space:

    1. $ωₜ' = f(ωₜ, yₒᵦₛ)$, followed by $θₜ' = ϕ⁻¹(ωₜ')$.
    2. $θₜ' = g(θₜ, ϕ, yₒᵦₛ)$.

    Exact Bayesian updates are generally intractable. A possible approximation
    is a gradient step on a variational loss:

    $$ ℒ = Dₖₗ(p(x∣θₜ) ∥ p(x∣θₜ')) - 𝐄_{x∼p(x∣θₜ)}[\log p(yₒᵦₛ∣x)] $$

    For $y = g(x) + ε$ with $ε ∼ 𝓝(0, R)$, the observation log-density is

    $$ \log p(yₒᵦₛ ∣ x) = \mathrm{const} - ½(yₒᵦₛ - g(x))ᵀR⁻¹(yₒᵦₛ - g(x)) $$
    """
