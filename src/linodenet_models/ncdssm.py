r"""Implementation of Neural Continuous Discrete State Space Model (NCDSSM)."""

__all__ = ["NCDSSM"]


from torch import Generator, Tensor, nn


class NCDSSM(nn.Module):
    r"""Neural Continuous Discrete State Space Model.

    References:
        - | Neural Continuous-Discrete State Space Models for Irregularly-Sampled Time Series
          | Ansari et al. (2020)
          | Proceedings of the 40th International Conference on Machine Learning (ICML 2023)
          | https://proceedings.mlr.press/v202/ansari23a.html
    """

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
        raise NotImplementedError

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
        raise NotImplementedError

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
        raise NotImplementedError
