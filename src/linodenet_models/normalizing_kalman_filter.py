"""Implementation of Normalizing Kalman Filter (NKF) for time series prediction.

References:
    - | Normalizing Kalman Filters for Multivariate Time Series Analysis
      | de Bézenac et al.
      | Advances in Neural Information Processing Systems (NeurIPS) 2020
      | https://proceedings.neurips.cc/paper/2020/hash/1f47cef5e38c952f94c5d61726027439-Abstract.html
"""

__all__ = ["NormalizingKalmanFilter"]

from torch import Tensor, nn


class NormalizingKalmanFilter(nn.Module):
    r"""Normalizing Kalman Filter (NKF) for time series prediction.

    zₜ ∼ 𝓝(μₜ, Σₜ)
    zₜ follows linear dynamics.
    y = ϕ(Hz + ε), for normalizing flow ϕ
    """

    def forward(
        self,
        *,
        timestamps: Tensor,  # (..., $T), float, padded NaN
        query_mask: Tensor,  # (..., $T, D), bool, padded False
        context_values: Tensor,  # (..., $T, D), float, padded Nan, sparse
        context_mask: Tensor,  # (..., $T, D), bool, padded False
        # μ₀=(..., D) Σ₀=(..., D, D)
        initial_state: tuple[Tensor, Tensor] | None = None,
        initial_time: Tensor | None = None,  # t₀, ()
    ) -> tuple[Tensor, Tensor]:  # (..., $T, D), (..., $T, D, D)
        r"""Compute the posterior latent states, given combined context/query time points."""
        raise NotImplementedError

    def predict(
        self,
        *,
        query_times: Tensor,  # Float[..., K], padded NaN, strictly increasing
        query_mask: Tensor,  # Bool[..., K, F]  padded False
        context_times: Tensor,  # Float[..., N], padded NaN, non-decreasing
        context_mask: Tensor,  # Bool[..., N, D], padded False
        context_values: Tensor,  # Float[..., N, D], padded NaN, sparse
        # μ₀=(..., D) Σ₀=(..., D, D)
        initial_state: tuple[Tensor, Tensor] | None = None,
        initial_time: Tensor | None = None,  # t₀, ()
    ) -> tuple[Tensor, Tensor]:  # (..., $K, D), (..., $K, D)
        r"""Compute the posterior latent states, given split context/query time points."""
        raise NotImplementedError

    def log_prob(
        self,
        values: Tensor,  # (..., $K, D)
        *,
        query_times: Tensor,  # Float[..., K], padded NaN, strictly increasing
        query_mask: Tensor,  # Bool[..., K, D], padded False
        context_times: Tensor,  # Float[..., N), padded NaN, non-decreasing
        context_values: Tensor,  # Float[..., N, D], padded NaN, sparse
        context_mask: Tensor,  # Bool[..., N, D], padded False
        initial_state: tuple[Tensor, Tensor] | None = None,
        initial_time: Tensor | None = None,  # t₀, ()
    ) -> Tensor:  # (..., $K)
        raise NotImplementedError

    def sample(
        self,
        size: int | tuple[int, ...] = (),  # *S
        *,
        query_times: Tensor,  # Float[..., K], padded NaN, strictly increasing
        query_mask: Tensor,  # Bool[..., K, D], padded False
        context_times: Tensor,  # Float[..., N], padded NaN, non-decreasing
        context_values: Tensor,  # Float[..., N, D], padded NaN, sparse
        context_mask: Tensor,  # Bool[..., N, D], padded False
        initial_state: tuple[Tensor, Tensor] | None = None,
        initial_time: Tensor | None = None,  # t₀, ()
    ) -> Tensor:  # (*S, ..., $K, D)
        raise NotImplementedError

    def sample_and_log_prob(
        self,
        size: int | tuple[int, ...] = (),  # *S
        *,
        query_times: Tensor,  # Float[..., K], padded NaN, strictly increasing
        query_mask: Tensor,  # Bool[..., K, D], padded False
        context_times: Tensor,  # Float[..., N], padded NaN, non-decreasing
        context_values: Tensor,  # Float[..., N, D], padded NaN, sparse
        context_mask: Tensor,  # Bool[..., N, D], padded False
        initial_state: tuple[Tensor, Tensor] | None = None,
        initial_time: Tensor | None = None,  # t₀, ()
    ) -> tuple[Tensor, Tensor]:  # (*S, ..., $K, D), (*S, ..., $K)
        raise NotImplementedError
