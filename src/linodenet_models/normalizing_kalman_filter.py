r"""Implementation of Normalizing Kalman Filter (NKF) for time series prediction.

References:
    - | Normalizing Kalman Filters for Multivariate Time Series Analysis
      | de Bézenac et al.
      | Advances in Neural Information Processing Systems (NeurIPS) 2020
      | https://proceedings.neurips.cc/paper/2020/hash/1f47cef5e38c952f94c5d61726027439-Abstract.html
"""

__all__ = ["ContinuousTimeNKF", "DiscreteTimeNKF"]

from typing import TYPE_CHECKING, Final, cast

import torch
from numpy.typing import ArrayLike
from torch import Generator, Tensor, nan, nn

from .kalman_filter import (
    ContinuousTimeKalmanFilter,
    DiscreteTimeKalmanFilter,
    marginal_gaussian_log_prob,
    marginal_gaussian_sample,
    marginal_gaussian_sample_and_log_prob,
)

if TYPE_CHECKING:
    from linodenet.mappings import Transform


def _reduce_logabsdet(logabsdet: Tensor, mask: Tensor) -> Tensor:
    r"""Reduce a transform log-Jacobian to one scalar per masked event.

    Coordinate-wise transforms in this repository return one log-Jacobian term per
    feature. General vector transforms usually return one scalar per full vector.
    The latter cannot be marginalized for partial observations; this is exactly
    the caveat discussed in Sec. 3, "Partial observations", of the NKF paper.
    """
    if logabsdet.shape == mask.shape:
        return torch.where(mask, logabsdet, torch.zeros_like(logabsdet)).sum(dim=-1)

    if logabsdet.shape == mask.shape[:-1]:
        partial = mask.any(dim=-1) & ~mask.all(dim=-1)
        if bool(partial.any()):
            raise NotImplementedError(
                "Partial observations with a vector-valued decoder require "
                "closed-form marginalization over missing flow dimensions. Use "
                "a coordinate-wise decoder for sparse feature masks.",
            )
        return torch.where(mask.any(dim=-1), logabsdet, torch.zeros_like(logabsdet))

    raise ValueError(
        f"Expected logabsdet shape {mask.shape} or {mask.shape[:-1]}, "
        f"got {logabsdet.shape}.",
    )


class DiscreteTimeNKF(nn.Module):
    r"""Normalizing Kalman Filter (NKF) with discrete, time-invariant dynamics.

    The implemented model follows Eq. (1) in the paper:

    .. math::
        lₜ &= F lₜ₋₁ + ϵₜ,      & ϵₜ &∼ 𝓝(0, Q) \\
        zₜ &= H lₜ + ηₜ,        & ηₜ &∼ 𝓝(0, R) \\
        yₜ &= f(zₜ).

    Here ``decoder`` is the invertible map $f:z↦y$ and is assumed to satisfy the
    :class:`linodenet.mappings.Transform` protocol. Filtering first pulls
    observations back to pseudo-observations $zₜ=f⁻¹(yₜ)$ as in Proposition 1.
    Likelihoods use the change-of-variables term from Eq. (2) / Proposition 3.

    Note:
        Exact sparse feature masks require a coordinate-wise/local decoder whose
        inverse and log-Jacobian factor over dimensions. For a global flow, partial
        observations require marginalization over missing flow dimensions, which
        Sec. 3 notes is not available in closed form for the main RealNVP NKF.
    """

    input_size: Final[int]
    hidden_size: Final[int]
    batch_first: Final[bool]

    decoder: nn.Module
    kalman: DiscreteTimeKalmanFilter

    pred_means: Tensor
    pred_scales: Tensor

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        *,
        decoder: nn.Module,
        system_matrix: ArrayLike | None = None,  # [n, n]
        observation_matrix: ArrayLike | None = None,  # [k, n]
        process_covariance: ArrayLike | float | None = 0.1,  # [n, n]
        measurement_covariance: ArrayLike | float | None = 1.0,  # [k, k]
        initial_mean: ArrayLike | float | None = 0.0,  # [n]
        initial_covariance: ArrayLike | float | None = 1.0,  # [n, n]
        use_cholesky: bool = False,
        learnable: bool = False,
        batch_first: bool = True,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_first = batch_first
        self.decoder = decoder
        self.kalman = DiscreteTimeKalmanFilter(
            input_size,
            hidden_size,
            system_matrix=system_matrix,
            observation_matrix=observation_matrix,
            process_covariance=process_covariance,
            measurement_covariance=measurement_covariance,
            initial_mean=initial_mean,
            initial_covariance=initial_covariance,
            use_cholesky=use_cholesky,
            learnable=learnable,
            batch_first=batch_first,
        )

        self.register_buffer("pred_means", None, persistent=False)
        self.register_buffer("pred_scales", None, persistent=False)

    def _decode_observations(
        self,
        values: Tensor,
        mask: Tensor,
        /,
    ) -> tuple[Tensor, Tensor]:
        r"""Compute $z=f⁻¹(y)$ and $\log|\det ∂f⁻¹/∂y|$ from Eq. (2)."""
        dense_values = values.masked_fill(~mask, 0.0)
        decoder = cast("Transform", self.decoder)
        pseudo_values, inverse_logabsdet = decoder.decode_and_logabsdet(
            dense_values,
        )
        logabsdet = _reduce_logabsdet(inverse_logabsdet, mask)
        return pseudo_values.masked_fill(~mask, nan), logabsdet

    def _encode_observations(
        self,
        values: Tensor,
        mask: Tensor,
        /,
    ) -> tuple[Tensor, Tensor]:
        r"""Compute $y=f(z)$ and $\log|\det ∂f/∂z|$ from Eq. (1c)."""
        dense_values = values.masked_fill(~mask, 0.0)
        decoder = cast("Transform", self.decoder)
        observations, forward_logabsdet = decoder.encode_and_logabsdet(
            dense_values,
        )
        logabsdet = _reduce_logabsdet(forward_logabsdet, mask)
        return observations.masked_fill(~mask, nan), logabsdet

    def forward(
        self,
        *,
        steps: Tensor,  # Long[..., $T], padded arbitrary, non-decreasing
        query_mask: Tensor,  # (..., $T, D), bool, padded False
        context_values: Tensor,  # (..., $T, D), float, padded NaN, sparse
        context_mask: Tensor,  # (..., $T, D), bool, padded False
        # μ₀=(..., D) Σ₀=(..., D, D)
        initial_state: tuple[Tensor, Tensor] | None = None,
        initial_step: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:  # (..., $T, D), (..., $T, D, D)
        r"""Compute posterior latent states given combined context/query steps.

        This is Proposition 1: filtering in observation space is equivalent to
        ordinary Kalman filtering with pseudo-observations $zₜ=f⁻¹(yₜ)$.
        """
        pseudo_values, _ = self._decode_observations(context_values, context_mask)
        return self.kalman.forward(
            steps=steps,
            context_values=pseudo_values,
            context_mask=context_mask,
            query_mask=query_mask,
            initial_state=initial_state,
            initial_step=initial_step,
        )

    def predict(
        self,
        *,
        query_steps: Tensor,  # Long[..., K], padded arbitrary, non-decreasing
        query_mask: Tensor,  # Bool[..., K, D], padded False
        context_steps: Tensor,  # Long[..., N], padded arbitrary, non-decreasing
        context_mask: Tensor,  # Bool[..., N, D], padded False
        context_values: Tensor,  # Float[..., N, D], padded NaN, sparse
        # μ₀=(..., D) Σ₀=(..., D, D)
        initial_state: tuple[Tensor, Tensor] | None = None,
        initial_step: Tensor | None = None,  # t₀, ()
    ) -> tuple[Tensor, Tensor]:  # (..., K, D), (..., K, D)
        r"""Return a point summary of the transformed predictive distribution.

        Forecasting follows Appendix A.4 in distribution. Since $f(Z)$ is not
        generally Gaussian, this method reports ``f(E[Z])`` and the marginal
        standard deviation of the pseudo-Gaussian $Z$. Use :meth:`log_prob` and
        :meth:`sample` for exact distributional operations.
        """
        pseudo_values, _ = self._decode_observations(context_values, context_mask)
        mean, cov = self.kalman.predict(
            query_steps=query_steps,
            query_mask=query_mask,
            context_steps=context_steps,
            context_values=pseudo_values,
            context_mask=context_mask,
            initial_state=initial_state,
            initial_step=initial_step,
        )
        self.pred_means, _ = self._encode_observations(mean, query_mask)
        self.pred_scales = (
            cov.diagonal(dim1=-2, dim2=-1)
            .clamp_min(0.0)
            .sqrt()
            .masked_fill(
                ~query_mask,
                nan,
            )
        )
        return self.pred_means, self.pred_scales

    def log_prob(
        self,
        values: Tensor,  # (..., K, D)
        *,
        query_steps: Tensor,  # Long[..., K], padded arbitrary, non-decreasing
        query_mask: Tensor,  # Bool[..., K, D], padded False
        context_steps: Tensor,  # Long[..., N], padded arbitrary, non-decreasing
        context_values: Tensor,  # Float[..., N, D], padded NaN, sparse
        context_mask: Tensor,  # Bool[..., N, D], padded False
        initial_state: tuple[Tensor, Tensor] | None = None,
        initial_step: Tensor | None = None,
    ) -> Tensor:  # (..., K)
        r"""Compute exact time-marginal log-likelihoods via Proposition 3."""
        pseudo_context, _ = self._decode_observations(context_values, context_mask)
        mean, cov = self.kalman.predict(
            query_steps=query_steps,
            query_mask=query_mask,
            context_steps=context_steps,
            context_values=pseudo_context,
            context_mask=context_mask,
            initial_state=initial_state,
            initial_step=initial_step,
        )

        mask = query_mask.expand(*values.shape)
        pseudo_values, inverse_logabsdet = self._decode_observations(values, mask)
        base_log_prob = marginal_gaussian_log_prob(
            pseudo_values,
            mean=mean.expand(*pseudo_values.shape),
            cov=cov.expand(*pseudo_values.shape[:-1], *cov.shape[-2:]),
            mask=mask,
        )
        return base_log_prob + inverse_logabsdet

    def sample(
        self,
        size: int | tuple[int, ...] = (),  # *S
        *,
        query_steps: Tensor,  # Long[..., K], padded arbitrary, non-decreasing
        query_mask: Tensor,  # Bool[..., K, D], padded False
        context_steps: Tensor,  # Long[..., N], padded arbitrary, non-decreasing
        context_values: Tensor,  # Float[..., N, D], padded NaN, sparse
        context_mask: Tensor,  # Bool[..., N, D], padded False
        initial_state: tuple[Tensor, Tensor] | None = None,
        initial_step: Tensor | None = None,
        rng: Generator | None = None,
    ) -> Tensor:  # (*S, ..., K, D)
        r"""Sample from the transformed time-marginal predictive distribution."""
        pseudo_context, _ = self._decode_observations(context_values, context_mask)
        mean, cov = self.kalman.predict(
            query_steps=query_steps,
            query_mask=query_mask,
            context_steps=context_steps,
            context_values=pseudo_context,
            context_mask=context_mask,
            initial_state=initial_state,
            initial_step=initial_step,
        )
        pseudo_samples = marginal_gaussian_sample(
            size,
            mean=mean,
            cov=cov,
            mask=query_mask,
            rng=rng,
        )
        sample_shape = (size,) if isinstance(size, int) else size
        mask = query_mask.expand(*sample_shape, *query_mask.shape)
        samples, _ = self._encode_observations(pseudo_samples, mask)
        return samples

    def sample_and_log_prob(
        self,
        size: int | tuple[int, ...] = (),  # *S
        *,
        query_steps: Tensor,  # Long[..., K], padded arbitrary, non-decreasing
        query_mask: Tensor,  # Bool[..., K, D], padded False
        context_steps: Tensor,  # Long[..., N], padded arbitrary, non-decreasing
        context_values: Tensor,  # Float[..., N, D], padded NaN, sparse
        context_mask: Tensor,  # Bool[..., N, D], padded False
        initial_state: tuple[Tensor, Tensor] | None = None,
        initial_step: Tensor | None = None,
        rng: Generator | None = None,
    ) -> tuple[Tensor, Tensor]:  # (*S, ..., K, D), (*S, ..., K)
        r"""Sample and score via Eq. (1c) and the inverse of Prop. 3."""
        pseudo_context, _ = self._decode_observations(context_values, context_mask)
        mean, cov = self.kalman.predict(
            query_steps=query_steps,
            query_mask=query_mask,
            context_steps=context_steps,
            context_values=pseudo_context,
            context_mask=context_mask,
            initial_state=initial_state,
            initial_step=initial_step,
        )
        pseudo_samples, base_log_prob = marginal_gaussian_sample_and_log_prob(
            size,
            mean=mean,
            cov=cov,
            mask=query_mask,
            rng=rng,
        )
        sample_shape = (size,) if isinstance(size, int) else size
        mask = query_mask.expand(*sample_shape, *query_mask.shape)
        samples, forward_logabsdet = self._encode_observations(pseudo_samples, mask)
        return samples, base_log_prob - forward_logabsdet


class ContinuousTimeNKF(nn.Module):
    r"""Normalizing Kalman Filter with continuous, time-invariant dynamics.

    The model keeps the paper's observation equation (Eq. (1c)),
    $yₜ=f(Hlₜ+ηₜ)$, but uses the continuous latent dynamics implemented by
    :class:`ContinuousTimeKalmanFilter`. Filtering still follows Proposition 1:
    observations are pulled back to pseudo-observations $zₜ=f⁻¹(yₜ)$ before
    applying ordinary Kalman updates. Likelihoods use the exact
    change-of-variables term from Eq. (2) / Proposition 3.

    Note:
        As for :class:`DiscreteTimeNKF`, exact partial feature masks require a
        coordinate-wise/local decoder. A global flow needs closed-form
        marginalization over missing output dimensions, which the paper notes is
        not available for the main global RealNVP NKF instantiation.
    """

    input_size: Final[int]
    hidden_size: Final[int]
    batch_first: Final[bool]

    decoder: nn.Module
    kalman: ContinuousTimeKalmanFilter

    pred_means: Tensor
    pred_scales: Tensor

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        *,
        decoder: nn.Module,
        system_matrix: ArrayLike | None = None,  # [n, n]
        observation_matrix: ArrayLike | None = None,  # [k, n]
        process_noise: ArrayLike | float = 0.1,  # [n, n]
        measurement_noise: ArrayLike | float = 1.0,  # [k, k]
        initial_mean: ArrayLike | float = 0.0,  # [n]
        initial_covariance: ArrayLike | float = 1.0,  # [n, n]
        use_cholesky: bool = False,
        initial_state_learnable: bool = True,
        process_noise_learnable: bool = False,
        observation_noise_learnable: bool = False,
        batch_first: bool = True,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_first = batch_first
        self.decoder = decoder
        self.kalman = ContinuousTimeKalmanFilter(
            input_size,
            hidden_size,
            system_matrix=system_matrix,
            observation_matrix=observation_matrix,
            process_noise=process_noise,
            measurement_noise=measurement_noise,
            initial_mean=initial_mean,
            initial_covariance=initial_covariance,
            use_cholesky=use_cholesky,
            initial_state_learnable=initial_state_learnable,
            process_noise_learnable=process_noise_learnable,
            observation_noise_learnable=observation_noise_learnable,
            batch_first=batch_first,
        )

        self.register_buffer("pred_means", None, persistent=False)
        self.register_buffer("pred_scales", None, persistent=False)

    def _decode_observations(
        self,
        values: Tensor,
        mask: Tensor,
        /,
    ) -> tuple[Tensor, Tensor]:
        r"""Compute $z=f⁻¹(y)$ and $\log|\det ∂f⁻¹/∂y|$ from Eq. (2)."""
        dense_values = values.masked_fill(~mask, 0.0)
        decoder = cast("Transform", self.decoder)
        pseudo_values, inverse_logabsdet = decoder.decode_and_logabsdet(
            dense_values,
        )
        logabsdet = _reduce_logabsdet(inverse_logabsdet, mask)
        return pseudo_values.masked_fill(~mask, nan), logabsdet

    def _encode_observations(
        self,
        values: Tensor,
        mask: Tensor,
        /,
    ) -> tuple[Tensor, Tensor]:
        r"""Compute $y=f(z)$ and $\log|\det ∂f/∂z|$ from Eq. (1c)."""
        dense_values = values.masked_fill(~mask, 0.0)
        decoder = cast("Transform", self.decoder)
        observations, forward_logabsdet = decoder.encode_and_logabsdet(
            dense_values,
        )
        logabsdet = _reduce_logabsdet(forward_logabsdet, mask)
        return observations.masked_fill(~mask, nan), logabsdet

    def forward(
        self,
        *,
        timestamps: Tensor,  # (..., T), float, padded NaN
        query_mask: Tensor,  # (..., T, D), bool, padded False
        context_values: Tensor,  # (..., T, D), float, padded NaN, sparse
        context_mask: Tensor,  # (..., T, D), bool, padded False
        initial_state: tuple[Tensor, Tensor] | None = None,
        initial_time: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:  # (..., T, D), (..., T, D, D)
        r"""Compute posterior latent states over combined context/query times."""
        pseudo_values, _ = self._decode_observations(context_values, context_mask)
        return self.kalman.forward(
            timestamps=timestamps,
            context_values=pseudo_values,
            context_mask=context_mask,
            query_mask=query_mask,
            initial_state=initial_state,
            initial_time=initial_time,
        )

    def predict(
        self,
        *,
        query_times: Tensor,  # Float[..., K], padded NaN, strictly increasing
        query_mask: Tensor,  # Bool[..., K, D], padded False
        context_times: Tensor,  # Float[..., N], padded NaN, non-decreasing
        context_mask: Tensor,  # Bool[..., N, D], padded False
        context_values: Tensor,  # Float[..., N, D], padded NaN, sparse
        initial_state: tuple[Tensor, Tensor] | None = None,
        initial_time: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:  # (..., K, D), (..., K, D)
        r"""Return a point summary of the transformed predictive distribution."""
        pseudo_values, _ = self._decode_observations(context_values, context_mask)
        mean, cov = self.kalman.predict(
            query_times=query_times,
            query_mask=query_mask,
            context_times=context_times,
            context_values=pseudo_values,
            context_mask=context_mask,
            initial_state=initial_state,
            initial_time=initial_time,
        )
        self.pred_means, _ = self._encode_observations(mean, query_mask)
        self.pred_scales = (
            cov.diagonal(dim1=-2, dim2=-1)
            .clamp_min(0.0)
            .sqrt()
            .masked_fill(
                ~query_mask,
                nan,
            )
        )
        return self.pred_means, self.pred_scales

    def log_prob(
        self,
        values: Tensor,  # (..., K, D)
        *,
        query_times: Tensor,  # Float[..., K], padded NaN, strictly increasing
        query_mask: Tensor,  # Bool[..., K, D], padded False
        context_times: Tensor,  # Float[..., N], padded NaN, non-decreasing
        context_values: Tensor,  # Float[..., N, D], padded NaN, sparse
        context_mask: Tensor,  # Bool[..., N, D], padded False
        initial_state: tuple[Tensor, Tensor] | None = None,
        initial_time: Tensor | None = None,
    ) -> Tensor:  # (..., K)
        r"""Compute exact time-marginal log-likelihoods via Proposition 3."""
        pseudo_context, _ = self._decode_observations(context_values, context_mask)
        mean, cov = self.kalman.predict(
            query_times=query_times,
            query_mask=query_mask,
            context_times=context_times,
            context_values=pseudo_context,
            context_mask=context_mask,
            initial_state=initial_state,
            initial_time=initial_time,
        )

        mask = query_mask.expand(*values.shape)
        pseudo_values, inverse_logabsdet = self._decode_observations(values, mask)
        base_log_prob = marginal_gaussian_log_prob(
            pseudo_values,
            mean=mean.expand(*pseudo_values.shape),
            cov=cov.expand(*pseudo_values.shape[:-1], *cov.shape[-2:]),
            mask=mask,
        )
        return base_log_prob + inverse_logabsdet

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
        initial_time: Tensor | None = None,
        rng: Generator | None = None,
    ) -> Tensor:  # (*S, ..., K, D)
        r"""Sample from the transformed time-marginal predictive distribution."""
        pseudo_context, _ = self._decode_observations(context_values, context_mask)
        mean, cov = self.kalman.predict(
            query_times=query_times,
            query_mask=query_mask,
            context_times=context_times,
            context_values=pseudo_context,
            context_mask=context_mask,
            initial_state=initial_state,
            initial_time=initial_time,
        )
        pseudo_samples = marginal_gaussian_sample(
            size,
            mean=mean,
            cov=cov,
            mask=query_mask,
            rng=rng,
        )
        sample_shape = (size,) if isinstance(size, int) else size
        mask = query_mask.expand(*sample_shape, *query_mask.shape)
        samples, _ = self._encode_observations(pseudo_samples, mask)
        return samples

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
        initial_time: Tensor | None = None,
        rng: Generator | None = None,
    ) -> tuple[Tensor, Tensor]:  # (*S, ..., K, D), (*S, ..., K)
        r"""Sample and score via Eq. (1c) and the inverse of Prop. 3."""
        pseudo_context, _ = self._decode_observations(context_values, context_mask)
        mean, cov = self.kalman.predict(
            query_times=query_times,
            query_mask=query_mask,
            context_times=context_times,
            context_values=pseudo_context,
            context_mask=context_mask,
            initial_state=initial_state,
            initial_time=initial_time,
        )
        pseudo_samples, base_log_prob = marginal_gaussian_sample_and_log_prob(
            size,
            mean=mean,
            cov=cov,
            mask=query_mask,
            rng=rng,
        )
        sample_shape = (size,) if isinstance(size, int) else size
        mask = query_mask.expand(*sample_shape, *query_mask.shape)
        samples, forward_logabsdet = self._encode_observations(pseudo_samples, mask)
        return samples, base_log_prob - forward_logabsdet
