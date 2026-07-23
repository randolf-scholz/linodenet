r"""Discrete Kalman Filter implementation."""

__all__ = [
    "ContinuousTimeKalmanFilter",
    "DiscreteTimeKalmanFilter",
    "marginal_gaussian_log_prob",
    "marginal_gaussian_sample",
    "marginal_gaussian_sample_and_log_prob",
]

import math
from typing import Final

import torch
from numpy.typing import ArrayLike
from torch import Generator, Tensor, einsum, nan, nn, stack
from torch.linalg import matrix_exp

from .utils import DiscreteTimeEventBatch, EventBatch

_LOG2PI = math.log(2.0 * math.pi)


def _log_spd(covariance: Tensor) -> Tensor:
    r"""Return the symmetric matrix logarithm of a positive-definite matrix."""
    covariance = (covariance + covariance.mT) / 2
    eigvals, eigvecs = torch.linalg.eigh(covariance)
    assert eigvals.min() > 0, "Covariance matrices must be positive definite."
    return (eigvecs * eigvals.log().unsqueeze(dim=-2)) @ eigvecs.mT


def _spd_expm(param: Tensor) -> Tensor:
    r"""Map an unconstrained square matrix to a positive-definite matrix."""
    sym = (param + param.mT) / 2
    return matrix_exp(sym)


def _as_mean(mean: ArrayLike | float, size: int) -> Tensor:
    r"""Convert scalar, vector, or ``None`` input to a mean vector."""
    value = torch.as_tensor(mean, dtype=torch.get_default_dtype())
    if value.shape == ():
        return value * torch.ones(size)
    if value.shape == (size,):
        return value
    raise ValueError(f"Invalid mean shape: {value.shape}")


def _as_covariance(covariance: ArrayLike | float, size: int) -> Tensor:
    r"""Convert scalar, matrix, or ``None`` input to a covariance matrix."""
    value = torch.as_tensor(covariance, dtype=torch.get_default_dtype())
    if value.shape == ():
        return value * torch.eye(size)
    if value.shape == (size,):
        return value.diag_embed()
    if value.shape == (size, size):
        return value
    raise ValueError(f"Invalid covariance shape: {value.shape}")


def _masked_covariance(covariance: Tensor, mask: Tensor) -> Tensor:
    r"""Restrict a dense covariance to the masked subspace."""
    active = mask.unsqueeze(-1) & mask.unsqueeze(-2)
    covariance = torch.where(active, covariance, torch.zeros_like(covariance))
    covariance = covariance + (~mask).to(covariance.dtype).diag_embed()
    return (covariance + covariance.mT) / 2


def marginal_gaussian_log_prob(
    values: Tensor,
    *,
    mean: Tensor,
    cov: Tensor,
    mask: Tensor,
) -> Tensor:
    r"""Compute log-likelihoods of masked Gaussian marginals.

    Args:
        values: Dense values with shape `(..., D)`.
        mean: Dense Gaussian means with shape `(..., D)`.
        cov: Dense Gaussian covariance matrices with shape `(..., D, D)`.
        mask: Boolean mask selecting the marginal dimensions to score, with
            shape `(..., D)`.

    Returns:
        Log-likelihoods with shape `(...)`. Rows with no selected dimensions
        have log-likelihood zero.
    """
    dim = values.shape[-1]
    assert values.shape == mean.shape
    assert cov.shape == (*values.shape, dim)
    assert mask.shape == values.shape
    assert mask.dtype == torch.bool

    residual = torch.where(mask, values - mean, torch.zeros_like(values))
    covariance = _masked_covariance(cov, mask)
    scale_tril = torch.linalg.cholesky(covariance)
    alpha = torch.cholesky_solve(residual.unsqueeze(-1), scale_tril).squeeze(-1)
    quadratic = (residual * alpha).sum(dim=-1)
    logdet = 2.0 * scale_tril.diagonal(dim1=-2, dim2=-1).log().sum(dim=-1)
    num_dims = mask.sum(dim=-1).to(values.dtype)
    return -0.5 * (quadratic + logdet + num_dims * _LOG2PI)


def marginal_gaussian_sample(
    size: int | tuple[int, ...] = (),
    *,
    mean: Tensor,
    cov: Tensor,
    mask: Tensor,
    rng: Generator | None = None,
) -> Tensor:
    r"""Sample from masked Gaussian marginals.

    Args:
        size: Sample shape.
        mean: Dense Gaussian means with shape `(..., D)`.
        cov: Dense Gaussian covariance matrices with shape `(..., D, D)`.
        mask: Boolean mask selecting the marginal dimensions to sample, with
            shape `(..., D)`.
        rng: Optional random number generator to use for sampling.

    Returns:
        Samples with shape `(*size, ..., D)`, with unselected dimensions filled
        with NaN.
    """
    dim = mean.shape[-1]
    assert cov.shape == (*mean.shape, dim)
    assert mask.shape == mean.shape
    assert mask.dtype == torch.bool

    sample_shape = (size,) if isinstance(size, int) else size
    covariance = _masked_covariance(cov, mask)
    scale_tril = torch.linalg.cholesky(covariance)
    noise = torch.randn(
        (*sample_shape, *mean.shape),
        dtype=mean.dtype,
        device=mean.device,
        generator=rng,
    )
    samples = torch.where(mask, mean, torch.zeros_like(mean)).expand(
        *sample_shape,
        *mean.shape,
    ) + (
        scale_tril.expand(*sample_shape, *scale_tril.shape) @ noise.unsqueeze(-1)
    ).squeeze(-1)
    return samples.masked_fill(~mask.expand(*sample_shape, *mask.shape), nan)


def marginal_gaussian_sample_and_log_prob(
    size: int | tuple[int, ...] = (),
    *,
    mean: Tensor,
    cov: Tensor,
    mask: Tensor,
    rng: Generator | None = None,
) -> tuple[Tensor, Tensor]:
    r"""Sample from masked Gaussian marginals and compute log-likelihoods."""
    dim = mean.shape[-1]
    assert cov.shape == (*mean.shape, dim)
    assert mask.shape == mean.shape
    assert mask.dtype == torch.bool

    sample_shape = (size,) if isinstance(size, int) else size
    covariance = _masked_covariance(cov, mask)
    scale_tril = torch.linalg.cholesky(covariance)
    noise = torch.randn(
        (*sample_shape, *mean.shape),
        dtype=mean.dtype,
        device=mean.device,
        generator=rng,
    )
    samples = torch.where(mask, mean, torch.zeros_like(mean)).expand(
        *sample_shape,
        *mean.shape,
    ) + (
        scale_tril.expand(*sample_shape, *scale_tril.shape) @ noise.unsqueeze(-1)
    ).squeeze(-1)
    samples = samples.masked_fill(~mask.expand(*sample_shape, *mask.shape), nan)

    residual = torch.where(
        mask.expand(*sample_shape, *mask.shape),
        samples - mean.expand(*sample_shape, *mean.shape),
        torch.zeros_like(samples),
    )
    alpha = torch.cholesky_solve(
        residual.unsqueeze(-1),
        scale_tril.expand(*sample_shape, *scale_tril.shape),
    ).squeeze(-1)
    quadratic = (residual * alpha).sum(dim=-1)
    logdet = 2.0 * scale_tril.diagonal(dim1=-2, dim2=-1).log().sum(dim=-1)
    num_dims = mask.sum(dim=-1).to(mean.dtype)
    log_prob = -0.5 * (
        quadratic
        + logdet.expand(*sample_shape, *logdet.shape)
        + num_dims.expand(*sample_shape, *num_dims.shape) * _LOG2PI
    )
    return samples, log_prob


class ContinuousTimeKalmanFilter(nn.Module):
    r"""Continuous time, time-invariant Kalman Filter.

    .. math::
        ∂ₜxₜ &= Fxₜ + wₜ  &  wₜ &~ N(0, Qₜ)  \\
          yₜ &= Hxₜ + vₜ  &  vₜ &~ N(0, Rₜ)

    ------------------ older docstring content ---------------
        .. math::
        x̂ₜ₊₁ &= x̂ₜ + Pₜ Hₜᵀ(Hₜ Pₜ   Hₜᵀ + Rₜ)⁻¹ (yₜ - Hₜ x̂ₜ) \\
        Pₜ₊₁ &= Pₜ - Pₜ Hₜᵀ(Hₜ Pₜ⁻¹ Hₜᵀ + Rₜ)⁻¹ Hₜ Pₜ⁻¹

    In the case of missing data:

    Substitute $yₜ← Sₜ⋅yₜ$, $Hₜ ← Sₜ⋅Hₜ$ and $Rₜ ← Sₜ⋅Rₜ⋅Sₜᵀ$ where $Sₜ$
    is the $mₜ×m$ projection matrix of the missing values. In this case:

    .. math::
        x̂' &= x̂ + P⋅Hᵀ⋅Sᵀ(SHPHᵀSᵀ + SRSᵀ)⁻¹ (Sy - SHx̂) \\
           &= x̂ + P⋅Hᵀ⋅Sᵀ(S (HPHᵀ + R) Sᵀ)⁻¹ S(y - Hx̂) \\
           &= x̂ + P⋅Hᵀ⋅(S⁺S)ᵀ (HPHᵀ + R)⁻¹ (S⁺S) (y - Hx̂) \\
           &= x̂ + P⋅Hᵀ⋅∏ₘᵀ (HPHᵀ + R)⁻¹ ∏ₘ (y - Hx̂) \\
        P' &= P - P⋅Hᵀ⋅Sᵀ(S H P⁻¹ Hᵀ Sᵀ + SRSᵀ)⁻¹ SH P⁻¹ \\
           &= P - P⋅Hᵀ⋅(S⁺S)ᵀ (H P⁻¹ Hᵀ + R)⁻¹ (S⁺S) H P⁻¹ \\
           &= P - P⋅Hᵀ⋅∏ₘᵀ (H P⁻¹ Hᵀ + R)⁻¹ ∏ₘ H P⁻¹


    .. note::
        The Kalman filter is a linear filter. The non-linear version is also possible,
        the so called Extended Kalman-Filter. Here, the non-linearity is linearized at
        the time of update.

        ..math ::
            x̂' &= x̂ + P⋅Hᵀ(HPHᵀ + R)⁻¹ (y - h(x̂)) \\
            P' &= P -  P⋅Hᵀ(HPHᵀ + R)⁻¹ H P

        where $H = \frac{∂h}{∂x}|_{x̂}$. Note that the EKF is generally not an optimal
        filter.
    """

    input_size: Final[int]
    hidden_size: Final[int]
    use_cholesky: Final[bool]
    batch_first: Final[bool]
    initial_state_learnable: Final[bool]
    process_noise_learnable: Final[bool]
    observation_noise_learnable: Final[bool]

    # PARAMETERS
    system_matrix: Tensor
    observation_matrix: Tensor
    initial_mean: Tensor

    # BUFFERS
    initial_cov: Tensor
    process_cov: Tensor
    measurement_cov: Tensor

    prior_latent_means: Tensor
    r"""The (a priori) latent mean μₖ for the most recent forward pass."""
    prior_latent_covs: Tensor
    r"""The (a priori) latent covariance Σₖ for the most recent forward pass."""
    prior_target_means: Tensor
    r"""The (a priori) predicted mean $yₖ=Hμₖ$ for the most recent forward pass."""
    prior_target_covs: Tensor
    r"""The (a priori) predicted covariance $Sₖ=HΣₖHᵀ+R$ for the most recent forward pass."""
    post_latent_means: Tensor
    r"""The (a posteriori) mean μₖ' after measurement update for the most recent forward pass."""
    post_latent_covs: Tensor
    r"""The (a posteriori) covariance Σₖ' after measurement update for the most recent forward pass."""
    post_target_means: Tensor
    r"""The (a posteriori) predicted mean yₖ'=Hμₖ' for the most recent forward pass."""
    post_target_covs: Tensor
    r"""The (a posteriori) predicted covariance Sₖ'=HΣₖ'Hᵀ+R for the most recent forward pass."""
    identity_matrix: Tensor
    r"""The identity matrix Iₙ used in the Joseph covariance update."""
    van_loan_matrix: Tensor
    r"""The Van Loan block matrix $[[F,Q],[0,-Fᵀ]]$ used for propagation."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        *,
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
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.use_cholesky = use_cholesky
        self.batch_first = batch_first
        self.initial_state_learnable = initial_state_learnable
        self.process_noise_learnable = process_noise_learnable
        self.observation_noise_learnable = observation_noise_learnable
        m = self.input_size
        n = self.hidden_size
        process_noise = _as_covariance(process_noise, n)
        measurement_noise = _as_covariance(measurement_noise, m)
        initial_covariance = _as_covariance(initial_covariance, n)

        # initialize parameters
        self.system_matrix = nn.Parameter(
            self._sample_default_system_matrix()
            if system_matrix is None
            else torch.as_tensor(system_matrix),
            requires_grad=True,
        )
        self.observation_matrix = nn.Parameter(
            self._sample_default_observation_matrix()
            if observation_matrix is None
            else torch.as_tensor(observation_matrix),
            requires_grad=True,
        )
        self.initial_mean = nn.Parameter(
            _as_mean(initial_mean, n),
            requires_grad=initial_state_learnable,
        )
        self._initial_covariance_param = nn.Parameter(
            _log_spd(initial_covariance),
            requires_grad=initial_state_learnable,
        )
        self._process_covariance_param = nn.Parameter(
            _log_spd(process_noise),
            requires_grad=process_noise_learnable,
        )
        self._measurement_covariance_param = nn.Parameter(
            _log_spd(measurement_noise),
            requires_grad=observation_noise_learnable,
        )

        # register buffers
        self.register_buffer("process_cov", process_noise.clone())
        self.register_buffer("measurement_cov", measurement_noise.clone())
        self.register_buffer("initial_cov", initial_covariance.clone())
        self.register_buffer("identity_matrix", torch.eye(n))
        self.register_buffer("van_loan_matrix", torch.empty(2 * n, 2 * n))
        self.update_buffers()

        self.register_buffer("prior_latent_means", None, persistent=False)
        self.register_buffer("prior_latent_covs", None, persistent=False)
        self.register_buffer("prior_target_means", None, persistent=False)
        self.register_buffer("prior_target_covs", None, persistent=False)
        self.register_buffer("post_latent_means", None, persistent=False)
        self.register_buffer("post_latent_covs", None, persistent=False)
        self.register_buffer("post_target_means", None, persistent=False)
        self.register_buffer("post_target_covs", None, persistent=False)

        self.register_buffer("pred_means", None, persistent=False)
        self.register_buffer("pred_covs", None, persistent=False)

        # validate model
        self.validate_parameters()
        self.validate_persistent_buffers()

    def validate_parameters(self) -> None:
        r"""Validate dimensions of parameters."""
        m = self.input_size
        n = self.hidden_size
        x = self.initial_mean
        P = self.initial_cov
        F = self.system_matrix
        Q = self.process_cov
        H = self.observation_matrix
        R = self.measurement_cov
        Q_param = self._process_covariance_param
        R_param = self._measurement_covariance_param
        P_param = self._initial_covariance_param

        assert F.shape == (n, n)
        assert Q.shape == (n, n)
        assert H.shape == (m, n)
        assert R.shape == (m, m)
        assert x.shape == (n,)
        assert P.shape == (n, n)
        assert Q_param.shape == (n, n)
        assert R_param.shape == (m, m)
        assert P_param.shape == (n, n)

        # check that covariance matrices are symmetric positive definite
        assert torch.allclose(Q, Q.mT), "Process noise Q not symmetric"
        assert torch.allclose(R, R.mT), "Measurement noise R not symmetric"
        assert torch.allclose(P, P.mT), "Initial covariance P0 not symmetric"
        assert torch.linalg.eigvalsh(Q).min() >= 0, (
            "Process noise Q not positive semidefinite"
        )
        assert torch.linalg.eigvalsh(R).min() > 0, (
            "Measurement noise R not positive definite"
        )
        assert torch.linalg.eigvalsh(P).min() > 0, (
            "Initial covariance P0 not positive definite"
        )

    def validate_persistent_buffers(self) -> None:
        r"""Validate dimensions of persistent buffers."""
        m = self.input_size
        n = self.hidden_size
        assert self.process_cov.shape == (n, n)
        assert self.measurement_cov.shape == (m, m)
        assert self.initial_cov.shape == (n, n)
        assert self.identity_matrix.shape == (n, n)
        assert self.van_loan_matrix.shape == (2 * n, 2 * n)

    def validate_buffers(self) -> None:
        r"""Validate dimensions of buffers populated by the last forward pass."""
        m = self.input_size
        n = self.hidden_size
        bs = self.prior_latent_means.shape[:-1]
        # check shapes
        assert self.prior_latent_means.shape == (*bs, n)
        assert self.prior_latent_covs.shape == (*bs, n, n)
        assert self.prior_target_means.shape == (*bs, m)
        assert self.prior_target_covs.shape == (*bs, m, m)
        assert self.post_latent_means.shape == (*bs, n)
        assert self.post_latent_covs.shape == (*bs, n, n)
        assert self.post_target_means.shape == (*bs, m)
        assert self.post_target_covs.shape == (*bs, m, m)

    @torch.no_grad()
    def _sample_default_system_matrix(self) -> Tensor:
        r"""Sample a random continuous-time system matrix.

        A skew-symmetric generator has purely imaginary eigenvalues, hence
        $\exp(tF)$ is orthogonal for the deterministic dynamics.
        """
        matrix = torch.randn(
            self.hidden_size,
            self.hidden_size,
            dtype=torch.get_default_dtype(),
        )
        return (matrix - matrix.mT) / (2 * self.hidden_size) ** 0.5

    @torch.no_grad()
    def _sample_default_observation_matrix(self) -> Tensor:
        r"""Sample a random observation matrix."""
        t = torch.randn(self.input_size, self.hidden_size)
        nn.init.kaiming_uniform_(t)
        return t

    def predict(
        self,
        *,
        query_times: Tensor,  # Float[..., K], padded NaN, non-decreasing
        query_mask: Tensor,  # Bool[..., K, F]  padded False
        context_times: Tensor,  # Float[..., N], padded NaN, non-decreasing
        context_mask: Tensor,  # Bool[..., N, D], padded False
        context_values: Tensor,  # Float[..., N, D], padded NaN, sparse
        # μ₀=(..., D) Σ₀=(..., D, D)
        initial_state: tuple[Tensor, Tensor] | None = None,
        initial_time: Tensor | None = None,  # t₀, ()
    ) -> tuple[Tensor, Tensor]:  # (..., $K, D), (..., $K, D)
        r"""Compute the predictive mean and variance over split time representation."""
        combined = EventBatch.from_request(
            context_times=context_times,
            context_values=context_values,
            context_mask=context_mask,
            query_times=query_times,
            query_mask=query_mask,
            batch_first=self.batch_first,
        )
        post_means, post_covs = self.forward(
            timestamps=combined.timestamps,  # Float[..., $T], padded NaN, non-decreasing
            context_values=combined.context_values,  # Float[..., $T, D], padded NaN, sparse
            context_mask=combined.context_mask,  # Bool[..., $T, D], padded False
            query_mask=combined.query_mask,  # Bool[..., $T, F], padded False
            initial_state=initial_state,
            initial_time=initial_time,
        )

        # initialize mask over valid time steps:
        # those with finite time and at least one observation or query coordinate.
        valid_steps = (combined.context_mask | combined.query_mask).any(dim=-1)
        mean_mask = valid_steps.unsqueeze(dim=-1)
        cov_mask = mean_mask.unsqueeze(dim=-1)

        pred_means, pred_covs = self.decode_state(
            (
                post_means.masked_fill(~mean_mask, 0.0),
                post_covs.masked_fill(~cov_mask, 0.0),
            )
        )
        pred_means = pred_means.masked_fill(~mean_mask, nan)
        pred_covs = pred_covs.masked_fill(~cov_mask, nan)

        self.pred_means = pred_means[..., *combined.query_indices, :]
        self.pred_covs = pred_covs[..., *combined.query_indices, :, :]
        return self.pred_means, self.pred_covs

    def forward(
        self,
        *,
        timestamps: Tensor,  # Float[..., $T], padded NaN
        query_mask: Tensor,  # Bool[..., $T, D], padded False
        context_values: Tensor,  # Float[..., $T, D], padded NaN, sparse
        context_mask: Tensor,  # Bool[..., $T, D], padded False
        # μ₀=(..., D) Σ₀=(..., D, D)
        initial_state: tuple[Tensor, Tensor] | None = None,
        initial_time: Tensor | None = None,  # t₀, ()
    ) -> tuple[Tensor, Tensor]:  # Float[..., $T, D], Float[..., $T, D, D]
        r"""Compute the posterior latent states over joint time representation.

        Args:
            timestamps: combined time points at which to filter.
            query_mask: Boolean mask selecting requested forecast entries.
            context_mask: Boolean mask selecting observed entries in ``values``.
            context_values: Sparse observations at context time points.
            initial_state: Optional initial latent state $(μ₀, Σ₀)$.
                If omitted, uses the model initial mean and covariance.
            initial_time: Optional initial time $t₀$. If omitted, defaults to
                the first time step.

        Returns:
            y_pred: Posterior predicted means $μ̂ₜ=\E[ŷₜ]$ at all time points.
            S_pred: Posterior predicted covariances $Σ̂ₜ=\Var[ŷₜ]$ at all time points.
        """
        # update cached tensors
        self.update_buffers()

        # initialize mask over valid time steps:
        # those with finite time and at least one observation or query coordinate.
        valid_steps = (context_mask | query_mask).any(dim=-1)
        mean_mask = valid_steps.unsqueeze(dim=-1)
        cov_mask = mean_mask.unsqueeze(dim=-1)

        if self.batch_first:
            # Move the time axis to the front.
            timestamps = timestamps.movedim(-1, 0)
            context_values = context_values.movedim(-2, 0)
            context_mask = context_mask.movedim(-2, 0)
            query_mask = query_mask.movedim(-2, 0)
            valid_steps = valid_steps.movedim(-1, 0)

        # check the shapes
        m = self.input_size
        n = self.hidden_size
        num_steps, *batch_shape, _ = context_values.shape
        assert timestamps.shape == (num_steps, *batch_shape)
        assert context_values.shape == (num_steps, *batch_shape, m)
        assert context_mask.shape == (num_steps, *batch_shape, m)
        assert query_mask.shape == (num_steps, *batch_shape, m)
        assert valid_steps.shape == (num_steps, *batch_shape)
        assert context_mask.dtype == torch.bool
        assert query_mask.dtype == torch.bool

        # initialize the state
        t = timestamps[0] if initial_time is None else initial_time
        x_pre, P_pre = (
            (self.initial_mean, self.initial_cov)
            if initial_state is None
            else initial_state
        )
        x_pre = x_pre.expand(*batch_shape, n)
        P_pre = P_pre.expand(*batch_shape, n, n)
        x_post = x_pre.clone()
        P_post = P_pre.clone()

        # initialize the buffers
        prior_latent_means: list[Tensor] = []
        prior_latent_covs: list[Tensor] = []
        post_latent_means: list[Tensor] = []
        post_latent_covs: list[Tensor] = []

        for t_obs, y_obs, mask, active in zip(
            timestamps,
            context_values,
            context_mask,
            valid_steps,
            strict=True,
        ):
            # Within the loop we use batch-first.
            delta = torch.where(active, t_obs - t, torch.zeros_like(t_obs - t))
            t = torch.where(active, t_obs, t)

            # propagate forward in time
            x_pre, P_pre = self.propagate_state(delta, (x_post, P_post))
            prior_latent_means.append(x_pre)
            prior_latent_covs.append(P_pre)

            # Update step
            x_update, P_update = self.update_state(y_obs, (x_pre, P_pre), mask=mask)
            x_post = torch.where(active[..., None], x_update, x_post)
            P_post = torch.where(active[..., None, None], P_update, P_post)
            post_latent_means.append(x_post)
            post_latent_covs.append(P_post)

        # TODO: consider optional backward pass with RTS smoother
        stack_dim_mean = -2 if self.batch_first else 0
        stack_dim_cov = -3 if self.batch_first else 0

        self.prior_latent_means = stack(prior_latent_means, dim=stack_dim_mean)
        self.prior_latent_covs = stack(prior_latent_covs, dim=stack_dim_cov)
        self.post_latent_means = stack(post_latent_means, dim=stack_dim_mean)
        self.post_latent_covs = stack(post_latent_covs, dim=stack_dim_cov)

        self.prior_latent_means = self.prior_latent_means.masked_fill(~mean_mask, nan)
        self.prior_latent_covs = self.prior_latent_covs.masked_fill(~cov_mask, nan)
        self.post_latent_means = self.post_latent_means.masked_fill(~mean_mask, nan)
        self.post_latent_covs = self.post_latent_covs.masked_fill(~cov_mask, nan)
        # self.validate_buffers()

        return self.post_latent_means, self.post_latent_covs

    def log_prob(
        self,
        values: Tensor,  # Float[..., $K, D]
        *,
        query_times: Tensor,  # Float[..., K], padded NaN, non-decreasing
        query_mask: Tensor,  # Bool[..., K, D], padded False
        context_times: Tensor,  # Float[..., N], padded NaN, non-decreasing
        context_values: Tensor,  # Float[..., N, D], padded NaN, sparse
        context_mask: Tensor,  # Bool[..., N, D], padded False
        initial_state: tuple[Tensor, Tensor] | None = None,
        initial_time: Tensor | None = None,  # t₀, ()
    ) -> Tensor:  # Float[..., $K]
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
        return marginal_gaussian_log_prob(
            values,
            mean=mean.expand(*values.shape),
            cov=cov.expand(*values.shape[:-1], *cov.shape[-2:]),
            mask=query_mask.expand(*values.shape),
        )

    def sample(
        self,
        size: int | tuple[int, ...] = (),  # *S
        *,
        query_times: Tensor,  # Float[..., K], padded NaN, non-decreasing
        query_mask: Tensor,  # Bool[..., K, D], padded False
        context_times: Tensor,  # Float[..., N], padded NaN, non-decreasing
        context_values: Tensor,  # Float[..., N, D], padded NaN, sparse
        context_mask: Tensor,  # Bool[..., N, D], padded False
        initial_state: tuple[Tensor, Tensor] | None = None,
        initial_time: Tensor | None = None,  # t₀, ()
        rng: Generator | None = None,
    ) -> Tensor:  # (*S, ..., $K, D)
        r"""Sample from the time-marginal distribution.

        .. math:: pₖ = p_{Y_{qₖ}}(yₖ | (t₁, y₁), ..., (tₙ, yₙ))
        """
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
        return marginal_gaussian_sample(
            sample_shape,
            mean=mean,
            cov=cov,
            mask=query_mask,
            rng=rng,
        )

    def sample_and_log_prob(
        self,
        size: int | tuple[int, ...] = (),  # *S
        *,
        query_times: Tensor,  # Float[..., K], padded NaN, non-decreasing
        query_mask: Tensor,  # Bool[..., K, D], padded False
        context_times: Tensor,  # Float[..., N], padded NaN, non-decreasing
        context_values: Tensor,  # Float[..., N, D], padded NaN, sparse
        context_mask: Tensor,  # Bool[..., N, D], padded False
        initial_state: tuple[Tensor, Tensor] | None = None,
        initial_time: Tensor | None = None,  # t₀, ()
        rng: Generator | None = None,
    ) -> tuple[Tensor, Tensor]:  # (*S, ..., $K, D), (*S, ..., $K)
        r"""Sample from the time-marginal distribution and yield log-probabilities.

        .. math:: pₖ = p_{Y_{qₖ}}(yₖ | (t₁, y₁), ..., (tₙ, yₙ))
        """
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
        return marginal_gaussian_sample_and_log_prob(
            sample_shape,
            mean=mean,
            cov=cov,
            mask=query_mask,
            rng=rng,
        )

    def update_buffers(self) -> None:
        r"""Refresh derived buffers from current parameters."""
        self.process_cov = _spd_expm(self._process_covariance_param)
        self.measurement_cov = _spd_expm(self._measurement_covariance_param)
        self.initial_cov = _spd_expm(self._initial_covariance_param)

        self.update_van_loan_matrix()

    def update_van_loan_matrix(self) -> None:
        r"""Refresh the Van Loan block matrix from current dynamics parameters."""
        F = self.system_matrix
        Q = self.process_cov
        self.van_loan_matrix = torch.cat(  # [[F, Q], [0, -Fᵀ]]
            [  # [2n, 2n]
                torch.cat([F, Q], dim=-1),
                torch.cat([torch.zeros_like(F), -F.mT], dim=-1),
            ],
            dim=0,
        )

    def propagate_state(
        self,
        delta: Tensor,
        state: tuple[Tensor, Tensor],
        /,
    ) -> tuple[Tensor, Tensor]:
        r"""Propagate latent mean and covariance through continuous dynamics."""
        x, P = state
        n = self.hidden_size

        # Concise implementation with a single matrix exponential.
        # exp([[F, Q], [0, -Fᵀ]])  = [[G, Φ], [0, G⁻ᵀ]]
        expMt = matrix_exp(self.van_loan_matrix * delta[..., None, None])
        G = expMt[..., :n, :n]
        Phi = expMt[..., :n, n:]

        # Propagate mean and covariance.
        x_new = einsum("...ij, ...j -> ...i", G, x)  # (*B, n)
        P_new = Phi + einsum("...ik, ...kl, ...jl -> ...ij", G, P, G)  # (*B, n, n)
        return x_new, P_new

    def decode_state(self, state: tuple[Tensor, Tensor], /) -> tuple[Tensor, Tensor]:
        r"""Decode latent mean and covariance to observation space."""
        x, P = state
        H = self.observation_matrix
        R = self.measurement_cov
        y = einsum("ij, ...j -> ...i", H, x)  # (*B, m)
        S = R + einsum("ik, ...kl, jl -> ...ij", H, P, H)  # (*B, m, m)
        return y, S

    def update_state(
        self,
        y_obs: Tensor,
        state: tuple[Tensor, Tensor],
        /,
        *,
        mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        r"""Update latent mean and covariance with a sparse observation."""
        y_pred, _ = self.decode_state(state)
        x, P = state
        H = self.observation_matrix
        R = self.measurement_cov
        H = torch.where(mask[..., None], H, 0.0)
        R = (
            torch.where(mask.unsqueeze(-1) & mask.unsqueeze(-2), R, 0.0)
            + (~mask).to(R.dtype).diag_embed()
        )

        # Innovation residual: ignore unobserved coordinates.
        r = torch.where(mask, y_obs - y_pred, torch.zeros_like(y_pred))

        # Kalman gain computation.
        K = self._compute_kalman_gain(P, H, R)  # (*B, n, m)

        # Mean update.
        x_new = x + einsum("...ij, ...j -> ...i", K, r)  # (*B, n)

        # Joseph covariance update.
        P_new = self._joseph_update(P, K, H, R)
        return x_new, P_new

    def _compute_kalman_gain(
        self,
        P: Tensor,
        H: Tensor,
        R: Tensor,
    ) -> Tensor:
        """Compute Kalman gain K.

        K = P Hᵀ S⁻¹

        Args:
            P: Prior covariance Σₖ of shape (*B, n, n)
            H: Masked observation matrix H of shape (*B, m, n)
            R: Masked measurement covariance of shape (*B, m, m)

        Returns:
            K: Kalman gain of shape (*B, n, m)
        """
        S = R + einsum("...ik, ...kl, ...jl -> ...ij", H, P, H)

        # Solve for K using Cholesky factors
        if self.use_cholesky:
            # S = LLᵀ ⟹ K = PHᵀL⁻ᵀL⁻¹
            # ⟹ LᵀKᵀ = (L⁻¹HP)ᵀ = G ⟹ L Gᵀ = HP
            # so, solve Gᵀ = solve_triangular(L, HP, lower=True)
            # then K = solve_triangular(L.T, G, lower=False).T
            raise NotImplementedError
        else:  # noqa: RET506
            # KS = PHᵀ ⟹ SᵀKᵀ = HP
            # NOTE: we can't use tensor.T for batched tensors.
            Kt = torch.linalg.solve(S.mT, H @ P)  # (*B, m, n)
            K = Kt.mT  # (*B, n, m)

        return K

    def _joseph_update(
        self,
        P: Tensor,
        K: Tensor,
        H: Tensor,
        R: Tensor,
    ) -> Tensor:
        """Compute Joseph form update for covariance.

        Σₖ' = (I - KH) Σₖ (I - KH)ᵀ + K R Kᵀ

        Args:
            P: Prior covariance Σₖ of shape (*B, n, n)
            K: Kalman gain K of shape (*B, n, m)
            H: Masked observation matrix H of shape (*B, m, n)
            R: Masked measurement covariance of shape (*B, m, m)

        Returns:
            P_new: Updated covariance Σₖ' of shape (*B, n, n)
        """
        I_KH = self.identity_matrix - einsum("...ik, ...kj -> ...ij", K, H)
        P_new = (
            einsum("...ik, ...kl, ...jl -> ...ij", I_KH, P, I_KH)  # (*B, n, n)
            + einsum("...ik, ...kl, ...jl -> ...ij", K, R, K)  # (*B, n, n)
        )
        return P_new


class DiscreteTimeKalmanFilter(nn.Module):
    r"""Discrete, time-invariant Kalman Filter.

    .. math::
        xₖ₊₁ &= Fxₖ + wₖ   &   wₖ &~ N(0, Q) \\
        yₖ   &= Hxₖ + vₖ   &   vₖ &~ N(0, R)
    """

    input_size: Final[int]
    hidden_size: Final[int]
    use_cholesky: Final[bool]
    batch_first: Final[bool]

    # PARAMETERS
    system_matrix: Tensor
    observation_matrix: Tensor
    process_covariance: Tensor
    measurement_covariance: Tensor
    initial_mean: Tensor
    initial_covariance: Tensor

    # BUFFERS
    identity_matrix: Tensor
    r"""The identity matrix Iₙ used in the Joseph covariance update."""
    prior_latent_means: Tensor
    r"""The (a priori) latent mean μₖ for the most recent forward pass."""
    prior_latent_covariances: Tensor
    r"""The (a priori) latent covariance Σₖ for the most recent forward pass."""
    prior_predicted_means: Tensor
    r"""The (a priori) predicted mean $yₖ=Hμₖ$ for the most recent forward pass."""
    prior_predicted_covariances: Tensor
    r"""The (a priori) predicted covariance $Sₖ=HΣₖHᵀ+R$ for the most recent forward pass."""
    posterior_latent_means: Tensor
    r"""The (a posteriori) mean μₖ' after measurement update for the most recent forward pass."""
    posterior_latent_covariances: Tensor
    r"""The (a posteriori) covariance Σₖ' after measurement update for the most recent forward pass."""
    posterior_predicted_means: Tensor
    r"""The (a posteriori) predicted mean yₖ'=Hμₖ' for the most recent forward pass."""
    posterior_predicted_covariances: Tensor
    r"""The (a posteriori) predicted covariance Sₖ'=HΣₖ'Hᵀ+R for the most recent forward pass."""
    pred_means: Tensor
    r"""The query predictive means from the most recent predict call."""
    pred_covs: Tensor
    r"""The query predictive covariances from the most recent predict call."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        *,
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
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.use_cholesky = use_cholesky
        self.batch_first = batch_first
        m = self.input_size
        n = self.hidden_size

        # initialize parameters
        self.system_matrix = nn.Parameter(
            self._sample_default_system_matrix()
            if system_matrix is None
            else torch.as_tensor(system_matrix),
            requires_grad=learnable,
        )
        self.observation_matrix = nn.Parameter(
            self._sample_default_observation_matrix()
            if observation_matrix is None
            else torch.as_tensor(observation_matrix),
            requires_grad=learnable,
        )
        self.process_covariance = nn.Parameter(
            _as_covariance(
                1.0 if process_covariance is None else process_covariance, n
            ),
            requires_grad=learnable,
        )
        self.measurement_covariance = nn.Parameter(
            _as_covariance(
                1.0 if measurement_covariance is None else measurement_covariance,
                m,
            ),
            requires_grad=learnable,
        )
        self.initial_mean = nn.Parameter(
            _as_mean(0.0 if initial_mean is None else initial_mean, n),
            requires_grad=learnable,
        )
        self.initial_covariance = nn.Parameter(
            _as_covariance(
                1.0 if initial_covariance is None else initial_covariance,
                n,
            ),
            requires_grad=learnable,
        )

        # register buffers
        self.register_buffer("identity_matrix", torch.eye(n))
        self.register_buffer("prior_latent_means", None, persistent=False)
        self.register_buffer("prior_latent_covariances", None, persistent=False)
        self.register_buffer("prior_predicted_means", None, persistent=False)
        self.register_buffer("prior_predicted_covariances", None, persistent=False)
        self.register_buffer("posterior_latent_means", None, persistent=False)
        self.register_buffer("posterior_latent_covariances", None, persistent=False)
        self.register_buffer("posterior_predicted_means", None, persistent=False)
        self.register_buffer("posterior_predicted_covariances", None, persistent=False)
        self.register_buffer("pred_means", None, persistent=False)
        self.register_buffer("pred_covs", None, persistent=False)

        # validate model
        self.validate_parameters()

    def validate_parameters(self) -> None:
        r"""Validate dimensions of parameters."""
        m = self.input_size
        n = self.hidden_size
        x = self.initial_mean
        P = self.initial_covariance
        F = self.system_matrix
        Q = self.process_covariance
        H = self.observation_matrix
        R = self.measurement_covariance

        assert F.shape == (n, n)
        assert Q.shape == (n, n)
        assert H.shape == (m, n)
        assert R.shape == (m, m)
        assert x.shape == (n,)
        assert P.shape == (n, n)
        assert self.identity_matrix.shape == (n, n)

        # check that covariance matrices are symmetric positive definite
        assert torch.allclose(Q, Q.mT), "Process noise Q not symmetric"
        assert torch.allclose(R, R.mT), "Measurement noise R not symmetric"
        assert torch.allclose(P, P.mT), "Initial covariance P0 not symmetric"
        assert torch.linalg.eigvalsh(Q).min() >= 0, (
            "Process noise Q not positive semidefinite"
        )
        assert torch.linalg.eigvalsh(R).min() > 0, (
            "Measurement noise R not positive definite"
        )
        assert torch.linalg.eigvalsh(P).min() > 0, (
            "Initial covariance P0 not positive definite"
        )

    def validate_buffers(self) -> None:
        r"""Validate dimensions of buffers populated by the last forward pass."""
        m = self.input_size
        n = self.hidden_size
        bs = self.prior_latent_means.shape[:-1]
        # check shapes
        assert self.prior_latent_means.shape == (*bs, n)
        assert self.prior_latent_covariances.shape == (*bs, n, n)
        assert self.prior_predicted_means.shape == (*bs, m)
        assert self.prior_predicted_covariances.shape == (*bs, m, m)
        assert self.posterior_latent_means.shape == (*bs, n)
        assert self.posterior_latent_covariances.shape == (*bs, n, n)
        assert self.posterior_predicted_means.shape == (*bs, m)
        assert self.posterior_predicted_covariances.shape == (*bs, m, m)

    @torch.no_grad()
    def _sample_default_system_matrix(self) -> Tensor:
        r"""Sample a random system matrix.

        We take an orthogonal matrix for stability.
        """
        matrix = torch.randn(
            self.hidden_size,
            self.hidden_size,
            dtype=torch.get_default_dtype(),
        )
        q, r = torch.linalg.qr(matrix)
        return q * r.diagonal().sign().clamp(min=1).unsqueeze(0)

    @torch.no_grad()
    def _sample_default_observation_matrix(self) -> Tensor:
        r"""Sample a random observation matrix."""
        t = torch.randn(self.input_size, self.hidden_size)
        nn.init.kaiming_uniform_(t)
        return t

    def predict(
        self,
        *,
        query_times: Tensor,  # Integer[..., $K], padded arbitrary, non-decreasing
        query_mask: Tensor,  # Bool[..., $K, D], padded False
        context_times: Tensor,  # Integer[..., $N], padded arbitrary, non-decreasing
        context_mask: Tensor,  # Bool[..., $N, D], padded False
        context_values: Tensor,  # Float[..., $N, D], padded NaN, sparse
        # μ₀=(..., d) Σ₀=(..., d, d)
        initial_state: tuple[Tensor, Tensor] | None = None,
        initial_time: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:  # (..., $K, D), (..., $K, D, D)
        r"""Compute posterior predictive means and covariances at query steps."""
        assert not torch.is_floating_point(query_times)
        assert not torch.is_floating_point(context_times)
        assert initial_time is None or not torch.is_floating_point(initial_time)
        combined = DiscreteTimeEventBatch.from_request(
            context_times=context_times,
            context_values=context_values,
            context_mask=context_mask,
            query_times=query_times,
            query_mask=query_mask,
            batch_first=self.batch_first,
        )
        post_means, post_covs = self.forward(
            steps=combined.steps,
            context_values=combined.context_values,
            context_mask=combined.context_mask,
            query_mask=combined.query_mask,
            initial_state=initial_state,
            initial_step=initial_time,
        )

        valid_steps = (combined.context_mask | combined.query_mask).any(dim=-1)
        mean_mask = valid_steps.unsqueeze(dim=-1)
        cov_mask = mean_mask.unsqueeze(dim=-1)

        pred_means, pred_covs = self.decode_state(
            (
                post_means.masked_fill(~mean_mask, 0.0),
                post_covs.masked_fill(~cov_mask, 0.0),
            )
        )
        pred_means = pred_means.masked_fill(~mean_mask, nan)
        pred_covs = pred_covs.masked_fill(~cov_mask, nan)

        self.pred_means = pred_means[..., *combined.query_indices, :]
        self.pred_covs = pred_covs[..., *combined.query_indices, :, :]
        return self.pred_means, self.pred_covs

    def forward(
        self,
        *,
        steps: Tensor,  # Long[..., $T], padded arbitrary, non-decreasing
        query_mask: Tensor,  # Bool[..., $T, D], padded False
        context_values: Tensor,  # Float[..., $T, D], padded NaN, sparse
        context_mask: Tensor,  # Bool[..., $T, D], padded False
        # μ₀=(..., D) Σ₀=(..., D, D)
        initial_state: tuple[Tensor, Tensor] | None = None,
        initial_step: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:  # Float[..., $T, D], Float[..., $T, D, D]
        r"""Compute posterior latent states over joint discrete event steps.

        Integer step gaps propagate by that many transition steps.
        """
        valid_steps = (context_mask | query_mask).any(dim=-1)
        mean_mask = valid_steps.unsqueeze(dim=-1)
        cov_mask = mean_mask.unsqueeze(dim=-1)
        assert not torch.is_floating_point(steps)
        assert initial_step is None or not torch.is_floating_point(initial_step)

        if self.batch_first:
            # Move the time axis to the front.
            steps = steps.movedim(-1, 0)
            context_values = context_values.movedim(-2, 0)
            context_mask = context_mask.movedim(-2, 0)
            query_mask = query_mask.movedim(-2, 0)
            valid_steps = valid_steps.movedim(-1, 0)

        # check the shapes
        m = self.input_size
        n = self.hidden_size
        num_steps, *batch_shape, _ = context_values.shape
        assert steps.shape == (num_steps, *batch_shape)
        assert context_values.shape == (num_steps, *batch_shape, m)
        assert context_mask.shape == (num_steps, *batch_shape, m)
        assert query_mask.shape == (num_steps, *batch_shape, m)
        assert valid_steps.shape == (num_steps, *batch_shape)
        assert context_mask.dtype == torch.bool
        assert query_mask.dtype == torch.bool

        # initialize the state
        step = steps[0] if initial_step is None else initial_step
        x_pre, P_pre = (
            (self.initial_mean, self.initial_covariance)
            if initial_state is None
            else initial_state
        )
        x_pre = x_pre.expand(*batch_shape, n)
        P_pre = P_pre.expand(*batch_shape, n, n)
        x_post = x_pre.clone()
        P_post = P_pre.clone()

        # initialize the buffers
        prior_latent_means: list[Tensor] = []
        prior_latent_covariances: list[Tensor] = []
        prior_predicted_means: list[Tensor] = []
        prior_predicted_covariances: list[Tensor] = []
        posterior_latent_means: list[Tensor] = []
        posterior_latent_covariances: list[Tensor] = []
        posterior_predicted_means: list[Tensor] = []
        posterior_predicted_covariances: list[Tensor] = []

        for step_obs, y_obs, mask, active in zip(
            steps,
            context_values,
            context_mask,
            valid_steps,
            strict=True,
        ):
            # Within the loop we use batch-first.
            delta = torch.where(active, step_obs - step, torch.zeros_like(step_obs))
            step = torch.where(active, step_obs, step)

            # Propagate forward in discrete time.
            prior_state = self.propagate_state(delta, (x_post, P_post))
            prior_latent_means.append(prior_state[0])
            prior_latent_covariances.append(prior_state[1])

            y_pre, S_pre = self.decode_state(prior_state)
            prior_predicted_means.append(y_pre)
            prior_predicted_covariances.append(S_pre)

            # Update step.
            x_update, P_update = self.update_state(y_obs, prior_state, mask=mask)
            x_post = torch.where(active[..., None], x_update, x_post)
            P_post = torch.where(active[..., None, None], P_update, P_post)
            posterior_latent_means.append(x_post)
            posterior_latent_covariances.append(P_post)

            y_post, S_post = self.decode_state((x_post, P_post))
            posterior_predicted_means.append(y_post)
            posterior_predicted_covariances.append(S_post)

        stack_dim_mean = -2 if self.batch_first else 0
        stack_dim_cov = -3 if self.batch_first else 0

        self.prior_latent_means = stack(prior_latent_means, dim=stack_dim_mean)
        self.prior_latent_covariances = stack(
            prior_latent_covariances,
            dim=stack_dim_cov,
        )
        self.prior_predicted_means = stack(
            prior_predicted_means,
            dim=stack_dim_mean,
        )
        self.prior_predicted_covariances = stack(
            prior_predicted_covariances,
            dim=stack_dim_cov,
        )
        self.posterior_latent_means = stack(posterior_latent_means, dim=stack_dim_mean)
        self.posterior_latent_covariances = stack(
            posterior_latent_covariances,
            dim=stack_dim_cov,
        )
        self.posterior_predicted_means = stack(
            posterior_predicted_means,
            dim=stack_dim_mean,
        )
        self.posterior_predicted_covariances = stack(
            posterior_predicted_covariances,
            dim=stack_dim_cov,
        )

        self.prior_latent_means = self.prior_latent_means.masked_fill(~mean_mask, nan)
        self.prior_latent_covariances = self.prior_latent_covariances.masked_fill(
            ~cov_mask,
            nan,
        )
        self.prior_predicted_means = self.prior_predicted_means.masked_fill(
            ~mean_mask,
            nan,
        )
        self.prior_predicted_covariances = self.prior_predicted_covariances.masked_fill(
            ~cov_mask, nan
        )
        self.posterior_latent_means = self.posterior_latent_means.masked_fill(
            ~mean_mask,
            nan,
        )
        self.posterior_latent_covariances = (
            self.posterior_latent_covariances.masked_fill(~cov_mask, nan)
        )
        self.posterior_predicted_means = self.posterior_predicted_means.masked_fill(
            ~mean_mask,
            nan,
        )
        self.posterior_predicted_covariances = (
            self.posterior_predicted_covariances.masked_fill(~cov_mask, nan)
        )
        self.validate_buffers()

        return self.posterior_latent_means, self.posterior_latent_covariances

    def log_prob(
        self,
        values: Tensor,  # Float[..., $K, D]
        *,
        query_times: Tensor,  # Long[..., $K], padded arbitrary, non-decreasing
        query_mask: Tensor,  # Bool[..., $K, D], padded False
        context_times: Tensor,  # Long[..., $N], padded arbitrary, non-decreasing
        context_values: Tensor,  # Float[..., $N, D], padded NaN, sparse
        context_mask: Tensor,  # Bool[..., $N, D], padded False
        initial_state: tuple[Tensor, Tensor] | None = None,
        initial_time: Tensor | None = None,
    ) -> Tensor:  # Float[..., $K]
        r"""Compute the time-marginal log-likelihood of the model."""
        assert not torch.is_floating_point(query_times)
        assert not torch.is_floating_point(context_times)
        assert initial_time is None or not torch.is_floating_point(initial_time)
        mean, cov = self.predict(
            query_times=query_times,
            query_mask=query_mask,
            context_times=context_times,
            context_values=context_values,
            context_mask=context_mask,
            initial_state=initial_state,
            initial_time=initial_time,
        )
        return marginal_gaussian_log_prob(
            values,
            mean=mean.expand(*values.shape),
            cov=cov.expand(*values.shape[:-1], *cov.shape[-2:]),
            mask=query_mask.expand(*values.shape),
        )

    def sample(
        self,
        size: int | tuple[int, ...] = (),  # *S
        *,
        query_times: Tensor,  # Integer[..., $K], padded arbitrary, non-decreasing
        query_mask: Tensor,  # Bool[..., $K, D], padded False
        context_times: Tensor,  # Integer[..., $N], padded arbitrary, non-decreasing
        context_values: Tensor,  # Float[..., $N, D], padded NaN, sparse
        context_mask: Tensor,  # Bool[..., $N, D], padded False
        initial_state: tuple[Tensor, Tensor] | None = None,
        initial_time: Tensor | None = None,
        rng: Generator | None = None,
    ) -> Tensor:  # (*S, ..., $K, D)
        r"""Sample from the time-marginal predictive distribution."""
        assert not torch.is_floating_point(query_times)
        assert not torch.is_floating_point(context_times)
        assert initial_time is None or not torch.is_floating_point(initial_time)
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
        return marginal_gaussian_sample(
            sample_shape,
            mean=mean,
            cov=cov,
            mask=query_mask,
            rng=rng,
        )

    def sample_and_log_prob(
        self,
        size: int | tuple[int, ...] = (),  # *S
        *,
        query_times: Tensor,  # Integer[..., $K], padded arbitrary, non-decreasing
        query_mask: Tensor,  # Bool[..., $K, D], padded False
        context_times: Tensor,  # Integer[..., $N], padded arbitrary, non-decreasing
        context_values: Tensor,  # Float[..., $N, D], padded NaN, sparse
        context_mask: Tensor,  # Bool[..., $N, D], padded False
        initial_state: tuple[Tensor, Tensor] | None = None,
        initial_time: Tensor | None = None,
        rng: Generator | None = None,
    ) -> tuple[Tensor, Tensor]:  # (*S, ..., $K, D), (*S, ..., $K)
        r"""Sample and score from the time-marginal predictive distribution."""
        assert not torch.is_floating_point(query_times)
        assert not torch.is_floating_point(context_times)
        assert initial_time is None or not torch.is_floating_point(initial_time)
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
        return marginal_gaussian_sample_and_log_prob(
            sample_shape,
            mean=mean,
            cov=cov,
            mask=query_mask,
            rng=rng,
        )

    def propagate_state(
        self,
        delta: Tensor,
        state: tuple[Tensor, Tensor],
        /,
    ) -> tuple[Tensor, Tensor]:
        r"""Propagate latent mean and covariance through discrete dynamics."""
        x, P = state
        if delta.dtype != torch.long:
            raise TypeError("Discrete Kalman transition gaps must be Long tensors.")
        if bool((delta < 0).any()):
            raise ValueError("Discrete Kalman steps must be non-decreasing.")

        x_new = x
        P_new = P
        for k in range(int(delta.max().item())):
            active = delta > k
            x_prop = einsum("ij, ...j -> ...i", self.system_matrix, x_new)
            P_prop = self.process_covariance + einsum(
                "ik, ...kl, jl -> ...ij",
                self.system_matrix,
                P_new,
                self.system_matrix,
            )
            x_new = torch.where(active[..., None], x_prop, x_new)
            P_new = torch.where(active[..., None, None], P_prop, P_new)

        return x_new, P_new

    def decode_state(self, state: tuple[Tensor, Tensor]) -> tuple[Tensor, Tensor]:
        r"""Decode latent mean and covariance to observation space."""
        x, P = state
        H = self.observation_matrix
        R = self.measurement_covariance
        y = einsum("ij, ...j -> ...i", H, x)  # (*B, m)
        S = R + einsum("ik, ...kl, jl -> ...ij", H, P, H)  # (*B, m, m)
        return y, S

    def update_state(
        self,
        y_obs: Tensor,
        state: tuple[Tensor, Tensor],
        /,
        *,
        mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        r"""Update latent mean and covariance with a sparse observation."""
        y_pred, _ = self.decode_state(state)
        x, P = state

        H = self.observation_matrix
        R = self.measurement_covariance
        M = mask.to(P.dtype)
        missing = (~mask).to(P.dtype)
        H_masked = M.unsqueeze(-1) * H
        R_masked = M.unsqueeze(-1) * M.unsqueeze(-2) * R + missing.diag_embed()

        # Innovation residual: ignore unobserved coordinates.
        r = torch.where(mask, y_obs - y_pred, torch.zeros_like(y_pred))

        # Kalman gain computation.
        K = self._compute_kalman_gain(P, H_masked, R_masked)  # (*B, n, m)

        # Mean update.
        x_new = x + einsum("...ij, ...j -> ...i", K, r)  # (*B, n)

        # Joseph covariance update.
        P_new = self._joseph_update(P, K, H_masked, R_masked)
        return x_new, P_new

    def _compute_kalman_gain(
        self,
        P: Tensor,
        H: Tensor,
        R: Tensor,
    ) -> Tensor:
        """Compute Kalman gain K.

        K = P Hᵀ S⁻¹

        Args:
            P: Prior covariance Σₖ of shape (*B, n, n)
            H: Masked observation matrix Hₖ of shape (*B, m, n)
            R: Masked measurement covariance Rₖ of shape (*B, m, m)

        Returns:
            K: Kalman gain of shape (*B, n, m)
        """
        S = R + einsum("...ik, ...kl, ...jl -> ...ij", H, P, H)

        # Solve for K using Cholesky factors
        if self.use_cholesky:
            # S = LLᵀ ⟹ K = PHᵀL⁻ᵀL⁻¹
            # ⟹ LᵀKᵀ = (L⁻¹HP)ᵀ = G ⟹ L Gᵀ = solve_triangular(L, HP, lower=True)
            # then K = solve_triangular(L.T, G, lower=False).T
            raise NotImplementedError
        else:  # noqa: RET506
            # KS = PHᵀ ⟹ SᵀKᵀ = HP
            # NOTE: we can't use tensor.T for batched tensors.
            Kt = torch.linalg.solve(S.mT, H @ P)  # (*B, m, n)
            K = Kt.mT  # (*B, n, m)

        return K

    def _joseph_update(
        self,
        P: Tensor,
        K: Tensor,
        H: Tensor,
        R: Tensor,
    ) -> Tensor:
        """Compute Joseph form update for covariance.

        Σₖ' = (I - KH) Σₖ (I - KH)ᵀ + K R Kᵀ

        Args:
            P: Prior covariance Σₖ of shape (*B, n, n)
            K: Kalman gain K of shape (*B, n, m)
            H: Masked observation matrix H of shape (*B, m, n)
            R: Masked measurement covariance of shape (*B, m, m)

        Returns:
            P_new: Updated covariance Σₖ' of shape (*B, n, n)
        """
        I_KH = self.identity_matrix - einsum("...ik, ...kj -> ...ij", K, H)
        P_new = (
            einsum("...ik, ...kl, ...jl -> ...ij", I_KH, P, I_KH)  # (*B, n, n)
            + einsum("...ik, ...kl, ...jl -> ...ij", K, R, K)  # (*B, n, n)
        )
        return P_new
