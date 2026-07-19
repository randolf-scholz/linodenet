r"""Discrete Kalman Filter implementation."""

__all__ = ["DiscreteKalmanFilter"]

from typing import Final

import torch
from numpy.typing import ArrayLike
from torch import Tensor, einsum, nan, nn, stack

from .kalman_filter import (
    _as_covariance,
    _as_mean,
    marginal_gaussian_log_prob,
    marginal_gaussian_sample,
    marginal_gaussian_sample_and_log_prob,
)
from .utils import DiscreteTimeEventBatch


class DiscreteKalmanFilter(nn.Module):
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
        query_steps: Tensor,  # Long[..., K], padded arbitrary, non-decreasing
        query_mask: Tensor,  # Bool[..., K, D], padded False
        context_steps: Tensor,  # Long[..., N], padded arbitrary, non-decreasing
        context_mask: Tensor,  # Bool[..., N, D], padded False
        context_values: Tensor,  # Float[..., N, D], padded NaN, sparse
        # μ₀=(..., d) Σ₀=(..., d, d)
        initial_state: tuple[Tensor, Tensor] | None = None,
        initial_step: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:  # (..., K, D), (..., K, D, D)
        r"""Compute posterior predictive means and covariances at query steps."""
        combined = DiscreteTimeEventBatch.from_request(
            context_steps=context_steps,
            context_values=context_values,
            context_mask=context_mask,
            query_steps=query_steps,
            query_mask=query_mask,
            batch_first=self.batch_first,
        )
        post_means, post_covs = self.forward(
            steps=combined.steps,
            context_values=combined.context_values,
            context_mask=combined.context_mask,
            query_mask=combined.query_mask,
            initial_state=initial_state,
            initial_step=initial_step,
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

        self.pred_means = pred_means[combined.query_indices]
        self.pred_covs = pred_covs[combined.query_indices]
        return self.pred_means, self.pred_covs

    def forward(
        self,
        *,
        steps: Tensor,  # Long[..., T], padded arbitrary, non-decreasing
        query_mask: Tensor,  # (..., T, D), bool, padded False
        context_values: Tensor,  # (..., T, D), float, padded NaN, sparse
        context_mask: Tensor,  # (..., T, D), bool, padded False
        # μ₀=(..., d) Σ₀=(..., d, d)
        initial_state: tuple[Tensor, Tensor] | None = None,
        initial_step: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:  # (..., T, d), (..., T, d, d)
        r"""Compute posterior latent states over joint discrete event steps.

        Integer step gaps propagate by that many transition steps.
        """
        valid_steps = (context_mask | query_mask).any(dim=-1)
        mean_mask = valid_steps.unsqueeze(dim=-1)
        cov_mask = mean_mask.unsqueeze(dim=-1)
        if steps.dtype != torch.long:
            raise TypeError("Discrete Kalman step indices must be Long tensors.")
        if initial_step is not None and initial_step.dtype != torch.long:
            raise TypeError("Discrete Kalman initial step must be a Long tensor.")

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
        r"""Compute the time-marginal log-likelihood of the model."""
        mean, cov = self.predict(
            query_steps=query_steps,
            query_mask=query_mask,
            context_steps=context_steps,
            context_values=context_values,
            context_mask=context_mask,
            initial_state=initial_state,
            initial_step=initial_step,
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
        query_steps: Tensor,  # Long[..., K], padded arbitrary, non-decreasing
        query_mask: Tensor,  # Bool[..., K, D], padded False
        context_steps: Tensor,  # Long[..., N], padded arbitrary, non-decreasing
        context_values: Tensor,  # Float[..., N, D], padded NaN, sparse
        context_mask: Tensor,  # Bool[..., N, D], padded False
        initial_state: tuple[Tensor, Tensor] | None = None,
        initial_step: Tensor | None = None,
    ) -> Tensor:  # (*S, ..., K, D)
        r"""Sample from the time-marginal predictive distribution."""
        sample_shape = (size,) if isinstance(size, int) else size
        mean, cov = self.predict(
            query_steps=query_steps,
            query_mask=query_mask,
            context_steps=context_steps,
            context_values=context_values,
            context_mask=context_mask,
            initial_state=initial_state,
            initial_step=initial_step,
        )
        return marginal_gaussian_sample(
            sample_shape,
            mean=mean,
            cov=cov,
            mask=query_mask,
        )

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
    ) -> tuple[Tensor, Tensor]:  # (*S, ..., K, D), (*S, ..., K)
        r"""Sample and score from the time-marginal predictive distribution."""
        sample_shape = (size,) if isinstance(size, int) else size
        mean, cov = self.predict(
            query_steps=query_steps,
            query_mask=query_mask,
            context_steps=context_steps,
            context_values=context_values,
            context_mask=context_mask,
            initial_state=initial_state,
            initial_step=initial_step,
        )
        return marginal_gaussian_sample_and_log_prob(
            sample_shape,
            mean=mean,
            cov=cov,
            mask=query_mask,
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
