r"""Discrete Kalman Filter implementation."""

__all__ = ["ContinuousKalmanFilter"]

from typing import Final

import torch
from numpy.typing import ArrayLike
from torch import Tensor, einsum, nan, nn, stack
from torch.linalg import matrix_exp


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


class ContinuousKalmanFilter(nn.Module):
    r"""Continuous, time-invariant Kalman Filter.

    .. math::
        ∂ₜxₜ &= Fxₜ + wₜ  &  wₜ &~ N(0, Qₜ)  \\
          yₜ &= Hxₜ + vₜ  &  vₜ &~ N(0, Rₜ)
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
        process_noise: ArrayLike | float = 1.0,  # [n, n]
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
        self.register_buffer("prior_latent_means", torch.empty(0, n))
        self.register_buffer("prior_latent_covs", torch.empty(0, n, n))
        self.register_buffer("prior_target_means", torch.empty(0, m))
        self.register_buffer("prior_target_covs", torch.empty(0, m, m))
        self.register_buffer("post_latent_means", torch.empty(0, n))
        self.register_buffer("post_latent_covs", torch.empty(0, n, n))
        self.register_buffer("post_target_means", torch.empty(0, m))
        self.register_buffer("post_target_covs", torch.empty(0, m, m))
        self.register_buffer("identity_matrix", torch.eye(n))
        self.register_buffer("van_loan_matrix", torch.empty(2 * n, 2 * n))
        self.update_buffers()

        # validate model
        self.validate_parameters()
        self.validate_buffers()

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

    def validate_buffers(self) -> None:
        m = self.input_size
        n = self.hidden_size
        bs = self.prior_latent_means.shape[:-1]
        # check shapes
        assert self.process_cov.shape == (n, n)
        assert self.measurement_cov.shape == (m, m)
        assert self.initial_cov.shape == (n, n)
        assert self.prior_latent_means.shape == (*bs, n)
        assert self.prior_latent_covs.shape == (*bs, n, n)
        assert self.prior_target_means.shape == (*bs, m)
        assert self.prior_target_covs.shape == (*bs, m, m)
        assert self.post_latent_means.shape == (*bs, n)
        assert self.post_latent_covs.shape == (*bs, n, n)
        assert self.post_target_means.shape == (*bs, m)
        assert self.post_target_covs.shape == (*bs, m, m)
        assert self.identity_matrix.shape == (n, n)
        assert self.van_loan_matrix.shape == (2 * n, 2 * n)

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

    def forward(
        self,
        times: Tensor,  # (..., $N + $K)
        values: Tensor,  # (..., $N + $K, D)
        context_mask: Tensor,  # (..., $N + $K, D), bool
        query_mask: Tensor,  # (..., $N + $K, D), bool
        *,  # μ₀=(..., D) Σ₀=(..., D, D)
        initial_state: tuple[Tensor, Tensor] | None = None,
        initial_time: Tensor | None = None,  # t₀, ()
    ) -> tuple[Tensor, Tensor]:  # (..., $N + $K, D), (..., $N + $K, D, D)
        r"""Filter and forecast over combined context/query time points.

        Args:
            times: Combined context and query time points.
            values: Sparse observations at context time points.
            context_mask: Boolean mask selecting observed entries in ``values``.
            query_mask: Boolean mask selecting requested forecast entries.
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
        valid_steps = times.isfinite() & (context_mask | query_mask).any(dim=-1)
        mean_mask = valid_steps.unsqueeze(dim=-1)
        cov_mask = mean_mask.unsqueeze(dim=-1)

        if self.batch_first:
            # Move the time axis to the front.
            times = times.moveaxis(-1, 0)
            values = values.moveaxis(-2, 0)
            context_mask = context_mask.moveaxis(-2, 0)
            query_mask = query_mask.moveaxis(-2, 0)
            valid_steps = valid_steps.moveaxis(-1, 0)

        # check the shapes
        m = self.input_size
        n = self.hidden_size
        num_steps, *batch_shape, _ = values.shape
        assert times.shape == (num_steps, *batch_shape)
        assert values.shape == (num_steps, *batch_shape, m)
        assert context_mask.shape == (num_steps, *batch_shape, m)
        assert query_mask.shape == (num_steps, *batch_shape, m)
        assert valid_steps.shape == (num_steps, *batch_shape)
        assert context_mask.dtype == torch.bool
        assert query_mask.dtype == torch.bool

        # initialize the state
        t = times[0] if initial_time is None else initial_time
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
        prior_predicted_means: list[Tensor] = []
        prior_predicted_covs: list[Tensor] = []
        post_latent_means: list[Tensor] = []
        post_latent_covs: list[Tensor] = []
        post_predicted_means: list[Tensor] = []
        post_predicted_covs: list[Tensor] = []

        for t_obs, y_obs, mask, active in zip(
            times,
            values,
            context_mask,
            valid_steps,
            strict=True,
        ):
            # Within the loop we use batch-first.
            delta = torch.where(active, t_obs - t, torch.zeros_like(t_obs - t))
            t = torch.where(active, t_obs, t)

            # propagate forward in time
            x_pre, P_pre = self.propagate_state(x_post, P_post, delta)
            prior_latent_means.append(x_pre)
            prior_latent_covs.append(P_pre)

            # make the prior prediction
            y_pre, S_pre = self.decode_state(x_pre, P_pre)
            prior_predicted_means.append(y_pre)
            prior_predicted_covs.append(S_pre)

            # Update step
            x_update, P_update = self.update_state(x_pre, P_pre, y_obs, y_pre, mask)
            x_post = torch.where(active[..., None], x_update, x_post)
            P_post = torch.where(active[..., None, None], P_update, P_post)
            post_latent_means.append(x_post)
            post_latent_covs.append(P_post)

            # make the posterior prediction
            y_post, S_post = self.decode_state(x_post, P_post)
            post_predicted_means.append(y_post)
            post_predicted_covs.append(S_post)

        stack_dim_mean = -2 if self.batch_first else 0
        stack_dim_cov = -3 if self.batch_first else 0

        self.prior_latent_means = stack(prior_latent_means, dim=stack_dim_mean)
        self.prior_latent_covs = stack(prior_latent_covs, dim=stack_dim_cov)
        self.prior_target_means = stack(prior_predicted_means, dim=stack_dim_mean)
        self.prior_target_covs = stack(prior_predicted_covs, dim=stack_dim_cov)
        self.post_latent_means = stack(post_latent_means, dim=stack_dim_mean)
        self.post_latent_covs = stack(post_latent_covs, dim=stack_dim_cov)
        self.post_target_means = stack(post_predicted_means, dim=stack_dim_mean)
        self.post_target_covs = stack(post_predicted_covs, dim=stack_dim_cov)

        self.prior_latent_means = self.prior_latent_means.masked_fill(~mean_mask, nan)
        self.prior_latent_covs = self.prior_latent_covs.masked_fill(~cov_mask, nan)
        self.prior_target_means = self.prior_target_means.masked_fill(~mean_mask, nan)
        self.prior_target_covs = self.prior_target_covs.masked_fill(~cov_mask, nan)
        self.post_latent_means = self.post_latent_means.masked_fill(~mean_mask, nan)
        self.post_latent_covs = self.post_latent_covs.masked_fill(~cov_mask, nan)
        self.post_target_means = self.post_target_means.masked_fill(~mean_mask, nan)
        self.post_target_covs = self.post_target_covs.masked_fill(~cov_mask, nan)

        self.validate_buffers()

        return self.post_target_means, self.post_target_covs

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
        x: Tensor,
        P: Tensor,
        delta: Tensor,
    ) -> tuple[Tensor, Tensor]:
        r"""Propagate latent mean and covariance through continuous dynamics."""
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

    def decode_state(self, x: Tensor, P: Tensor) -> tuple[Tensor, Tensor]:
        r"""Decode latent mean and covariance to observation space."""
        H = self.observation_matrix
        R = self.measurement_cov
        y = einsum("ij, ...j -> ...i", H, x)  # (*B, m)
        S = R + einsum("ik, ...kl, jl -> ...ij", H, P, H)  # (*B, m, m)
        return y, S

    def update_state(
        self,
        x: Tensor,
        P: Tensor,
        y: Tensor,
        y_pred: Tensor,
        mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        r"""Update latent mean and covariance with a sparse observation."""
        H = self.observation_matrix
        R = self.measurement_cov
        M = mask.to(P.dtype)
        missing = (~mask).to(P.dtype)
        H_masked = M.unsqueeze(-1) * H
        R_masked = M.unsqueeze(-1) * M.unsqueeze(-2) * R + missing.diag_embed()

        # Innovation residual: ignore unobserved coordinates.
        r = torch.where(mask, y - y_pred, torch.zeros_like(y_pred))

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
