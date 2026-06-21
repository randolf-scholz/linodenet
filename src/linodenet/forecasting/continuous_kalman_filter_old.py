r"""Discrete Kalman Filter implementation."""

__all__ = ["ContinuousKalmanFilter"]

from typing import Final

import scipy
import torch
from numpy.typing import ArrayLike
from torch import Tensor, einsum, nan, nn, stack
from torch.linalg import matrix_exp


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

    # PARAMETERS
    system_matrix: Tensor
    observation_matrix: Tensor
    process_covariance: Tensor
    measurement_covariance: Tensor
    initial_time: Tensor
    initial_mean: Tensor
    initial_covariance: Tensor

    # BUFFERS
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

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        *,
        system_matrix: ArrayLike,  # [n, n]
        observation_matrix: ArrayLike,  # [k, n]
        process_covariance: ArrayLike | float,  # [n, n]
        measurement_covariance: ArrayLike | float,  # [k, k]
        initial_time: float = 0.0,
        initial_mean: ArrayLike | None = None,  # [n]
        initial_covariance: ArrayLike | None = None,  # [n, n]
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
            torch.eye(hidden_size)
            if process_covariance is None
            else process_covariance * torch.eye(hidden_size)
            if isinstance(process_covariance, (float, int))
            else torch.as_tensor(process_covariance),
            requires_grad=learnable,
        )
        self.measurement_covariance = nn.Parameter(
            torch.eye(input_size)
            if measurement_covariance is None
            else measurement_covariance * torch.eye(input_size)
            if isinstance(measurement_covariance, (float, int))
            else torch.as_tensor(measurement_covariance),
            requires_grad=learnable,
        )
        self.initial_time = nn.Parameter(
            torch.tensor(initial_time), requires_grad=False
        )
        self.initial_mean = nn.Parameter(
            torch.zeros(hidden_size)
            if initial_mean is None
            else torch.as_tensor(initial_mean),
            requires_grad=learnable,
        )
        self.initial_covariance = nn.Parameter(
            torch.eye(hidden_size)
            if initial_covariance is None
            else torch.as_tensor(initial_covariance),
            requires_grad=learnable,
        )

        # register buffers
        self.register_buffer("prior_latent_means", torch.empty(0, n))
        self.register_buffer("prior_latent_covariances", torch.empty(0, n, n))
        self.register_buffer("prior_predicted_means", torch.empty(0, m))
        self.register_buffer("prior_predicted_covariances", torch.empty(0, m, m))
        self.register_buffer("posterior_latent_means", torch.empty(0, n))
        self.register_buffer("posterior_latent_covariances", torch.empty(0, n, n))
        self.register_buffer("posterior_predicted_means", torch.empty(0, m))
        self.register_buffer("posterior_predicted_covariances", torch.empty(0, m, m))

        # validate model
        self.validate_parameters()
        self.validate_buffers()

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

        # check that covariance matrices are symmetric positive definite
        assert torch.allclose(Q, Q.transpose(-1, -2)), "Process noise Q not symmetric"
        assert torch.allclose(R, R.transpose(-1, -2)), (
            "Measurement noise R not symmetric"
        )
        assert torch.allclose(P, P.transpose(-1, -2)), (
            "Measurement noise R not symmetric"
        )
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
        """Sample a random system matrix.

        We take an orthogonal matrix for stability.
        """
        orthogonal_matrix = scipy.stats.ortho_group.rvs(dim=self.hidden_size)
        return torch.as_tensor(orthogonal_matrix, dtype=torch.float32)

    @torch.no_grad()
    def _sample_default_observation_matrix(self) -> Tensor:
        r"""Sample a random observation matrix."""
        t = torch.randn(self.input_size, self.hidden_size)
        nn.init.kaiming_uniform_(t)
        return t

    def forward(
        self,
        query: Tensor,  # (..., $K)
        context: tuple[Tensor, Tensor],  # (..., $N), (..., $N, D)
        initial_state: tuple[
            Tensor,  # t₀, ()
            Tensor,  # μ₀, (..., D)
            Tensor,  # Σ₀, (..., D, D)
        ]
        | None = None,
    ) -> tuple[Tensor, Tensor]:  # (..., $K, D), (..., $K, D, D)
        r"""Predict ``n_steps`` into the future given observations.

        Args:
            query: time points at which to forecast
            context: known observations $(t, y)$
            initial_state: initial latent state $(t₀, μ₀, Σ₀)$

        Returns:
            y_pred: Predicted means $μ̂ₜ=\E[ŷₜ]$ for each query time
            S_pred: Predicted covariances $Σ̂ₜ=\Var[ŷₜ]$ for each query time
        """
        n_steps = query.shape[-1]
        times, values = context
        bs = values.shape[:-2] if self.batch_first else values.shape[1:-1]

        F = self.system_matrix
        Q = self.process_covariance
        R = self.measurement_covariance
        H = self.observation_matrix
        m = self.input_size
        n = self.hidden_size

        if initial_state is None:
            t = self.initial_time
            x = self.initial_mean
            P = self.initial_covariance
        else:
            t, x, P = initial_state

        assert x.shape in [(*bs, n), (n,)]
        assert P.shape in [(*bs, n, n), (n, n)]

        if x.shape == (n,):
            x = x.expand(*bs, n)
        if P.shape == (n, n):
            P = P.expand(*bs, n, n)

        assert x.shape == (*bs, n)
        assert P.shape == (*bs, n, n)

        # pre-allocate outputs / variables
        y_pred = values.new_full((*bs, n_steps, m), nan)
        S_pred = values.new_full((*bs, n_steps, m, m), nan)
        x_new = x.clone()
        P_new = P.clone()

        # setup buffers
        prior_latent_means: list[Tensor] = []
        prior_latent_covariances: list[Tensor] = []
        prior_predicted_means: list[Tensor] = []
        prior_predicted_covariances: list[Tensor] = []
        posterior_latent_means: list[Tensor] = []
        posterior_latent_covariances: list[Tensor] = []
        posterior_predicted_means: list[Tensor] = []
        posterior_predicted_covariances: list[Tensor] = []

        M = torch.cat(  # [[F, Q], [0, -Fᵀ]]
            [  # [2n, 2n]
                torch.cat([F, Q], dim=-1),
                torch.cat([torch.zeros_like(F), -F.mT], dim=-1),
            ],
            dim=0,
        )

        valid_context = times.isfinite() & values.isfinite().all(dim=-1)
        t_slices = times.unbind(-1) if self.batch_first else times.unbind(0)
        y_slices = values.unbind(-2) if self.batch_first else values.unbind(0)
        valid_slices = (
            valid_context.unbind(-1) if self.batch_first else valid_context.unbind(0)
        )

        for t_obs, y_obs, valid in zip(t_slices, y_slices, valid_slices, strict=True):
            # Within the loop we use batch-first.
            valid_mean = valid.unsqueeze(dim=-1)
            valid_covariance = valid_mean.unsqueeze(dim=-1)
            delta = torch.where(valid, t_obs - t, torch.zeros_like(t_obs - t))

            # concise implementation with a single matrix exponential. Possibly less efficient.
            expMt = matrix_exp(M * delta[..., None, None])  # [[G, Φ], [0, G⁻ᵀ]]
            G = expMt[..., :n, :n]
            Phi = expMt[..., :n, n:]
            x = einsum("...ij, ...j -> ...i", G, x_new)  # (*B, n)
            P = Phi + einsum("...ik, ...kl, ...jl -> ...ij", G, P_new, G)  # (*B, n, n)
            prior_latent_means.append(
                torch.where(valid_mean, x, torch.full_like(x, nan))
            )
            prior_latent_covariances.append(
                torch.where(valid_covariance, P, torch.full_like(P, nan))
            )

            # Prediction step (use einsum to deal with batch dims)
            y_hat = einsum("ij, ...j -> ...i", H, x)  # (*B, m)
            S = R + einsum("ik, ...kl, jl -> ...ij", H, P, H)  # (*B, m, m)
            prior_predicted_means.append(
                torch.where(valid_mean, y_hat, torch.full_like(y_hat, nan))
            )
            prior_predicted_covariances.append(
                torch.where(valid_covariance, S, torch.full_like(S, nan))
            )

            # Update step
            r = y_obs.nan_to_num(0.0) - y_hat  # innovation (*B, m)
            K = self._compute_kalman_gain(P, H, S)  # (*B, n, m)
            # update mean and covariance
            x_update = x + einsum("...ij, ...j -> ...i", K, r)  # (*B, n)
            P_update = self._joseph_update(P, K, H)  # (*B, n, n)
            x_new = torch.where(valid_mean, x_update, x_new)
            P_new = torch.where(valid_covariance, P_update, P_new)
            t = torch.where(valid, t_obs, t)
            posterior_latent_means.append(
                torch.where(valid_mean, x_new, torch.full_like(x_new, nan))
            )
            posterior_latent_covariances.append(
                torch.where(valid_covariance, P_new, torch.full_like(P_new, nan))
            )

            # compute the posterior predicted mean and covariance
            y_new = einsum("ij, ...j -> ...i", H, x_new)  # (*B, m)
            S_new = R + einsum("ik, ...kl, jl -> ...ij", H, P_new, H)  # (*B, m, m)
            posterior_predicted_means.append(
                torch.where(valid_mean, y_new, torch.full_like(y_new, nan))
            )
            posterior_predicted_covariances.append(
                torch.where(valid_covariance, S_new, torch.full_like(S_new, nan))
            )

        # store buffers
        self.prior_latent_means = stack(prior_latent_means, dim=-2)
        self.prior_latent_covariances = stack(prior_latent_covariances, dim=-3)
        self.prior_predicted_means = stack(prior_predicted_means, dim=-2)
        self.prior_predicted_covariances = stack(prior_predicted_covariances, dim=-3)
        self.posterior_latent_means = stack(posterior_latent_means, dim=-2)
        self.posterior_latent_covariances = stack(posterior_latent_covariances, dim=-3)
        self.posterior_predicted_means = stack(posterior_predicted_means, dim=-2)
        self.posterior_predicted_covariances = stack(
            posterior_predicted_covariances, dim=-3
        )
        self.validate_buffers()

        q_slices = query.unbind(-1) if self.batch_first else query.unbind(0)
        for k, q in enumerate(q_slices):
            # Within the loop we use batch-first.
            valid = q.isfinite()
            valid_mean = valid.unsqueeze(dim=-1)
            valid_covariance = valid_mean.unsqueeze(dim=-1)
            delta = torch.where(valid, q - t, torch.zeros_like(q - t))

            # concise implementation with a single matrix exponential. Possibly less efficient.
            expMt = matrix_exp(M * delta[..., None, None])  # [[G, Φ], [0, G⁻ᵀ]]
            G = expMt[..., :n, :n]
            Phi = expMt[..., :n, n:]
            x = einsum("...ij, ...j -> ...i", G, x_new)  # (*B, n)
            P = einsum("...ik, ...kl, ...jl -> ...ij", G, P_new, G) + Phi  # (*B, n, n)

            # Prediction step (use einsum to deal with batch dims)
            y_hat = einsum("ij, ...j -> ...i", H, x)  # (*B, m)
            S = R + einsum("ik, ...kl, jl -> ...ij", H, P, H)  # (*B, m, m)
            y_pred[..., k, :] = torch.where(
                valid_mean, y_hat, torch.full_like(y_hat, nan)
            )
            S_pred[..., k, :, :] = torch.where(
                valid_covariance, S, torch.full_like(S, nan)
            )

            x_new = torch.where(valid_mean, x, x_new)
            P_new = torch.where(valid_covariance, P, P_new)
            t = torch.where(valid, q, t)

        return y_pred, S_pred

    def _compute_kalman_gain(
        self,
        P: Tensor,
        H: Tensor,
        S: Tensor,
    ) -> Tensor:
        """Compute Kalman gain K.

        K = P Hᵀ S⁻¹

        Args:
            P: Prior covariance Σₖ of shape (*B, n, n)
            H: Observation matrix Hₖ of shape (m, n)
            S: Innovation covariance Sₖ of shape (*B, m, m)

        Returns:
            K: Kalman gain of shape (*B, n, m)
        """
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
    ) -> Tensor:
        """Compute Joseph form update for covariance.

        Σₖ' = (I - KH) Σₖ (I - KH)ᵀ + K R Kᵀ

        Args:
            P: Prior covariance Σₖ of shape (*B, n, n)
            K: Kalman gain K of shape (*B, n, m)
            H: Observation matrix H of shape (m, n)

        Returns:
            P_new: Updated covariance Σₖ' of shape (*B, n, n)
        """
        R = self.measurement_covariance
        I = torch.eye(self.hidden_size, device=P.device)
        I_KH = I - einsum("...ik, kj -> ...ij", K, H)  # (*B, n, n)
        P_new = (
            einsum("...ik, ...kl, ...jl -> ...ij", I_KH, P, I_KH)  # (*B, n, n)
            + einsum("...ik, kl, ...jl -> ...ij", K, R, K)  # (*B, n, n)
        )
        return P_new
