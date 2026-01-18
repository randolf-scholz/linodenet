r"""Discrete Kalman Filter implementation."""

__all__ = ["DiscreteKalmanFilter"]

from typing import Final, Optional

import scipy
import torch
from numpy.typing import ArrayLike
from torch import Tensor, einsum, jit, nn, stack

from linodenet.signatures import signature


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

    @jit.export
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

    @jit.export
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

    @signature(
        "[int[q], (..., *n, d), Optional[(..., d), (..., d, d)]] "
        "-> [(..., *q, d), (..., *q, d, d)]"
    )
    def forward(
        self,
        n_steps: int,
        y_obs: Tensor,
        initial_state: Optional[tuple[Tensor, Tensor]] = None,
    ) -> tuple[Tensor, Tensor]:
        r"""Predict ``n_steps`` into the future given observations.

        Args:
            n_steps: Number of steps to predict into the future
            y_obs: Input sequence $(y₁, ..., yₙ)$
            initial_state: Initial hidden state $(μ₀, Σ₀)$

        Returns:
            y_pred: Predicted means $μ̂ₖ=\E[ŷₖ]$ for $k=1,...,m$
            S_pred: Predicted covariances $Σ̂ₖ=\Var[ŷₖ]$ for $k=1,...,m$
        """
        F = self.system_matrix
        Q = self.process_covariance
        R = self.measurement_covariance
        H = self.observation_matrix
        m = self.input_size
        n = self.hidden_size
        bs = y_obs.shape[:-2]
        device = self.system_matrix.device

        if initial_state is None:
            x = self.initial_mean
            P = self.initial_covariance
        else:
            x, P = initial_state

        assert x.shape in [(*bs, n), (n,)]
        assert P.shape in [(*bs, n, n), (n, n)]

        if x.shape == (n,):
            x = x.expand(*bs, n)
        if P.shape == (n, n):
            P = P.expand(*bs, n, n)

        assert x.shape == (*bs, n)
        assert P.shape == (*bs, n, n)

        # pre-allocate outputs / variables
        y_pred = torch.empty(*bs, n_steps, m, device=device)
        S_pred = torch.empty(*bs, n_steps, m, m, device=device)
        K = torch.empty(*bs, n, m, device=device)
        S = torch.empty(*bs, m, m, device=device)
        r = torch.empty(*bs, m, device=device)
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

        slices = y_obs.unbind(-2) if self.batch_first else y_obs.unbind(0)

        for y in slices:
            # Within the loop we use batch-first.

            # propagate to next step
            x = einsum("ij, ...j -> ...i", F, x_new)  # (*B, n)
            P = Q + einsum("ik, ...kl, jl -> ...ij", F, P_new, F)  # (*B, n, n)
            prior_latent_means.append(x)
            prior_latent_covariances.append(P)

            # Prediction step (use einsum to deal with batch dims)
            y_hat = einsum("ij, ...j -> ...i", H, x)  # (*B, m)
            S = R + einsum("ik, ...kl, jl -> ...ij", H, P, H)  # (*B, m, m)
            prior_predicted_means.append(x)
            prior_predicted_covariances.append(S)

            # Update step
            r = y - y_hat  # innovation (*B, m)
            K = self._compute_kalman_gain(P, H, S)  # (*B, n, m)
            # update mean and covariance
            x_new = x + einsum("...ij, ...j -> ...i", K, r)  # (*B, n)
            P_new = self._joseph_update(P, K, H)  # (*B, n, n)
            posterior_latent_means.append(x_new)
            posterior_latent_covariances.append(P_new)

            # compute the posterior predicted mean and covariance
            y_new = einsum("ij, ...j -> ...i", H, x_new)  # (*B, m)
            S_new = R + einsum("ik, ...kl, jl -> ...ij", H, P_new, H)  # (*B, m, m)
            posterior_predicted_means.append(y_new)
            posterior_predicted_covariances.append(S_new)

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

        for t in range(n_steps):
            # propagate to next step
            x = einsum("ij, ...j -> ...i", F, x_new)  # (*B, n)
            P = Q + einsum("ik, ...kl, jl -> ...ij", F, P_new, F)  # (*B, n, n)

            # Prediction step (use einsum to deal with batch dims)
            y_hat = einsum("ij, ...j -> ...i", H, x)  # (*B, m)
            S = R + einsum("ik, ...kl, jl -> ...ij", H, P, H)  # (*B, m, m)
            y_pred[..., t, :] = y_hat
            S_pred[..., t, :, :] = S

            x_new = x
            P_new = P

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
            Kt = torch.linalg.solve(S.transpose(-2, -1), H @ P)  # (*B, m, n)
            K = Kt.transpose(-2, -1)  # (*B, n, m)

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
