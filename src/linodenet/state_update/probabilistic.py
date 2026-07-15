r"""Probabilistic state-update protocol.

Probabilistic State Update
--------------------------
- We need sampling distribution. What if we have multiple independent measurements
  of the same quantity at time t?
  - ⟹ We approximate the observational distribution.
  - Option 1: Empirical distribution
  - Option 2: Posteriors from a Bayesian model

Study Kalman Filter from a probabilistic perspective.
Transfer observations to the latent linear state.

Experiment with parametrized KalmanCell.

.. math:: ΣH'(HΣ H' + R)⁻¹(Hx - y)

R is observed or a hyperparameter.
"""

__all__ = [
    # Classes
    "GaussianForwardUpdater",
    "GaussianReverseUpdater",
    # functions
    "probabilistic_kalman_update",
    "discrete_probabilistic_kalman_update",
]


from typing import Any, Optional

import torch
from torch import Tensor, nn
from torch.distributions import MultivariateNormal

from linodenet.distributions import Dirac
from linodenet.distributions.gaussian import (
    CovarianceType,
    GaussianParams,
    argmin_forward_kl,
    argmin_reverse_kl,
)
from linodenet.mappings import Transform
from linodenet.nn.containers import Constant

from .base import AbstractStateUpdate

type ScalarLike = Tensor | float


def probabilistic_kalman_update(
    obs: MultivariateNormal,
    val: MultivariateNormal,
    /,
    H: Optional[Tensor] = None,
) -> MultivariateNormal:
    r"""The classical Kalman Filter, rephrased in probabilistic terms.

    .. math::
        μ' = μ - ΣH'(HΣH' + R)⁻¹(Hμ - y)
        Σ' = Σ - ΣH'(HΣH' + R)⁻¹HΣ

    Note that the formula for the update derives from the conditional distribution of a joint
    normal distribution: If $p(x, y) = 𝓝([μ₁, μ₂], [[Σ₁₁, Σ₁₂], [Σ₂₁, Σ₂₂]])$, then
    $p(x∣y) = 𝓝(μ', Σ')$ where $μ' = μ₁ - Σ₁₂Σ₂₂⁻¹(μ₂ - y)$ and $Σ' = Σ₁₁ - Σ₁₂Σ₂⁻¹Σ₂₁$.

    Args:
        obs: The observation distribution.
        val: The state distribution.
        H: The observation matrix.

    Returns:
        The updated state distribution.
    """
    μ: Tensor = val.mean
    Σ: Tensor = val.covariance_matrix
    y: Tensor = obs.mean
    R: Tensor = obs.covariance_matrix

    if H is None:
        yhat = μ
        S = Σ
        Q = S + R
    else:
        yhat = torch.einsum("ij, ...j -> ...i", H, μ)
        S = torch.einsum("mj, jn -> mn", H, Σ)  # S=HΣ
        Q = torch.einsum("mj, nj -> mn", S, H) + R  # Q=HΣH' + R

    # perform cholesky decomposition (stability + ensure psd cov update)
    L = torch.linalg.cholesky(Q)
    # NOTE: compute ΣH'Q⁻¹r = ΣH'(LLᵀ)⁻¹r = ΣH'L⁻ᵀ(L⁻¹r)
    #   so we solve: Lz = r and L⋅P = HΣ
    #   then the update are ∆μ = Pᵀz and ∆Σ = PᵀP
    z = torch.linalg.solve_triangular(L, yhat - y, upper=False)
    P = torch.linalg.solve_triangular(L, S, upper=False)

    # perform the update
    μ = μ - torch.einsum("ji, ...j -> ...i", P, z)
    Σ = Σ - torch.einsum("mj, jn -> mn", P, P)

    return MultivariateNormal(μ, Σ)


def discrete_probabilistic_kalman_update(
    observation: Dirac,
    state: MultivariateNormal,
    *,
    R: Tensor,
    H: Optional[Tensor] = None,
) -> MultivariateNormal:
    """The classical Kalman Filter, rephrased in probabilistic terms.

    Note that the observation is allowed to be sparse, i.e. contain missing values.
    In this case, the update formula is obtained by marginalizing the observation distribution:

    .. math::
        μ' = μ - ΣH'(HΣH' + R)⁻¹(Hμ - y)
        Σ' = Σ - ΣH'(HΣH' + R)⁻¹HΣ

    References:
        Kalman filter with outliers and missing observations,
        T. Cipra & R. Romera, 1997
        https://link.springer.com/article/10.1007/BF02564705
    """
    y = observation.data  # (..., m)
    mask = torch.isnan(y)  # (..., m)

    if H is None:
        H = torch.eye(state.event_shape[-1], device=state.mean.device)

    H = H[..., mask]  # (..., n, m_obs)
    R = R[..., mask][..., mask, :]  # (..., m_obs, m_obs)

    obs_dist = MultivariateNormal(y, R)
    return probabilistic_kalman_update(obs_dist, state, H=H)


def _make_retention_modules(
    retention: ScalarLike | tuple[ScalarLike, ScalarLike],
    /,
    *,
    learnable: bool,
) -> tuple[nn.Module, nn.Module]:
    r"""Construct retention modules with shared or split learnable logits."""
    match retention:
        case [rho_mu, rho_sigma]:
            strength_mu = torch.as_tensor(rho_mu)
            strength_sigma = torch.as_tensor(rho_sigma)
            if not torch.all((strength_mu >= 0.0) & (strength_mu <= 1.0)):
                raise ValueError(f"ρ_mu must be in [0, 1], got {rho_mu!r}")
            if not torch.all((strength_sigma > 0.0) & (strength_sigma <= 1.0)):
                raise ValueError(f"ρ_sigma must be in (0, 1], got {rho_sigma!r}")

            retention_mu = nn.Sequential(
                Constant(torch.logit(strength_mu), learnable=learnable),
                nn.Sigmoid(),
            )
            retention_sigma = nn.Sequential(
                Constant(torch.logit(strength_sigma), learnable=learnable),
                nn.Sigmoid(),
            )
            return retention_mu, retention_sigma

        case rho:
            strength = torch.as_tensor(rho)
            if not torch.all((strength >= 0.0) & (strength <= 1.0)):
                raise ValueError(f"ρ must be in [0, 1], got {rho!r}")
            shared_retention = nn.Sequential(
                Constant(torch.logit(strength), learnable=learnable),
                nn.Sigmoid(),
            )
            return shared_retention, shared_retention


class GaussianForwardUpdater(nn.Module, AbstractStateUpdate[GaussianParams, Tensor]):
    r"""Perform an exact Gaussian forward-KL update of the observation loss.

    .. math:: θ₊ = \argmin_θ -\log q(y_obs∣θ) + γ⋅\kl(𝓝(θ₋)，𝓝(θ))

    Let $θ₋$ denote the current latent Gaussian parameters and $q(y_obs∣θ)$
    be the predictive density induced by the decoder.
    The decoder is used only to pull the observation back into latent space:
    if $(z, \log|\det 𝐃ϕ⁻¹(y_obs)|) = ϕ⁻¹(y_obs)$, then the Jacobian term
    is constant with respect to $θ$, so the minimizer is exactly

    .. math:: θ₊ = \argmin_θ -\log 𝓝(z; θ) + γ⋅\kl(𝓝(θ₋, Σ₋)，𝓝(θ))

    which is evaluated in closed form by `argmin_forward_kl`.

    The parameter ``retention`` is the forward-KL weight $ρ∈[0,1]$:

    - larger $ρ$ keeps $θ₊$ closer to $θ₋$
    - smaller $ρ$ lets the observation move the posterior more aggressively
    """

    retention_mu: nn.Module
    r"""MODULE: the forward-KL weight for the mean."""
    retention_sigma: nn.Module
    r"""MODULE: the forward-KL weight for the covariance."""

    rho_mu: Tensor
    r"""BUFFER: the most recent computed ρ_mu."""
    rho_sigma: Tensor
    r"""BUFFER: the most recent computed ρ_sigma."""

    def __init__(
        self,
        *,
        decoder: nn.Module,  # & Transform[Tensor, Tensor]
        parametrization: str | CovarianceType,
        retention: ScalarLike | tuple[ScalarLike, ScalarLike] = 0.5,
        retention_learnable: bool = True,
    ) -> None:
        super().__init__()

        self.decoder: Transform[Tensor, Tensor] = decoder  # type: ignore[assignment]
        self.parametrization = CovarianceType(parametrization)
        self.retention_mu, self.retention_sigma = _make_retention_modules(
            retention,
            learnable=retention_learnable,
        )

        if not retention_learnable:
            # freeze the module (mark parameters as frozen)
            for param in self.retention_mu.parameters():
                param.requires_grad = False
            for param in self.retention_sigma.parameters():
                param.requires_grad = False

        # buffers for rho_mu, rho_sigma
        self.register_buffer("rho_mu", None, persistent=False)
        self.register_buffer("rho_sigma", None, persistent=False)

    def forward(
        self, y_obs: Tensor, theta: GaussianParams, /, *, context: Any | None = None
    ) -> GaussianParams:
        r"""Return the exact forward-KL Gaussian update $θ₊$."""
        z = self.decoder.inverse(y_obs)

        self.rho_mu = self.retention_mu(context)
        self.rho_sigma = self.retention_sigma(context)

        return argmin_forward_kl(
            z,
            theta,
            retention=(self.rho_mu, self.rho_sigma),
            parametrization=self.parametrization,
        )


class GaussianReverseUpdater(nn.Module, AbstractStateUpdate[GaussianParams, Tensor]):
    r"""Perform an exact Gaussian reverse-KL update of the observation loss.

    .. math:: θ₊ = \argmin_θ -\log q(y_obs∣θ) + \kl(𝓝(θ)，𝓝(θ₋))

    Let $θ₋$ denote the current latent Gaussian parameters and $q(y_obs∣θ)$
    be the predictive density induced by the decoder.
    The decoder is used only to pull the observation back into latent space:
    if $(z, \log|\det 𝐃ϕ⁻¹(y_obs)|) = ϕ⁻¹(y_obs)$, then the Jacobian term
    is constant with respect to $θ$, so the minimizer is exactly

    .. math:: θ₊ = \argmin_θ -\log 𝓝(z; θ) + \kl(𝓝(θ)，𝓝(θ₋))

    which is evaluated in closed form by `argmin_reverse_kl`.

    The parameter ``retention`` uses the reverse-KL retention coordinates:

    - $ρ_μ ∈ [0, 1]$ nominally controls how strongly the mean is anchored
    - $ρ_Σ ∈ (0, 1]$ controls the retained covariance fraction
    """

    retention_mu: nn.Module
    r"""MODULE: the reverse-KL nominal retention for the mean."""
    retention_sigma: nn.Module
    r"""MODULE: the reverse-KL retention for the covariance."""

    rho_mu: Tensor
    r"""BUFFER: the most recent computed ρ_mu."""
    rho_sigma: Tensor
    r"""BUFFER: the most recent computed ρ_sigma."""

    def __init__(
        self,
        *,
        decoder: nn.Module,  # & Transform[Tensor, Tensor]
        parametrization: str | CovarianceType,
        retention: ScalarLike | tuple[ScalarLike, ScalarLike] = 0.5,
        retention_learnable: bool = True,
    ) -> None:
        super().__init__()

        self.decoder: Transform[Tensor, Tensor] = decoder  # type: ignore[assignment]
        self.parametrization = CovarianceType(parametrization)
        self.retention_mu, self.retention_sigma = _make_retention_modules(
            retention,
            learnable=retention_learnable,
        )

        if not retention_learnable:
            for param in self.retention_mu.parameters():
                param.requires_grad = False
            for param in self.retention_sigma.parameters():
                param.requires_grad = False

        self.register_buffer("rho_mu", None, persistent=False)
        self.register_buffer("rho_sigma", None, persistent=False)

    def forward(
        self, y_obs: Tensor, theta: GaussianParams, /, *, context: Any | None = None
    ) -> GaussianParams:
        r"""Return the exact reverse-KL Gaussian update $θ₊$."""
        z = self.decoder.inverse(y_obs)

        self.rho_mu = self.retention_mu(context)
        self.rho_sigma = self.retention_sigma(context)

        return argmin_reverse_kl(
            z,
            theta,
            retention=(self.rho_mu, self.rho_sigma),
            parametrization=self.parametrization,
        )
