r"""Gaussian state updates using conditioning and KL regularization.

The Kalman helpers describe conditioning a Gaussian state on an observation
distribution. The Gaussian updater modules instead pull an observation through
an invertible decoder and minimize a latent Gaussian observation loss with
forward- or reverse-KL anchoring to the prior.

Retention parameters control how much of the prior is preserved. Their
relationship to KL weights depends on the divergence direction.
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

from imtskit.distributions import Dirac
from imtskit.distributions.gaussian import (
    CovarianceType,
    GaussianParams,
    argmin_forward_kl,
    argmin_reverse_kl,
)
from imtskit.mappings import Transform
from imtskit.nn.containers import Constant

from .base import AbstractStateUpdate

type ScalarLike = Tensor | float


def probabilistic_kalman_update(
    obs: MultivariateNormal,
    val: MultivariateNormal,
    /,
    H: Optional[Tensor] = None,
) -> MultivariateNormal:
    r"""Form a Gaussian conditioning update from an observation distribution.

    For prior parameters $(μ, Σ)$ and observation parameters $(y, R)$, the
    intended Kalman update is

    $$\begin{aligned}
        S &= HΣHᵀ + R    \\
        μ' &= μ - ΣHᵀS⁻¹(Hμ - y)  \\
        Σ' &= Σ - ΣHᵀS⁻¹HΣ
    \end{aligned}$$

    These equations follow from conditioning a joint Gaussian: for mean
    $(μ₁, μ₂)$ and covariance blocks $Σᵢⱼ$, conditioning on $y$ gives

    $$\begin{aligned}
        μ' &= μ₁ + Σ₁₂Σ₂₂⁻¹(y - μ₂) \\
        Σ' &= Σ₁₁ - Σ₁₂Σ₂₂⁻¹Σ₂₁
    \end{aligned}$$

    Note:
        This implementation is provisional. With Cholesky-whitened cross-covariance
        $P$, the current covariance contraction computes $PP$ rather than $PᵀP$.
        It therefore does not generally realize the intended covariance update.

    Args:
        obs: Observation distribution with mean $y$ and covariance $R$.
        val: Prior state distribution with mean $μ$ and covariance $Σ$.
        H: Observation matrix. If `None`, use direct observations with $H=𝕀$.

    Returns:
        A Gaussian distribution constructed from the computed mean and covariance.
    """
    μ: Tensor = val.mean
    Σ: Tensor = val.covariance_matrix  # ty: ignore[invalid-assignment]
    y: Tensor = obs.mean
    R: Tensor = obs.covariance_matrix  # ty: ignore[invalid-assignment]

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
    r"""Prototype Kalman update for a point observation with missing coordinates.

    The intended update marginalizes the observation model to the observed
    coordinates, using the corresponding rows of $H$ and submatrix of $R$.

    Note:
        The current implementation selects NaN coordinates and passes the original
        observation vector to `MultivariateNormal`. It does not correctly implement
        the intended sparse update.

    Args:
        observation: Point observation represented by a `Dirac` distribution.
        state: Prior Gaussian state distribution.
        R: Observation noise covariance.
        H: Observation matrix. If `None`, construct an identity observation matrix.

    References:
        - Kalman filter with outliers and missing observations \
          T. Cipra and R. Romera \
          TEST, 6, 379–395, 1997 \
          <https://doi.org/10.1007/BF02564705>
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
    r"""Compute a Gaussian observation update with forward-KL anchoring.

    Let $θ₋ = (μ₋, Σ₋)$ be the current latent Gaussian parameters. The decoder
    must be an invertible map $ϕ$ independent of $θ$, with `inverse(y_obs)`
    returning the latent observation tensor $z = ϕ⁻¹(y_obs)$. Its log-Jacobian
    term is constant in $θ$ and does not affect the minimizer.

    For shared retention $ρ$ with $0 < ρ < 1$, the objective is

    $$
        θ₊ = \argmin_θ -\log 𝓝(z; θ) + λ\KL(𝓝(θ₋) ∥ 𝓝(θ)),
        \qquad λ = \frac{ρ}{1-ρ}
    $$

    The retention is a fraction, not the KL weight itself. Shared retention
    uses the same fraction for the mean and covariance. A pair supplies
    separate fractions $(ρₘ, ρₛ)$. Writing $δ = z - μ₋$, the covariance-form update is

    $$\begin{aligned}
        μ₊ &= ρₘμ₋ + (1-ρₘ)z     \\
        Σ₊ &= ρₛΣ₋ + (1-ρₛ)ρₘδδᵀ
    \end{aligned}$$

    The admissible ranges at evaluation are $ρₘ ∈ [0, 1]$ and $ρₛ ∈ (0, 1]$.
    Unit retention preserves the corresponding prior parameter. Other covariance
    representations are handled by `argmin_forward_kl`.
    """

    retention_mu: nn.Module
    r"""MODULE: Forward-KL retention fraction for the mean."""
    retention_sigma: nn.Module
    r"""MODULE: Forward-KL retention fraction for the covariance."""

    rho_mu: Tensor
    r"""BUFFER: Most recent mean retention $ρₘ$, unset before the first forward call."""
    rho_sigma: Tensor
    r"""BUFFER: Most recent covariance retention $ρₛ$, unset before the first forward call."""

    def __init__(
        self,
        *,
        decoder: nn.Module,  # & Transform[Tensor, Tensor]
        parametrization: str | CovarianceType,
        retention: ScalarLike | tuple[ScalarLike, ScalarLike] = 0.5,
        retention_learnable: bool = True,
    ) -> None:
        super().__init__()

        self.decoder: Transform[Tensor, Tensor] = decoder  # type: ignore
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
    r"""Compute a Gaussian observation update with reverse-KL anchoring.

    Let $θ₋ = (μ₋, Σ₋)$ be the current latent Gaussian parameters. The decoder
    must be an invertible map $ϕ$ independent of $θ$, with `inverse(y_obs)`
    returning the latent observation tensor $z = ϕ⁻¹(y_obs)$. Its log-Jacobian
    term is constant in $θ$ and does not affect the minimizer.

    Writing $θ = (μ, Σ)$ in covariance coordinates, the updated parameters are

    $$ θ₊ = \argmin_θ J(θ; θ₋, z) $$

    For finite anchoring weights, the objective is

    $$\begin{aligned}
        J(θ; θ₋, z) &= -\log 𝓝(z; θ) + \frac{λ_μ}{2}(μ-μ₋)ᵀΣ₋⁻¹(μ-μ₋) \\
            &\quad + \frac{λ_Σ}{2}\left(\tr(ΣΣ₋⁻¹) - \log\det(ΣΣ₋⁻¹) - d\right)
    \end{aligned}$$

    Here $d$ is the latent-state dimension. When $λ_μ = λ_Σ = λ > 1$, this
    reduces to the reverse-KL objective

    $$ θ₊ = \argmin_θ -\log 𝓝(z; θ) + λ\KL(𝓝(θ) ∥ 𝓝(θ₋)) $$

    The parameter `retention` uses the reverse-KL retention coordinates:

    - $ρ_μ ∈ [0, 1]$ controls nominal mean anchoring, with
        $λ_μ = ρ_μ/(1-ρ_μ)$. The realized mean retention also depends on the innovation.
    - $ρ_Σ ∈ (0, 1]$ controls the retained covariance fraction, with $λ_Σ = 1/(1-ρ_Σ)$.
    """

    retention_mu: nn.Module
    r"""MODULE: Reverse-KL nominal mean retention."""
    retention_sigma: nn.Module
    r"""MODULE: Reverse-KL covariance retention."""

    rho_mu: Tensor
    r"""BUFFER: Most recent mean retention $ρₘ$, unset before the first forward call."""
    rho_sigma: Tensor
    r"""BUFFER: Most recent covariance retention $ρₛ$, unset before the first forward call."""

    def __init__(
        self,
        *,
        decoder: nn.Module,  # & Transform[Tensor, Tensor]
        parametrization: str | CovarianceType,
        retention: ScalarLike | tuple[ScalarLike, ScalarLike] = 0.5,
        retention_learnable: bool = True,
    ) -> None:
        super().__init__()

        self.decoder: Transform[Tensor, Tensor] = decoder  # type: ignore
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
