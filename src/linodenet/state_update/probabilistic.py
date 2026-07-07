r"""Probabilistic state-update protocol.

NOTE: WIP, not yet implemented.

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
    # Protocols & ABCs
    "ProbabilisticStateUpdate",
    "EmpiricalStateUpdate",
    "DiracStateUpdate",
    # Classes
    "NaturalGaussianUpdater",
    "probabilistic_kalman_update",
    "discrete_probabilistic_kalman_update",
]

from abc import abstractmethod
from math import expm1, log
from typing import Optional, Protocol, runtime_checkable

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.distributions import MultivariateNormal

from linodenet.distributions import Dirac, Distribution, Empirical
from linodenet.distributions.gaussian import (
    multivariate_gaussian_log_likelihood,
)
from linodenet.mappings import Transform


@runtime_checkable
class ProbabilisticStateUpdate[P: Distribution, Q: Distribution](Protocol):
    r"""Protocol for probabilistic state updates.

    The goal of a probabilistic state update is to update the distribution of the hidden state,
    given the current observation. This is done by updating the parameters of the distribution
    that represents the state.

    In practice, this is done by acting on the parameters that define the distribution model,
    rather than in function space itself.
    """

    @abstractmethod
    def __call__(self, y: Q, x: P, /) -> P: ...


class DiracStateUpdate[D: Distribution](ProbabilisticStateUpdate[D, Dirac]):
    r"""Protocol for probabilistic state update with single-value observations."""

    @abstractmethod
    def __call__(self, y: Tensor | Dirac, x: D, /) -> D: ...


class EmpiricalStateUpdate[D: Distribution](ProbabilisticStateUpdate[D, Empirical]):
    r"""Protocol for probabilistic state update with discrete observations."""

    @abstractmethod
    def __call__(self, y: Tensor | Empirical, x: D, /) -> D: ...


class NaturalGaussianUpdater(nn.Module):
    r"""Closed-form proximal update for a Gaussian latent state.

    Assumptions:
        1. The latent distribution is Gaussian: $x ∼ 𝓝(μ, Σ)$.
        2. The decoder is an invertible transform $y = h(x)$ providing
           `decode_and_logabsdet`.
        3. We observe a single Dirac value $y$.
        4. The Gaussian parameters are passed explicitly as the tuple $(μ, Σ)$.

    Let

    .. math:: z = h⁻¹(y), \qquad \log p_θ(y) = \log 𝓝(z; μ, Σ) + \log|\det\frac{∂z}{∂y}|

    Since the Jacobian term is independent of $(μ, Σ)$, this module computes the
    exact minimizer of the KL-regularized objective

    .. math:: \min_{μ', Σ'} -\log 𝓝(z; μ', Σ') + λ⋅\mathrm{KL}(𝓝(μ, Σ) ∣ 𝓝(μ', Σ'))

    Writing $η = (1 + λ)⁻¹$ and $δ = z - μ$, the unique Gaussian minimizer is

    .. math:: μ' = (1-η)μ + η z \qquad Σ' = (1-η)Σ + η(1-η)δδᵀ
    """

    decoder: Transform[Tensor, Tensor]
    r"""Decoder used to pull observations back to latent space."""
    raw_lambda: Tensor
    r"""Unconstrained parameter whose softplus defines the positive $λ$."""
    log_prob: Tensor
    r"""BUFFER: The most recent predictive log-likelihood $log p_θ(y)$."""

    def __init__(
        self,
        *,
        decoder: Transform[Tensor, Tensor],
        lambda_init: float = 1.0,
    ) -> None:
        super().__init__()
        if lambda_init <= 0:
            raise ValueError(f"Expected lambda_init > 0, got {lambda_init}.")
        self.decoder = decoder
        raw_lambda = log(expm1(lambda_init))
        self.raw_lambda = nn.Parameter(torch.tensor(raw_lambda))
        self.register_buffer("log_prob", torch.empty(()), persistent=False)

    @property
    def lambda_(self) -> Tensor:
        r"""Return the positive regularization parameter $λ$."""
        return F.softplus(self.raw_lambda) + torch.finfo(self.raw_lambda.dtype).eps

    def forward(
        self,
        y: Tensor,
        params: tuple[Tensor, Tensor],
        /,
    ) -> tuple[Tensor, Tensor]:
        r"""Return the updated Gaussian parameters $(μ', Σ')$."""
        mu, sigma = params
        if mu.ndim < 1:
            raise ValueError(
                f"Expected μ to have at least one dimension, got {mu.shape}."
            )
        if sigma.ndim < 2 or sigma.shape[-2:] != (mu.shape[-1], mu.shape[-1]):
            raise ValueError(
                "Expected Σ to have shape (..., d, d) matching μ.shape[-1], "
                f"got μ.shape={mu.shape} and Σ.shape={sigma.shape}."
            )
        if mu.shape[:-1] != sigma.shape[:-2]:
            raise ValueError(
                "Expected μ and Σ to share the same batch shape, "
                f"got μ.shape={mu.shape} and Σ.shape={sigma.shape}."
            )

        # Pull back y ↦ z so p_θ(y) = 𝓝(z; μ, Σ) · │det ∂z/∂y│.
        z, logabsdet = self.decoder.decode_and_logabsdet(y)
        self.log_prob = (
            multivariate_gaussian_log_likelihood(
                z,
                mean=mu,
                covariance_matrix=sigma,
            )
            + logabsdet
        )

        eta = (1 + self.lambda_).reciprocal()
        delta = z - mu
        outer = torch.einsum("...i, ...j -> ...ij", delta, delta)

        # Exact proximal solution: μ' = (1-η)μ + ηz and Σ' = (1-η)Σ + η(1-η)δδᵀ.
        mu_new = mu + eta * delta
        sigma_new = (1 - eta) * sigma + eta * (1 - eta) * outer
        return mu_new, sigma_new


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
