r"""Probabilistic filter protocol.

NOTE: WIP, not yet implemented.

Probabilistic Filter
--------------------
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
    "ProbabilisticFilter",
    "EmpiricalFilter",
    "DiracFilter",
    # Classes
    "probabilistic_kalman_filter",
    "discrete_probabilistic_kalman_filter",
]

from abc import abstractmethod
from typing import Optional, Protocol, runtime_checkable

import torch
from torch import Tensor
from torch.distributions import MultivariateNormal

from linodenet.distributions import Dirac, Distribution, Empirical


@runtime_checkable
class ProbabilisticFilter[P: Distribution, Q: Distribution](Protocol):
    r"""Protocol for probabilistic filters.

    The goal of a probabilistic filter is to update the distribution of the hidden state,
    given the current observation. This is done by updating the parameters of the distribution
    that represents the state.

    In practice, this is done by acting on the parameters that define the distribution model,
    rather than in function space itself.
    """

    @abstractmethod
    def __call__(self, y: Q, x: P, /) -> P: ...


class DiracFilter[D: Distribution](ProbabilisticFilter[D, Dirac]):
    r"""Protocol for probabilistic filter with single value observations."""

    @abstractmethod
    def __call__(self, y: Tensor | Dirac, x: D, /) -> D: ...


class EmpiricalFilter[D: Distribution](ProbabilisticFilter[D, Empirical]):
    r"""Protocol for probabilistic filter with discrete observations."""

    @abstractmethod
    def __call__(self, y: Tensor | Empirical, x: D, /) -> D: ...


def probabilistic_kalman_filter(
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


def discrete_probabilistic_kalman_filter(
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
    return probabilistic_kalman_filter(obs_dist, state, H=H)
