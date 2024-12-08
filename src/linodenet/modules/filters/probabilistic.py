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

.. math:: ΣH'(HΣ H' + R)^{-1}(Hx - y)

R is observed or a hyperparameter.
"""

__all__ = [
    # Protocols & ABCs
    "ProbabilisticFilter",
    "DiscreteProbabilisticFilter",
    "SingleValueProbabilisticFilter",
    # Classes
    "probabilistic_kalman_filter",
    "discrete_probabilistic_kalman_filter",
]

from abc import abstractmethod
from typing import Optional, Protocol, runtime_checkable

import torch
from torch import Tensor
from torch.distributions import Distribution, MultivariateNormal

from linodenet.random.distributions.empirical import Dirac, Empirical


@runtime_checkable
class ProbabilisticFilter(Protocol):
    r"""Protocol for probabilistic filters.

    The goal of a probabilistic filter is to update the distribution of the hidden state,
    given the current observation. This is done by updating the parameters of the distribution
    that represents the state.

    In practice, this is done by acting on the parameters that define the distribution model,
    rather than in function space itself.
    """

    @abstractmethod
    def __call__(self, y: Distribution, x: Distribution, /) -> Distribution: ...


class DiscreteProbabilisticFilter(Protocol):
    r"""Protocol for probabilistic filter with discrete observations."""

    @abstractmethod
    def __call__(self, y: Empirical, x: Distribution, /) -> Distribution: ...


class SingleValueProbabilisticFilter(Protocol):
    r"""Protocol for probabilistic filter with single value observations."""

    @abstractmethod
    def __call__(self, y: Dirac, x: Distribution, /) -> Distribution: ...


def probabilistic_kalman_filter(
    y: MultivariateNormal,
    x: MultivariateNormal,
    /,
    H: Optional[Tensor] = None,
) -> MultivariateNormal:
    r"""The classical Kalman Filter, rephrased in probabilistic terms.

    .. math::
        μ' = μ - ΣH'(HΣH' + R)^{-1}(Hμ - y)
        Σ' = Σ - ΣH'(HΣH' + R)^{-1}HΣ

    Note that the formula for the update derives from the conditional distribution of a joint
    normal distribution: If $p(x, y) = 𝓝([μ₁, μ₂], [[Σ₁₁, Σ₁₂], [Σ₂₁, Σ₂₂]])$, then
    $p(x∣y) = 𝓝(μ', Σ')$ where $μ' = μ₁ - Σ₁₂Σ₂₂⁻¹(μ₂ - y)$ and $Σ' = Σ₁₁ - Σ₁₂Σ₂⁻¹Σ₂₁$.

    Args:
        y: The observation distribution.
        x: The state distribution.
        H: The observation matrix.

    Returns:
        The updated state distribution.
    """
    μ, Σ = x.mean, x.covariance_matrix
    y, R = y.mean, y.covariance_matrix

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
    /,
    R: Tensor,
    H: Optional[Tensor] = None,
) -> MultivariateNormal:
    """The classical Kalman Filter, rephrased in probabilistic terms.

    Note that the observation is allowed to be sparse, i.e. contain missing values.
    In this case, the update formula is obtained by marginalizing the observation distribution:

    .. math::
        μ' = μ - ΣH'(HΣH' + R)^{-1}(Hμ - y)
        Σ' = Σ - ΣH'(HΣH' + R)^{-1}HΣ

    References:
        Kalman filter with outliers and missing observations,
        T. Cipra & R. Romera, 1997
        https://link.springer.com/article/10.1007/BF02564705
    """
    y = observation.data  # (..., m)
    mask = torch.isnan(y)  # (..., m)
    H = H[..., mask]
    R = R[..., mask][..., mask, :]

    obs_dist = MultivariateNormal(y, R)
    return probabilistic_kalman_filter(obs_dist, state, H=H)
