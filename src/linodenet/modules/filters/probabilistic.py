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
    "ProbabilisticFilter",
    "Empirical",
    "KalmanCell",
]

from abc import abstractmethod
from typing import Optional, Protocol, runtime_checkable

import torch
from torch import Tensor, nn
from torch.distributions import Distribution, MultivariateNormal

from linodenet.constants import EMPTY_SHAPE


@runtime_checkable
class ProbabilisticFilter(Protocol):
    r"""Protocol for probabilistic filters.

    The goal of a probabilistic filter is to update the distribution of the hidden state,
    given the current observation. This is done by updating the parameters of the distribution
    that represents the state.

    In practice, this is done by acting on the parameters that define the distribution model,
    rather than in function space itself.
    """

    decoder: Distribution
    r"""The model for the conditional distribution $p(y|x)$."""

    @abstractmethod
    def __call__(self, y: Distribution, x: Distribution, /) -> Distribution:
        r"""Forward pass of the filter $p' = F(q, p)$."""
        ...


class Empirical(Distribution):
    r"""The empirical distribution.

    .. math:: p(x) = \frac{1}{n} ∑_{i=1}^n δ(x - xᵢ)
    """

    data: Tensor
    r"""CONST: The dataset that defines the empirical distribution."""

    def __init__(self, data: Tensor, /) -> None:
        r"""Initialize the empirical distribution."""
        super().__init__()
        self.data = data  # shape: (n, ...)
        self.n = data.shape[0]
        self.shape = data.shape[1:]

    def sample(self, sample_shape: torch.Size = EMPTY_SHAPE) -> Tensor:
        r"""Sample from the empirical distribution."""
        idx = torch.randint(self.n, sample_shape)
        return self.data[idx]

    def log_prob(self, value: Tensor) -> Tensor:
        r"""Log probability of the empirical distribution.

        Formally, we set δ(0) = ∞ and δ(x) = 0 for x ≠ 0.

        .. Signature: ``[...] -> [...]``.
        """
        # value.shape: (B, ...),
        assert value.shape[1:] == self.shape
        mask = value.unsqueeze(1) == self.data  # shape: (B, n, ...)
        mask = mask.any(dim=1)
        return torch.where(mask, torch.inf, -torch.inf)


class KalmanCell(nn.Module):
    r"""The classical Kalman Filter.

    .. math::
        μ' = μ - ΣH'(HΣH' + R)^{-1}(Hμ - y)
        Σ' = Σ - ΣH'(HΣH' + R)^{-1}HΣ

    The formula for the update derives from the conditional distribution of a joint
    normal distribution:

    .. math::
        p(x, y) &= N([μ₁, μ₂], [[Σ₁₁, Σ₁₂], [Σ₂₁, Σ₂₂]]) \\
        ⟹ p(x∣y) &= N(μ', Σ') \\
            μ' &= μ₁ - Σ₁₂Σ₂₂⁻¹(μ₂ - y) \
            Σ' &= Σ₁₁ - Σ₁₂Σ₂⁻¹Σ₂₁) \\

    plugging in $μ₁ = μ, μ₂ = Hμ, Σ₁₁ = Σ, Σ₁₂ = ΣHᵀ, Σ₂₂ = HΣHᵀ + R$ grants the result.

    The classical Kalman Filter can natively deal both with missing values, and duplicate observations.

    The filter can be modified so that it natively deals with missing values.
    This is possible since the normal distribution can be analytically marginalized.

    The Kalman Filter assumes that all variables are normally distributed.

    In particular, the classical filter assumes that the observation $y$ comes from
    a normal distribution $N(y, R)$.

    On the other hand, given prior assumption $y∼N(0,R)$ and empirical observations
    y₁, ..., yₙ, we get the following posterior distribution for $y$, via Bayes' rule:

    p(y) = N(μ, Σ)  ⟹  p(y|y₁, ..., yₙ) = N(μ', Σ')
    where μ' = (Σ⁻¹ + R⁻¹)⁻¹ (Σ⁻¹μ + R⁻¹ȳ) and Σ' = (Σ⁻¹ + R⁻¹)⁻¹
    """

    # PARAMETERS
    H: Optional[Tensor]
    r"""PARAM: the observation matrix."""
    R: Tensor
    r"""PARAM: The observation noise covariance matrix."""

    def forward(self, y: Tensor, x: MultivariateNormal) -> MultivariateNormal:
        r"""Forward pass of the Kalman Cell."""
        mu, sigma = x.mean, x.covariance_matrix
        mask = torch.isnan(y)

        yhat = self.H @ mu
        residual = yhat - y

        mask = ~torch.isnan(y)  # (..., m)
        z = self.h(x)  # (..., m)
        z = torch.where(mask, z - y, self.ZERO)  # (..., m)
        z = torch.einsum("ij, ...j -> ...i", self.A, z)
        z = torch.where(mask, z, self.ZERO)  # (..., m)
        z = self.ht(z)  # (..., n)
        z = torch.einsum("ij, ...j -> ...i", self.B, z)
        return x + self.epsilon * self.layers(z)

        # Update the state
        # ΣH = Σ @ self.H.T
        # HΣ = self.H @ Σ
        # K = ΣH @ torch.inverse(HΣ + self.R)
        # x = x + K @ (y - self.H @ x)
        # Σ = Σ - K @ HΣ
        # return x, Σ
