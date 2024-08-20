r"""Models of the LinODE-Net package."""

__all__ = [
    "ForecastingModel",
    "ProbabilisticForecastingModel",
]

from abc import abstractmethod
from typing import Optional, Protocol

from torch import Tensor
from torch.distributions import Distribution


class ForecastingModel(Protocol):
    r"""Protocol for (point)-forecasting models.

    A point-forecasting model is a model that predicts a single value for a future
    time step, given the past observations.

    .. math::   ŷ(t，S，M) = model(t, S, M)

    The inputs to the model are:

      - $t = (t₁, t₂, …, tₙ)$ is the time index of the future time steps.
      - $S = ((τ₁, x₁), (τ₂, x₂), …, (τₘ, xₘ))$ are the observations and covariates.
      - $M$ contains the static covariates.

    The general forecasting model must allow arbitrary number of observations and
    target time steps.
    """

    @abstractmethod
    def __call__(self, t: Tensor, S: Tensor, /, M: Optional[Tensor] = None) -> Tensor:
        r"""Forward pass of the model."""
        ...


class ProbabilisticForecastingModel(Protocol):
    r"""Protocol for probabilistic forecasting models.

    A probabilistic forecasting model is a model that predicts the conditional
    distribution of the future time series, given the past observations.

    .. math::   p̂(yₜ | t, S, M) = model(t, S, M)

    We distinguish between 2 types of probabilistic forecasting models:

    - fully joint-distribution models predict the joint distribution
      $p(𝐲_𝐭 | 𝐭, S, M) = p(y_{t₁}, …, y_{tₙ} | (t₁, ..., tₙ), S, M)$
    - per-time-step models predict the marginal distribution
      $p(y_{tₖ} | tₖ, S, M)$ for each time step $tₖ ∈ T$.
    """

    @abstractmethod
    def __call__(
        self, t: Tensor, S: Tensor, /, M: Optional[Tensor] = None
    ) -> Distribution:
        r"""Forward pass of the model."""
        ...
