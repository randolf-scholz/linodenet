r"""Gradient based state update.

Idea: We can directly update the latent state using the gradient of some loss function.

Example: Gradient update with point forecasting.
    Let $ŷ = f(z)$ be the prediciton from the latent state $z$ and $y$ be an observation.
    Consider the loss function $ℓ(z) = ‖ŷ(z) - y‖²$. We can compute an updated posterior
    state by performing 1 gradient step along this loss: $z' = z - η∇₟ℓ(z)$

    For the MSE loss, this update has the idempotency property: if $ŷ=y$, then
    $∇₟‖ŷ(z) - y‖² = 2(∂ŷ/∂z)(̂y -y) = 0$, and so $z'=z$.

    Additionally, one may add a regularization term like $r = λ‖z - z₋‖$, that is
    proportional to the distance between the current and previous state.


Example: Gradient update with probabilistic forecasting.
    Let z ∼ F(θ) be the latent state and assume we use some normalizing flow $ϕ$ to get the
    predictive distribution, i.e. $p̂(y∣t) = p₟(ϕ⁻¹(y))|𝐃ϕ⁻¹(y)|$.
    Then we can define the loss function as the negative log-likelihood of the observation $y$,
    and possibly a regularization term depending on the similarity of the prior and posterior distribution:

    Jₜ(θ; θ₋, y) = -\log p̂(y∣θ) + λ⋅𝐝(F(θ), F(θ₋))

    An important case is when $F(θ) = 𝓝(μ, Σ)$ is a normal distribution, and $𝐝$ is the KL divergence.
    Because in this case, there is a closed-form solution for the KL-term, which allows us to
    compute the exact gradient update.
"""

from math import expm1, log
from typing import Any, Callable

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils import _pytree

from linodenet.distributions import Distribution
from linodenet.distributions.gaussian import kl
from linodenet.mappings import Transform


class GradientStepUpdater(nn.Module):
    r"""Single gradient-step updater for latent distribution parameters.

    .. math:: ℒ(z) = ∇₟ℓ(f(z), y) + λ d(z, z₋)
                z' = z₋ - η∇₟ℒ(z₋)
    """

    def __init__(
        self,
        decoder: Transform[Tensor, Tensor],
        loss: nn.Module | str = "l2",
        regularizer: nn.Module | str = "l2",
        regularization_strength: float = 1e-3,
    ) -> None:
        super().__init__()

        match loss:
            case nn.Module():
                self.loss = loss
            case "l1":
                self.loss = nn.L1Loss()
            case "l2":
                self.loss = nn.MSELoss()
            case _:
                raise ValueError(f"Unknown loss: {loss!r}")

        match regularizer:
            case nn.Module():
                self.regularizer = regularizer
            case "l1":
                self.regularizer = nn.L1Loss()
            case "l2":
                self.regularizer = nn.MSELoss()
            case _:
                raise ValueError(f"Unknown regularizer: {regularizer!r}")

        self.regularization_strength = regularization_strength

    def value_and_grad(self, z_prev: Tensor, y: Tensor) -> tuple[Tensor, Tensor]:
        r"""Computes f(z) and ∇₟ℒ(z) at z_prev, where ℒ(z) = ℓ(f(z), y) + λ d(z, z_prev)."""


class GaussianGradientStepUpdater(nn.Module):
    r"""Perform a gradient based update assuming a latent Gaussian distribution.

    .. math:: Jₜ(θ; θ₋, y_obs) = -\log q(y_obs∣θ) + λ\kl(𝓝(μ, Σ), 𝓝(μ₋, Σ₋))

    where $θ = (μ, Σ)$ are the parameters of the latent Gaussian distribution,
    $θ₋$ are the parameters before the update, and $y_obs$ is the observed value.
    The first term is the negative log-likelihood of the observation under the current parameters,
    and the second term is a KL divergence regularization that encourages the updated parameters
    to stay close to the previous parameters.
    """

    def __init__(
        self,
        decoder: Transform[Tensor, Tensor],
        loss: nn.Module,
        regularizer: nn.Module,
        regularization_strength: float,
    ) -> None:
        super().__init__()
        raise NotImplementedError

    def update_covariance(
        self, theta: tuple[Tensor, Tensor], y_obs: Tensor
    ) -> tuple[Tensor, Tensor]:
        r"""Gradient step assuming parameterization $θ=(μ, Σ)$."""
        raise NotImplementedError

    def update_precision(
        self, theta: tuple[Tensor, Tensor], y_obs: Tensor
    ) -> tuple[Tensor, Tensor]:
        r"""Gradient step assuming parameterization $θ=(μ, Λ)$, with $Σ=Λ⁻¹$."""
        raise NotImplementedError

    def update_cholesky(
        self, theta: tuple[Tensor, Tensor], y_obs: Tensor
    ) -> tuple[Tensor, Tensor]:
        r"""Gradient step assuming parameterization $θ=(μ, L)$, with $Σ=LLᵀ$."""
        raise NotImplementedError
