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

__all__ = [
    "GradientStepUpdater",
    "GaussianGradientStepUpdater",
]

from functools import partial

import torch
from torch import Tensor, nn

from linodenet.distributions.gaussian import (
    CovarianceType,
    GaussianParams,
    kl,
    log_prob,
)
from linodenet.mappings import Transform


class GradientStepUpdater(nn.Module):
    r"""Single gradient-step updater for latent distribution parameters.

    .. math:: ℒ(z) = ∇₟ℓ(f(z), y) + λ d(z, z₋)
                z' = z₋ - η∇₟ℒ(z₋)
    """

    def __init__(
        self,
        *,
        decoder: nn.Module,
        loss: nn.Module | str = "l2",
        regularizer: nn.Module | str = "l2",
        regularization_strength: float = 1e-3,
        step_size: float = 1e-2,
    ) -> None:
        super().__init__()

        self.decoder = decoder
        self.regularization_strength = nn.Parameter(
            torch.as_tensor(regularization_strength)
        )
        self.step_size = nn.Parameter(torch.as_tensor(step_size))

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

    @partial(torch.func.grad, argnums=1)
    def grad_fn(self, z: Tensor, z_prev: Tensor, y: Tensor) -> Tensor:
        return (
            self.loss(self.decoder(z), y)  # ℓ(f(z), y)
            + self.regularization_strength * self.regularizer(z, z_prev)  # λ‖z-z₋‖²
        )

    def forward(self, z_prev: Tensor, y: Tensor) -> Tensor:
        r"""Computes z_prev - η∇₟ℒ(z_prev), where ℒ(z) = ℓ(f(z), y) + λ d(z, z_prev)."""
        return z_prev - self.step_size * self.grad_fn(z_prev, z_prev, y)


class GaussianGradientStepUpdater(nn.Module):
    r"""Perform a gradient based update assuming a latent Gaussian distribution.

    .. math:: Jₜ(θ; θ₋, y_obs) = -\log q(y_obs∣θ) + λ⋅\kl(𝓝(μ, Σ), 𝓝(μ₋, Σ₋))

    where $q(y_obs∣θ) = p(ϕ⁻¹(y_obs)∣θ) |𝐃ϕ⁻¹(y_obs)|$ is the predictive distribution
    obtained by pushing the latent Gaussian through a decoder $ϕ$, and $θ = (μ, Σ)$
    are the parameters of the latent Gaussian distribution. $θ₋$ denote the parameters
    before the update, and $y_obs$ is the observed value.
    The first term is the negative log-likelihood of the observation under the current parameters,
    and the second term is a KL divergence regularization that encourages the updated parameters
    to stay close to the previous parameters.
    """

    def __init__(
        self,
        *,
        decoder: nn.Module,  # & Transform[Tensor, Tensor]
        parametrization: str,
        regularization_strength: float = 1e-3,
        step_size: float = 1e-2,
        step_size_mean: float | None = None,
        step_size_cov: float | None = None,
    ) -> None:
        super().__init__()
        if parametrization != CovarianceType.LOG_CHOLESKY:
            raise NotImplementedError(
                "Only 'log-cholesky' parametrization is currently supported."
            )

        self.decoder: Transform[Tensor, Tensor] = decoder  # type: ignore[assignment]
        self.parametrization = parametrization

        self.regularization_strength = nn.Parameter(
            torch.as_tensor(regularization_strength)
        )
        self.step_size_mean = nn.Parameter(
            torch.as_tensor(step_size if step_size_mean is None else step_size_mean)
        )
        self.step_size_cov = nn.Parameter(
            torch.as_tensor(step_size if step_size_cov is None else step_size_cov)
        )

    def log_prob(self, vals: Tensor, theta: GaussianParams, /) -> Tensor:
        r"""Compute the log probability of the input values under the current parameters."""
        z, logabsdet = self.decoder.encode_and_logabsdet(vals)
        return log_prob(z, theta, parametrization=self.parametrization) + logabsdet

    def forward(self, theta: GaussianParams, y_obs: Tensor) -> GaussianParams:
        r"""Return the updated Gaussian parameters $(μ', Σ')$."""
        grad_fn = torch.func.grad(
            lambda mean, cov: (
                -self.log_prob(y_obs, (mean, cov)).mean()
                + (
                    self.regularization_strength
                    * kl(
                        (mean, cov), theta, parametrization=self.parametrization
                    ).mean()
                )
            ),
            argnums=(0, 1),
        )
        grad_mean, grad_cov = grad_fn(*theta)

        mean_post = theta[0] - self.step_size_mean * grad_mean
        cov_post = theta[1] - self.step_size_cov * grad_cov
        return mean_post, cov_post
