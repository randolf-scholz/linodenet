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
    "LpLoss",
    "GradientStepUpdater",
    "GaussianGradientStepUpdater",
    "lp_loss",
]

from collections.abc import Callable
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


def lp_loss(
    x: Tensor,  # (..., d)
    y: Tensor,  # (..., d)
    /,
    *,
    p: float = 2.0,
    dim: int = -1,
    aggregation: str = "mean",
) -> Tensor:  # (...)
    r"""Compute a per-batch-element $Lᵖ$ reconstruction loss $‖x-y‖ₚᵖ$."""
    match aggregation:
        case "sum":
            return (x - y).abs().pow(p).sum(dim=dim)
        case "mean":
            return (x - y).abs().pow(p).mean(dim=dim)
        case _:
            raise ValueError(f"Unexpected aggregation: {aggregation!r}")


class LpLoss(nn.Module):
    r"""Compute a per-batch-element $Lᵖ$ reconstruction loss $‖x-y‖ₚᵖ$."""

    def __init__(
        self,
        p: float = 2.0,
        dim: int = -1,
        aggregation: str = "mean",
    ) -> None:
        super().__init__()
        if p <= 0:
            raise ValueError(f"Expected p > 0, got {p!r}.")
        if aggregation not in {"sum", "mean"}:
            raise ValueError(
                f"Expected aggregation to be 'sum' or 'mean', got {aggregation!r}."
            )

        self.p = p
        self.dim = dim
        self.aggregation = aggregation

    __call__: Callable[[Tensor, Tensor], Tensor]

    def forward(self, x: Tensor, y: Tensor, /) -> Tensor:
        return lp_loss(x, y, p=self.p, dim=self.dim, aggregation=self.aggregation)


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
                self.loss = LpLoss(p=1.0)
            case "l2":
                self.loss = LpLoss(p=2.0)
            case _:
                raise ValueError(f"Unknown loss: {loss!r}")

        match regularizer:
            case nn.Module():
                self.regularizer = regularizer
            case "l1":
                self.regularizer = LpLoss(p=1.0)
            case "l2":
                self.regularizer = LpLoss(p=2.0)
            case _:
                raise ValueError(f"Unknown regularizer: {regularizer!r}")

    @partial(torch.func.vmap, in_dims=(None, 0, 0, 0))
    @partial(torch.func.grad, argnums=1)
    def _grad_fn_no_mask(
        self,
        z: Tensor,  # (B, d)
        z_prev: Tensor,  # (B, d)
        y: Tensor,  # (B, e)
        /,
    ) -> Tensor:  # (B)
        return (
            self.loss(self.decoder(z), y)  # ℓ(f(z), y)
            + self.regularization_strength * self.regularizer(z, z_prev)  # λ‖z-z₋‖²
        )

    @partial(torch.func.vmap, in_dims=(None, 0, 0, 0, 0))
    @partial(torch.func.grad, argnums=1)
    def _grad_fn_with_mask(
        self,
        z: Tensor,  # (B, d)
        z_prev: Tensor,  # (B, d)
        y: Tensor,  # (B, e)
        mask: Tensor,  # (B, e)
        /,
    ) -> Tensor:  # (B)
        return (
            # ℓ(f(z), y)
            self.loss(self.decoder(z), y, mask=mask)  # pyright: ignore[reportCallIssue]
            + self.regularization_strength * self.regularizer(z, z_prev)  # λ⋅‖z-z₋‖²
        )

    def grad_fn(
        self, z: Tensor, z_prev: Tensor, y: Tensor, /, *, mask: Tensor | None = None
    ) -> Tensor:
        r"""Return the gradient while preserving the input batch shape."""
        z_flat = z.reshape(-1, z.shape[-1])
        z_prev_flat = z_prev.reshape(-1, z_prev.shape[-1])
        y_flat = y.reshape(-1, y.shape[-1])

        grad = (
            self._grad_fn_no_mask(
                z_flat,
                z_prev_flat,
                y_flat,
            )
            if mask is None
            else self._grad_fn_with_mask(
                z_flat,
                z_prev_flat,
                y_flat,
                mask.reshape(-1, mask.shape[-1]),
            )
        )

        return grad.reshape_as(z)

    __call__: Callable[[Tensor, Tensor], Tensor]

    def forward(
        self,
        z: Tensor,  # (..., d)
        y: Tensor,  # (..., e)
        /,
        mask: Tensor | None = None,  # (..., e)
    ) -> Tensor:  # (..., d)
        r"""Computes z_prev - η∇₟ℒ(z_prev), where ℒ(z) = ℓ(f(z), y) + λ⋅d(z, z_prev)."""
        return z - self.step_size * self.grad_fn(z, z, y, mask=mask)


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

    @partial(torch.func.vmap, in_dims=(None, (0, 0), (0, 0), 0))
    @partial(torch.func.grad, argnums=1)
    def _grad_fn_flat_batch(
        self,
        theta: GaussianParams,
        theta_prev: GaussianParams,
        y_obs: Tensor,
        /,
    ) -> Tensor:
        return (
            -self.log_prob(y_obs, theta)  # -log q(y∣θ)
            + (  # λ⋅d(θ, θ₋)
                self.regularization_strength
                * kl(theta, theta_prev, parametrization=self.parametrization)
            )
        )

    def grad_fn(
        self, theta: GaussianParams, theta_prev: GaussianParams, y_obs: Tensor, /
    ) -> GaussianParams:
        r"""Return the gradient while preserving the input batch shape."""
        mean, cov = theta
        mean_prev, cov_prev = theta_prev
        mean_flat = mean.reshape(-1, mean.shape[-1])
        cov_flat = cov.reshape(-1, cov.shape[-2], cov.shape[-1])

        grad_mean, grad_cov = self._grad_fn_flat_batch(
            (mean_flat, cov_flat),
            (
                mean_prev.reshape_as(mean_flat),
                cov_prev.reshape_as(cov_flat),
            ),
            y_obs.reshape(-1, y_obs.shape[-1]),
        )
        return grad_mean.reshape_as(mean), grad_cov.reshape_as(cov)

    __call__: Callable[[GaussianParams, Tensor], GaussianParams]

    def forward(self, theta: GaussianParams, y_obs: Tensor, /) -> GaussianParams:
        r"""Return the updated Gaussian parameters $(μ', Σ')$."""
        mean, cov = theta
        grad_mean, grad_cov = self.grad_fn(theta, theta, y_obs)
        mean_post = mean - self.step_size_mean * grad_mean
        cov_post = cov - self.step_size_cov * grad_cov
        return mean_post, cov_post
