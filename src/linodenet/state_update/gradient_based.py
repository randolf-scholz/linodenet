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
    "GaussianForwardUpdater",
    "lp_loss",
]

from collections.abc import Callable
from functools import partial
from typing import Any

import torch
from torch import Tensor, nn

from linodenet.distributions.gaussian import (
    CovarianceType,
    GaussianParams,
    argmin_forward_kl,
)
from linodenet.mappings import Transform
from linodenet.nn.containers import Constant

from .base import AbstractStateUpdate, SparseVectorStateUpdate

type ScalarLike = Tensor | float


def lp_loss(
    x: Tensor,  # (..., d)
    y: Tensor,  # (..., d)
    /,
    *,
    mask: Tensor | None = None,  # (..., d)
    p: float = 2.0,
    dim: int = -1,
    aggregation: str = "sum",
) -> Tensor:  # (...)
    r"""Compute a per-batch-element $Lᵖ$ reconstruction loss $‖x-y‖ₚᵖ$."""
    r = x - y
    if mask is not None:
        r = torch.where(mask, r, 0.0)
        count = mask.sum(dim=dim)
    else:
        count = r.shape[-1]

    match aggregation:
        case "sum":
            return r.abs().pow(p).sum(dim=dim)
        case "mean":
            return r.abs().pow(p).sum(dim=dim).div(count)
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

    def forward(self, x: Tensor, y: Tensor, /, *, mask: Tensor | None = None) -> Tensor:
        return lp_loss(
            x,
            y,
            mask=mask,
            p=self.p,
            dim=self.dim,
            aggregation=self.aggregation,
        )


class GradientStepUpdater(nn.Module, SparseVectorStateUpdate):
    r"""Perform a gradient based update.

    This updater performs a single explicit gradient step on the observation loss:

    .. math:: z₊ = z₋ - η∇₟ℓ(f(z₋), y)

    In this one-step setting, a smooth anchor penalty like $λ⋅d(z, z₋)$ does not
    influence the step when evaluated at $z = z₋$, because its gradient vanishes at
    the anchor. The ``regularizer`` and ``regularization_strength`` arguments are
    therefore kept only for API compatibility and configuration round-tripping; they
    do not affect the update.
    """

    def __init__(
        self,
        input_size: int = -1,
        hidden_size: int = -1,
        *,
        decoder: nn.Module,
        loss: nn.Module | str = "l2",
        step_size: Tensor | float = 1e-2,
    ) -> None:
        super().__init__()
        SparseVectorStateUpdate.__init__(
            self,
            input_size=input_size,
            hidden_size=hidden_size,
        )

        self.decoder = decoder
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

    @partial(torch.func.grad, argnums=2)
    def _grad_fn_no_mask(
        self,
        y: Tensor,  # (..., e)
        x: Tensor,  # (..., d)
        /,
    ) -> Tensor:  # (...,)
        # Note: ∇_θ ∑ᵢ ℓ(θᵢ) = (∇_{θ₁} ℓ(θ₁), ..., ∇_{θₙ} ℓ(θₙ))
        return self.loss(self.decoder(x), y).sum()  # ℓ(f(x), y)

    @partial(torch.func.grad, argnums=2)
    def _grad_fn_with_mask(
        self,
        y: Tensor,  # (..., e)
        x: Tensor,  # (..., d)
        mask: Tensor,  # (..., e)
        /,
    ) -> Tensor:  # (...,)
        # Note: ∇_θ ∑ᵢ ℓ(θᵢ) = (∇_{θ₁} ℓ(θ₁), ..., ∇_{θₙ} ℓ(θₙ))
        return self.loss(self.decoder(x), y, mask=mask).sum()  # pyright: ignore[reportCallIssue]

    def grad_fn(self, y: Tensor, x: Tensor, /, *, mask: Tensor | None = None) -> Tensor:
        r"""Return the gradient while preserving the input batch shape."""
        return (
            self._grad_fn_no_mask(y, x)
            if mask is None
            else self._grad_fn_with_mask(y, x, mask)
        )

    def forward(
        self,
        y: Tensor,  # (..., e)
        x: Tensor,  # (..., d)
        /,
        *,
        mask: Tensor | None = None,  # (..., e)
    ) -> Tensor:  # (..., d)
        r"""Compute $x - η∇ₓℒ(x)$ with canonical state-update argument order."""
        return x - self.step_size * self.grad_fn(y, x, mask=mask)


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

        match retention:
            case [rho_mu, rho_sigma]:
                strength_mu = torch.as_tensor(rho_mu)
                strength_sigma = torch.as_tensor(rho_sigma)
                if not torch.all((strength_mu >= 0.0) & (strength_mu <= 1.0)):
                    raise ValueError(f"ρ_mu must be in [0, 1], got {rho_mu!r}")
                if not torch.all((strength_sigma > 0.0) & (strength_sigma <= 1.0)):
                    raise ValueError(f"ρ_sigma must be in (0, 1], got {rho_sigma!r}")

                param_mu = nn.Parameter(torch.logit(strength_mu), requires_grad=True)
                param_sigma = nn.Parameter(
                    torch.logit(strength_sigma), requires_grad=True
                )

                self.retention_mu = nn.Sequential(
                    Constant(param_mu),
                    nn.Sigmoid(),
                )
                self.retention_sigma = nn.Sequential(
                    Constant(param_sigma),
                    nn.Sigmoid(),
                )

            case rho:
                strength = torch.as_tensor(rho)
                if not torch.all((strength >= 0.0) & (strength <= 1.0)):
                    raise ValueError(f"ρ must be in [0, 1], got {rho!r}")
                param = nn.Parameter(
                    torch.logit(strength), requires_grad=retention_learnable
                )
                self.retention_mu = nn.Sequential(
                    Constant(param),
                    nn.Sigmoid(),
                )
                self.retention_sigma = nn.Sequential(
                    Constant(param),
                    nn.Sigmoid(),
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
        r"""Return the exact reverse-KL Gaussian update $θ₊$."""
        z = self.decoder(y_obs)

        if context is None:
            self.rho_mu = self.retention_mu()
            self.rho_sigma = self.retention_sigma()
        else:
            self.rho_mu = self.retention_mu(context)
            self.rho_sigma = self.retention_sigma(context)

        return argmin_forward_kl(
            z,
            theta,
            retention=(self.rho_mu, self.rho_sigma),
            parametrization=self.parametrization,
        )
