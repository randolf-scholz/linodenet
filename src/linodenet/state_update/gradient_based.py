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
    "lp_loss",
]

from collections.abc import Callable
from functools import partial

import torch
from torch import Tensor, nn

from .base import SparseVectorStateUpdate


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
