r"""Update latent states using gradients of observation losses.

## Point predictions

Let $ŷ = f(x)$ be a prediction from latent state $x$ and let $y$ be an
observation. A single gradient step on $ℓ(x) = ‖f(x) - y‖²$ gives

$$ x' = x - η∇ₓℓ(x), \qquad ∇ₓℓ(x) = 2𝐃f(x)ᵀ(f(x) - y) $$

An exact prediction $f(x) = y$ is a fixed point of this update because the
gradient vanishes there. This does not imply that the update is idempotent
for other states.

## Predictive distributions

For latent density $p_θ$ and an invertible decoder $ϕ$ independent of $θ$,
the predictive density is

$$ q(y∣θ) = p_θ(ϕ⁻¹(y))|\det 𝐃ϕ⁻¹(y)| $$

One may combine its negative log-likelihood with a distributional anchor:

$$ J(θ; θ₋, y) = -\log q(y∣θ) + λd(p_θ, p_{θ₋}) $$

Gaussian KL divergences have closed-form expressions, which can be
differentiated to obtain regularization gradients.
"""

__all__ = [
    "LpLoss",
    "GradientStepUpdater",
    "lp_loss",
]

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
    r"""Compute an aggregated $Lᵖ$ reconstruction loss.

    The summed loss is $∑ᵢ |xᵢ - yᵢ|ᵖ$ along `dim`. With `aggregation="mean"`,
    divide by the observed-coordinate count when a mask is supplied, or by the
    last dimension of `x - y` otherwise. The usual featurewise mean uses `dim=-1`.

    True mask entries include coordinates in the loss. If all entries in a
    masked slice are false, the sum is zero and the mean is undefined (`NaN`).

    Raises:
        ValueError: The aggregation is neither `"sum"` nor `"mean"`.
    """
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
    r"""Compute summed or mean $Lᵖ$ reconstruction losses.

    This wraps `lp_loss` with a positive exponent $p$ and defaults to mean
    aggregation. True mask entries identify coordinates included in the loss.
    An entirely masked slice has zero summed loss and an undefined mean loss.
    """

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
    r"""Perform one gradient step on the observation loss.

    $$ x₊ = x₋ - η\left[∇ₓℓ(f(x), y)\right]_{x=x₋} $$

    Here `decoder` defines $f$, and the learnable step size $η$ is initialized
    from `step_size`. The `"l1"` and `"l2"` loss presets use mean absolute
    and mean squared reconstruction losses, respectively.

    A supplied mask uses true entries for observed coordinates and is forwarded
    to the loss. Custom losses must accept `mask` when masked updates are used.
    NaNs are not inferred as missing values when no mask is supplied.
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
        return self.loss(self.decoder(x), y, mask=mask).sum()

    def grad_fn(self, y: Tensor, x: Tensor, /, *, mask: Tensor | None = None) -> Tensor:
        r"""Return the observation-loss gradient with respect to the current state.

        The returned tensor has the same shape as `x`. A supplied mask is passed
        to the loss with true entries marking observed coordinates.
        """
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
