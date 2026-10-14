r"""Inverse softplus implementation for PyTorch tensors."""

__all__ = ["inverse_softplus"]

import torch
from torch import Tensor

from signatures import signature


@signature("[(...)] -> (...)")
def inverse_softplus(x: Tensor, /) -> Tensor:
    r"""Compute the inverse of `torch.nn.functional.softplus`.

    This uses the stable identity

    $$ \softplus⁻¹(x) = x + \log(1 - e⁻ˣ) $$

    which avoids the overflow in $\log(\op{expm1}(x))$ for large $x$.

    Args:
        x: Input tensor. Positive values are in the inverse's domain.

    Returns:
        Tensor with the same shape as `x`.
        Zero yields $-∞$ as a limiting value, and negative inputs yield NaN.
    """
    return x + (-torch.expm1(-x)).log()
