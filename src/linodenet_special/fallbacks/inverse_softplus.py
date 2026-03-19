r"""Inverse softplus implementation for PyTorch tensors."""

__all__ = ["inverse_softplus"]

import torch
from torch import Tensor

from signatures import signature


@signature("[(...)] -> (...)")
def inverse_softplus(x: Tensor, /) -> Tensor:
    r"""Compute the inverse of `torch.nn.functional.softplus`.

    This uses the stable identity

    .. math::
       \operatorname{softplus}^{-1}(x) = x + \log(1 - e^{-x})

    which avoids the overflow in `log(expm1(x))` for large `x`.

    Args:
        x: Input tensor. The function is defined for $x ≥ 0$.

    Returns:
        Tensor with the same shape as `x`.
    """
    return x + (-torch.expm1(-x)).log()
