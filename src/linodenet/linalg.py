r"""Linear algebra utility functions."""

__all__ = [
    "pad",
    "scaled_norm",
    "geometric_mean",
]

from typing import Optional

import torch
from torch import Tensor, jit


@jit.script
def pad(
    x: Tensor,
    value: float,
    pad_width: int,
    dim: int = -1,
    prepend: bool = False,
) -> Tensor:
    r"""Pad a tensor with a constant value along a given dimension."""
    shape = list(x.shape)
    shape[dim] = pad_width
    z = torch.full(shape, value, dtype=x.dtype, device=x.device)

    if prepend:
        return torch.cat((z, x), dim=dim)
    return torch.cat((x, z), dim=dim)


@jit.script
def geometric_mean(
    x: Tensor,
    axis: Optional[int | list[int]] = None,
    keepdim: bool = False,
) -> Tensor:
    r"""Geometric mean of a tensor.

    .. signature:: ``(..., n) -> (...)``
    """
    if axis is None:
        dim = list(range(x.ndim))
    elif isinstance(axis, int):
        dim = [axis]
    else:
        dim = axis

    return x.log().nanmean(dim=dim, keepdim=keepdim).exp()


@jit.script
def scaled_norm(
    x: Tensor,
    p: float = 2.0,
    axis: Optional[int | list[int]] = None,
    keepdim: bool = False,
) -> Tensor:
    r"""Shortcut for scaled norm.

    .. signature:: ``(..., n) -> ...``
    """
    # TODO: deal with nan values
    x = x.abs()

    if axis is None:
        dim = list(range(x.ndim))
    elif isinstance(axis, int):
        dim = [axis]
    else:
        dim = axis

    if p == float("inf"):
        return x.amax(dim=dim, keepdim=keepdim)
    if p == -float("inf"):
        return x.amin(dim=dim, keepdim=keepdim)
    if p == 0:
        return geometric_mean(x, axis=dim, keepdim=keepdim)

    # NOTE: preconditioning with x_max is not necessary, but it helps with numerical stability and prevents overflow
    x_max = x.abs().amax(dim=dim, keepdim=True)
    result = x_max * (x / x_max).pow(p).mean(dim=dim, keepdim=True).pow(1 / p)
    return result.squeeze(dim=dim * (1 - int(keepdim)))  # branchless
    # return x.pow(p).mean(dim=dim, keepdim=keepdim).pow(1 / p)
