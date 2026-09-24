r"""Statistical tests."""

__all__ = [
    "is_standardized",
]

from collections.abc import Sequence
from typing import Optional, SupportsFloat

import torch
from torch import Tensor

from signatures import signature


def _get_dims(dim: None | int | Sequence[int], values: Tensor) -> list[int]:
    return (
        [dim]
        if isinstance(dim, int)
        else list(range(values.ndim))
        if dim is None
        else list(dim)
    )


def _get_tol(tol: float | None, values: Tensor, *, dims: list[int]) -> float:
    if isinstance(tol, SupportsFloat):
        return float(tol)

    # default: 3-sigma rule
    output_lengths = torch.tensor([values.shape[k] for k in dims])
    count = output_lengths.prod()
    tol = 3.0 / count.sqrt().item()
    return tol


@signature("(..., *ds) -> (...)")
def is_standardized(
    values: Tensor,
    /,
    *,
    dim: None | int | tuple[int, ...] | list[int] = -1,
    tol: Optional[float] = None,
) -> Tensor:
    r"""Check if a tensor has zero mean and unit variance."""
    # NOTE: Often, normality will be achieved approximately, in terms of the CTL.
    #   As the sample mean of n-many samples from a normal distribution is distributed as N(μ, σ²/n),
    #   Knowing that the input should be N(0, 1), we can expect the sample mean to be distributed as N(0, 1/n)
    #   That is with standard deviation 1/√n.
    #   Therefore, to get k-sigma confidence, we should check whether the mean is outside the interval
    #   [-k/√n, k/√n]
    dims = _get_dims(dim, values)

    # compute mean an stdv
    tol = _get_tol(tol, values, dims=dims)

    mean_values = values.mean(dim=dims)
    stdv_values = values.std(dim=dims)

    # check that the mean is close to 0 and stdv is close to
    mean_valid = mean_values.abs() <= tol
    stdv_valid = (stdv_values - 1.0).abs() <= tol
    return mean_valid & stdv_valid
