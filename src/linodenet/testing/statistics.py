r"""Statistical tests."""

__all__ = [
    "is_standardized",
]

from typing import Optional

import torch
from torch import Tensor


def is_standardized(
    values: Tensor,
    /,
    *,
    dim: None | int | tuple[int, ...] | list[int] = -1,
    tol: Optional[float] = None,
) -> Tensor:
    r"""Check if a tensor has zero mean and unit variance.

    .. signature:: ``[..., *D] -> bool[...]``
    """
    # NOTE: Often, normality will be achieved approximately, in terms of the CTL.
    #   As the sample mean of n-many samples from a normal distribution is distributed as N(μ, σ²/n),
    #   Knowing that the input should be N(0, 1), we can expect the sample mean to be distributed as N(0, 1/n)
    #   That is with standard deviation 1/√n.
    #   Therefore, to get k-sigma confidence, we should check whether the mean is outside the interval
    #   [-k/√n, k/√n]
    dims: list[int] = (
        [dim]
        if isinstance(dim, int)
        else list(range(values.dim()))
        if dim is None
        else list(dim)
    )

    # compute mean an stdv
    if tol is None:
        output_lengths = torch.tensor([values.shape[k] for k in dims])
        count = output_lengths.prod()
        # default: 3-sigma rule
        tol = 3.0 / count.sqrt().item()

    mean_values = values.mean(dim=dims)
    stdv_values = values.std(dim=dims)

    # check that the mean is close to 0 and stdv is close to
    mean_valid = mean_values.abs() <= tol
    stdv_valid = (stdv_values - 1.0).abs() <= tol
    return mean_valid & stdv_valid
