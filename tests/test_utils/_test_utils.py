"""Test utils."""

__all__ = [
    # Functions
    "camel2snake",
    "geometric_mean",
    "scaled_norm",
    "snake2camel",
    "visualize_distribution",
]


import numpy as np
import torch
from matplotlib.offsetbox import AnchoredText
from matplotlib.pyplot import Axes
from numpy.typing import ArrayLike, NDArray
from scipy.stats import mode
from torch import Tensor, jit
from typing_extensions import Literal, Optional, TypeAlias, Union

Location: TypeAlias = Literal[
    "upper right",
    "upper left",
    "lower left",
    "lower right",
    "center left",
    "center right",
    "lower center",
    "upper center",
    "center",
]


@torch.no_grad()
def visualize_distribution(
    data: ArrayLike,
    /,
    *,
    ax: Axes,
    num_bins: int = 50,
    log: bool = True,
    loc: Location = "upper right",
    print_stats: bool = True,
    extra_stats: Optional[dict] = None,
) -> None:
    r"""Plot the distribution of x in the given figure axes.

    Args:
        data: Data to plot.
        ax: Axes to plot into.
        num_bins: Number of bins to use for histogram.
        log: If True, use log base 10, if `float`, use  log w.r.t. this base
        loc: Location of 'stats' text.
        print_stats: Add table of mean, std, min, max, median, mode to plot
        extra_stats: Additional things to add to the 'stats' table
    """
    if isinstance(data, Tensor):
        data = data.detach().cpu().numpy()

    x: NDArray[np.float64] = np.asarray(data, dtype=float).flatten()
    nans = np.isnan(x)
    x = x[~nans]

    ax.grid(axis="x")
    ax.set_axisbelow(True)

    if log:
        base = 10 if log is True else log
        tol = 2**-24 if np.issubdtype(x.dtype, np.float32) else 2**-53
        z = np.log10(np.maximum(x, tol))
        ax.set_xscale("log", base=base)
        ax.set_yscale("log", base=base)
        low = np.floor(np.quantile(z, 0.01))
        high = np.ceil(np.quantile(z, 1 - 0.01))
        x = x[(z >= low) & (z <= high)]
        bins = np.logspace(low, high, num=num_bins, base=10)
    else:
        low = np.quantile(x, 0.01)
        high = np.quantile(x, 1 - 0.01)
        bins = np.linspace(low, high, num=num_bins)

    ax.hist(x, bins=bins, density=True)  # type: ignore[arg-type]

    if print_stats:
        stats = {
            "NaNs": f"{100 * np.mean(nans):6.2%}",
            "Mode": f"{mode(x)[0]: .2g}",
            "Min": f"{np.min(x): .2g}",
            "Median": f"{np.median(x): .2g}",
            "Max": f"{np.max(x): .2g}",
            "Mean": f"{np.mean(x): .2g}",
            "Stdev": f"{np.std(x): .2g}",
        }
        if extra_stats is not None:
            stats |= {str(key): str(val) for key, val in extra_stats.items()}

        pad = max(map(len, stats), default=0)
        table = "\n".join([f"{key:<{pad}}  {val}" for key, val in stats.items()])

        # use mono-spaced font
        textbox = AnchoredText(
            table,
            loc=loc,
            borderpad=0.0,
            prop={"family": "monospace"},
        )
        textbox.patch.set_alpha(0.8)
        ax.add_artist(textbox)


@jit.script
def geometric_mean(
    x: Tensor,
    axis: Union[None, int, list[int]] = None,
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
    axis: Union[None, int, list[int]] = None,
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


def camel2snake(string: str) -> str:
    r"""Convert camel case to snake case."""
    return "".join(["_" + c.lower() if c.isupper() else c for c in string]).lstrip("_")


def snake2camel(string: str) -> str:
    r"""Convert snake case to camel case."""
    return "".join([c.title() for c in string.split("_")])
