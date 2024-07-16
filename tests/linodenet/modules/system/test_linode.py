r"""Test error of linear ODE against odeint."""

import logging
import random

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
from numpy.typing import NDArray
from scipy.integrate import odeint
from tqdm.autonotebook import trange
from typing_extensions import Any, Literal, Optional

from linodenet.config import PROJECT
from linodenet.modules import LinODE
from tests.test_utils import scaled_norm, visualize_distribution

RESULT_DIR = PROJECT.RESULTS_DIR[__file__]
__logger__ = logging.getLogger(__name__)
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("PIL").setLevel(logging.WARNING)


def compute_linode_error(
    *,
    num: Optional[int] = None,
    dim: Optional[int] = None,
    precision: Literal["single", "double"] = "single",
    relative_error: bool = True,
    device: Optional[torch.device] = None,
) -> NDArray:
    r"""Compare `LinODE` against `scipy.odeint` on linear system.

    .. Signature:: `` -> (q, N)``
    """
    N = num or random.choice([10 * k for k in range(1, 11)])
    D = dim or random.choice([2**k for k in range(1, 8)])
    logger = __logger__.getChild(f"{LinODE.__name__}-test-{N}-{D}")

    numpy_dtype: type[np.number]
    torch_dtype: torch.dtype
    rng = np.random.default_rng()

    if precision == "single":
        eps = 2**-24
        numpy_dtype = np.float32
        torch_dtype = torch.float32
    elif precision == "double":
        eps = 2**-53
        numpy_dtype = np.float64
        torch_dtype = torch.float64
    else:
        raise ValueError

    t0, t1 = rng.uniform(low=-10, high=10, size=(2,))
    t0, t1 = min(t0, t1), max(t0, t1)  # make sure t0 ≤ t1
    A = (rng.normal(size=(D, D)) / np.sqrt(D)).astype(numpy_dtype)
    x0 = rng.normal(size=D).astype(numpy_dtype)
    T = rng.uniform(low=t0, high=t1, size=N - 2)
    T = np.sort([t0, *T, t1]).astype(numpy_dtype)

    def func(_, x):
        return A @ x

    X = torch.tensor(odeint(func, x0, T, tfirst=True), dtype=torch_dtype)

    # A_torch = torch.tensor(A, dtype=torch_dtype, device=device)
    T_torch = torch.tensor(T, dtype=torch_dtype, device=device)
    x0_torch = torch.tensor(x0, dtype=torch_dtype, device=device)

    model = LinODE(
        input_size=D,
        cell={
            "kernel_initialization": A,
            "scalar": 1.0,
            "scalar_learnable": False,
        },
    )
    model.to(dtype=torch_dtype, device=device)
    assert model.cell.scalar == 1.0

    Xhat = model(T_torch, x0_torch)
    Xhat = Xhat.clone().detach().cpu()

    err = (X - Xhat).abs()

    if relative_error:
        err /= X.abs() + eps

    # NOTE: shape:
    logger.debug("shapes: X:%s Xhat:%s err:5%s", X.shape, Xhat.shape, err.shape)
    return np.array([scaled_norm(err, p=p, keepdim=False) for p in (1, 2, np.inf)])


def make_error_plots(
    *,
    error_single: NDArray,
    error_double: NDArray,
    logger: logging.Logger = __logger__,
    **extra_stats: Any,
) -> None:
    r"""Create histogram plot of the errors."""
    assert error_single.shape == error_double.shape
    print(error_single.shape)
    num_samples = error_single.shape[1]

    with plt.style.context("bmh"):
        fig, ax = plt.subplots(
            ncols=3,
            nrows=2,
            figsize=(10, 5),
            tight_layout=True,
            sharey="row",
            sharex="all",
        )

    logger.info("generating figure")
    for i, err in enumerate((error_single, error_double)):
        for j, p in enumerate((1, 2, np.inf)):
            visualize_distribution(
                err[j], log=True, ax=ax[i, j], extra_stats=extra_stats
            )
            if j == 0:
                ax[i, 0].annotate(
                    f"FP{32 * (i + 1)}",
                    xy=(0, 0.5),
                    xytext=(-ax[i, 0].yaxis.labelpad - 5, 0),
                    xycoords=ax[i, 0].yaxis.label,
                    textcoords="offset points",
                    size="xx-large",
                    ha="right",
                    va="center",
                )
            if i == 1:
                ax[i, j].set_xlabel(f"scaled, relative L{p} distance")

    fig.suptitle(
        r"Difference $x^{\text{(LinODE)}}$ and $x^{\text{(odeint)}}$"
        f" -- {num_samples} random systems"
    )

    fig.savefig(RESULT_DIR / "LinODE_odeint_comparison.pdf")


@pytest.mark.flaky(reruns=3)
def test_linode_error(make_plots: bool, *, num_samples: int = 100) -> None:
    r"""Compare LinODE against scipy.odeint on random linear system."""
    LOGGER = __logger__.getChild(LinODE.__name__)
    LOGGER.info("Testing %s.", LinODE)
    extra_stats = {"Samples": num_samples}

    LOGGER.info("Generating %i samples in single precision", num_samples)
    err_single = np.array(
        [compute_linode_error(precision="single") for _ in trange(num_samples)],
        dtype=np.float32,
    ).T

    LOGGER.info("Generating %i samples in double precision", num_samples)
    err_double = np.array(
        [compute_linode_error(precision="double") for _ in trange(num_samples)],
        dtype=np.float64,
    ).T

    if make_plots:
        make_error_plots(
            error_single=err_single,
            error_double=err_double,
            logger=LOGGER,
            **extra_stats,
        )

    levels = (10.0**k for k in (0, 2, 4))
    for err, tol in zip(err_single, levels, strict=True):
        q = np.nanquantile(err, 0.99)
        LOGGER.info("99%% quantile %f", q)
        assert q <= tol, f"99% quantile {q=} larger than allowed {tol=}"
    # Note that the matching of the predictions is is 4 order of magnitude better in FP64.
    # Since 10^4 ~ 2^13
    levels = (10.0**k for k in (-4, -2, -0))
    for err, tol in zip(err_double, levels, strict=True):
        q = np.nanquantile(err, 0.99)
        LOGGER.info("99%% quantile %f", q)
        assert q <= tol, f"99% quantile {q=} larger than allowed  {tol=}"
    LOGGER.info("%s passes test ✔ ", LinODE)
