#!/usr/bin/env python
r"""Test error of linear ODE against odeint."""

import logging
import random
from typing import Literal, Optional

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
from numpy.typing import NDArray
from scipy.integrate import odeint
from tqdm.autonotebook import trange

from linodenet.config import PROJECT
from linodenet.models import LinODE
from tsdm.linalg import scaled_norm
from tsdm.viz import visualize_distribution

logging.basicConfig(level=logging.INFO)
__logger__ = logging.getLogger(__name__)
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("PIL").setLevel(logging.WARNING)
RESULT_DIR = PROJECT.TEST_RESULTS_PATH / (PROJECT.TEST_RESULTS_PATH / __file__).stem
RESULT_DIR.mkdir(parents=True, exist_ok=True)


def compute_linode_error(
    *,
    num: Optional[int] = None,
    dim: Optional[int] = None,
    precision: Literal["single", "double"] = "single",
    relative_error: bool = True,
    device: Optional[torch.device] = None,
) -> NDArray:
    r"""Compare `LinODE` against `scipy.odeint` on linear system."""
    numpy_dtype: type[np.number]
    torch_dtype: torch.dtype

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

    num = num or random.choice([10 * k for k in range(1, 11)])
    dim = dim or random.choice([2**k for k in range(1, 8)])
    t0, t1 = np.random.uniform(low=-10, high=10, size=(2,))
    A = (np.random.randn(dim, dim) / np.sqrt(dim)).astype(numpy_dtype)
    x0 = np.random.randn(dim).astype(numpy_dtype)
    T = np.random.uniform(low=t0, high=t1, size=num - 2)
    T = np.sort([t0, *T, t1]).astype(numpy_dtype)

    def func(_, x):
        return A @ x

    X = np.array(odeint(func, x0, T, tfirst=True))

    # A_torch = torch.tensor(A, dtype=torch_dtype, device=device)
    T_torch = torch.tensor(T, dtype=torch_dtype, device=device)
    x0_torch = torch.tensor(x0, dtype=torch_dtype, device=device)

    model = LinODE(input_size=dim, cell={"kernel_initialization": A, "scalar": 1.0})
    model.to(dtype=torch_dtype, device=device)

    Xhat = model(T_torch, x0_torch)
    Xhat = Xhat.clone().detach().cpu()

    err = (X - Xhat).abs()

    if relative_error:
        err /= abs(X) + eps

    result = np.array([scaled_norm(err, p=p) for p in (1, 2, np.inf)])
    return result


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

    for err, tol in zip(err_single, (10.0**k for k in (0, 2, 4))):
        q = np.nanquantile(err, 0.99)
        LOGGER.info("99%% quantile %f", q)
        assert q <= tol, f"99% quantile {q=} larger than allowed {tol=}"
    # Note that the matching of the predictions is is 4 order of magnitude better in FP64.
    # Since 10^4 ~ 2^13
    for err, tol in zip(err_double, (10.0**k for k in (-4, -2, -0))):
        q = np.nanquantile(err, 0.99)
        LOGGER.info("99%% quantile %f", q)
        assert q <= tol, f"99% quantile {q=} larger than allowed  {tol=}"
    LOGGER.info("%s passes test ✔ ", LinODE)

    if not make_plots:
        return

    with plt.style.context("bmh"):
        fig, ax = plt.subplots(
            ncols=3,
            nrows=2,
            figsize=(10, 5),
            tight_layout=True,
            sharey="row",
            sharex="all",
        )

    LOGGER.info("generating figure")
    for i, err in enumerate((err_single, err_double)):
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
    LOGGER.info("all done")


def _main() -> None:
    test_linode_error(num_samples=1000, make_plots=True)


if __name__ == "__main__":
    _main()
