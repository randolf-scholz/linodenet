r"""Test error of linear ODE against odeint."""

import datetime
import logging
import random
import subprocess
from typing import Literal, Optional

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
from numpy.typing import NDArray
from scipy.integrate import solve_ivp
from tqdm.autonotebook import trange

from linodenet.state_propagation.flows import LinearFlow
from linodenet_special import scaled_norm
from tests.testing import PROJECT, visualize_distribution

RESULT_DIR = PROJECT.RESULTS_DIR[__file__]
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("PIL").setLevel(logging.WARNING)


def compute_linode_error(
    *,
    num: Optional[int] = None,
    dim: Optional[int] = None,
    precision: Literal["single", "double"] = "single",
    relative_error: bool = True,
    device: Optional[str | torch.device] = None,
) -> tuple[float, float, float]:
    r"""Compare `LinODE` against `scipy.odeint` on linear system.

    .. Signature:: `` -> (q, N)``
    """
    N = num or random.choice([10 * k for k in range(2, 11)])
    D = dim or random.choice([2**k for k in range(1, 8)])
    logger = logging.getLogger(f"{__name__}/{LinearFlow.__name__}-test-{N}-{D}")

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

    A = (rng.normal(size=(D, D)) / np.sqrt(D)).astype(numpy_dtype)
    x0: NDArray = rng.normal(size=(D,)).astype(numpy_dtype)

    t0, t1 = rng.uniform(low=-10, high=10, size=(2,)).astype(numpy_dtype)
    t0, t1 = min(t0, t1), max(t0, t1) + 1e-5  # make sure t0 < t1
    t_span = rng.uniform(low=t0, high=t1, size=N - 2).astype(numpy_dtype)
    t_span = np.unique(np.concatenate(([t0], t_span, [t1])))

    def func(_: NDArray, x: NDArray) -> NDArray:
        return A @ x

    sol = solve_ivp(
        func,
        [t0, t1],
        y0=x0,
        t_eval=t_span,
        vectorized=True,
    )
    assert sol.y.shape == (D, len(t_span)), "Shape mismatch"

    X = torch.tensor(sol.y.T, dtype=torch_dtype)

    # A_torch = torch.tensor(A, dtype=torch_dtype, device=device)
    T_torch = torch.tensor(t_span, dtype=torch_dtype, device=device)
    x0_torch = torch.tensor(x0, dtype=torch_dtype, device=device)

    flow = LinearFlow(
        input_size=D,
        kernel_initialization=A,
        scalar=1.0,
        scalar_learnable=False,
    )
    flow.to(dtype=torch_dtype, device=device)
    # assert model.cell.scalar == 1.0

    Xhat = flow.forecast(T_torch, x0_torch, t0=t0)
    Xhat = Xhat.clone().detach().cpu()

    residual = (X - Xhat).abs()

    if relative_error:
        residual /= X.abs() + eps

    # NOTE: shape:
    logger.debug("shapes: X:%s Xhat:%s err:5%s", X.shape, Xhat.shape, residual.shape)
    return (
        float(scaled_norm(residual, p=1, keepdim=False)),
        float(scaled_norm(residual, p=2, keepdim=False)),
        float(scaled_norm(residual, p=np.inf, keepdim=False)),
    )


def make_error_plots(
    *,
    error_single: NDArray,
    error_double: NDArray,
    logger: logging.Logger,
) -> None:
    r"""Create histogram plot of the errors."""
    assert error_single.shape == error_double.shape, "Single and double shape mismatch"
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
            visualize_distribution(err[j], log=True, ax=ax[i, j])
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

    # add current date and time to the figure
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    fig.text(0.99, 0.01, now, ha="right", va="bottom", fontsize=8, color="gray")

    # add current git commit hash to the figure
    try:
        git_hash = (
            subprocess.check_output(["/usr/bin/git", "rev-parse", "--short", "HEAD"])
            .strip()
            .decode("utf-8")
        )
    except Exception:
        logger.exception("Could not get git hash")
    else:
        fig.text(
            0.01,
            0.01,
            f"git:{git_hash}",
            ha="left",
            va="bottom",
            fontsize=8,
            color="gray",
        )

    fig.suptitle(
        r"Difference $x^{\text{(LinODE)}}$ and $x^{\text{(odeint)}}$"
        f" -- {num_samples} random systems"
    )

    fig.savefig(RESULT_DIR / "LinODE_odeint_comparison.pdf")
    fig.savefig(RESULT_DIR / "LinODE_odeint_comparison.svg")
    fig.savefig(RESULT_DIR / "LinODE_odeint_comparison.png", dpi=300)


@pytest.mark.flaky(reruns=3)
@pytest.mark.parametrize("num_samples", [100], ids=lambda n_samples: f"{n_samples=}")
@pytest.mark.parametrize("quantile", [0.95], ids=lambda q: f"{q=}")
@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("precision", ["single", "double"])
def test_linode_error(
    *,
    precision: Literal["single", "double"],
    device: str,
    quantile: float,
    num_samples: int,
) -> None:
    r"""Compare LinODE against scipy.odeint on random linear system."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    match precision:
        case "single":
            tolerances = (10**-2, 10**-1, 10**1)
        case "double":
            tolerances = (10**-2, 10**-1, 10**1)
        case _:
            raise AssertionError(f"Unknown precision {precision}")

    logger = logging.getLogger(f"{__name__}/{LinearFlow.__name__}")
    logger.info("Testing %s.", LinearFlow)

    logger.info(f"Generating {num_samples} samples in {precision} precision")
    errors = np.array(
        [
            compute_linode_error(precision=precision, device=device)
            for _ in trange(num_samples)
        ],
        dtype=np.float32,
    ).T

    for err, tol in zip(errors, tolerances, strict=True):
        # we want that the error is smaller than the tolerance in 95% of the cases
        q = np.nanquantile(err, quantile)
        logger.info(f"{quantile}% quantile {q}")
        assert q <= tol, f"{quantile} quantile {q=} larger than allowed {tol=}"
        if 100 * q < tol:
            raise AssertionError(
                f"The tolerance seems too loose: {quantile} quantile {q} << {tol}"
            )
    logger.info("%s passes test ✔ ", LinearFlow)


@pytest.mark.slow
def test_make_error_plot(num_samples: int = 100) -> None:  # noqa: PT028
    logger = logging.getLogger(f"{__name__}/{LinearFlow.__name__}")
    logger.info("Testing %s.", LinearFlow)

    logger.info(f"Generating {num_samples} samples in single precision")
    err_single = np.array(
        [compute_linode_error(precision="single") for _ in trange(num_samples)],
        dtype=np.float32,
    ).T

    logger.info(f"Generating {num_samples} samples in double precision")
    err_double = np.array(
        [compute_linode_error(precision="double") for _ in trange(num_samples)],
        dtype=np.float64,
    ).T

    make_error_plots(
        error_single=err_single,
        error_double=err_double,
        logger=logger,
    )


if __name__ == "__main__":
    test_make_error_plot()
