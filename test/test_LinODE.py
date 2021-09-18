r"""Test error of linear ODE against odeint."""

import logging

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.integrate import odeint
from tqdm.auto import trange

from linodenet.models import LinODE
from tsdm.plot import visualize_distribution
from tsdm.util import scaled_norm

logger = logging.getLogger(__name__)
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("PIL").setLevel(logging.WARNING)


def linode_error(
    dim=None,
    num=None,
    precision="single",
    relative_error=True,
    device=None,
):
    r"""Compare LinODE against scipy.odeint on linear system."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if precision == "single":
        eps = 2 ** -24
        numpy_dtype = np.float32
        torch_dtype = torch.float32
    elif precision == "double":
        eps = 2 ** -53
        numpy_dtype = np.float64  # type: ignore
        torch_dtype = torch.float64
    else:
        raise ValueError

    num = np.random.randint(low=20, high=1000) or num
    dim = np.random.randint(low=2, high=100) or dim
    t0, t1 = np.random.uniform(low=-10, high=10, size=(2,))
    A = (np.random.randn(dim, dim) / np.sqrt(dim)).astype(numpy_dtype)
    x0 = np.random.randn(dim).astype(numpy_dtype)
    T = np.random.uniform(low=t0, high=t1, size=num - 2)
    T = np.sort([t0, *T, t1]).astype(numpy_dtype)

    def func(t, x):  # noqa
        return A @ x

    X = np.array(odeint(func, x0, T, tfirst=True))

    A = torch.tensor(A, dtype=torch_dtype, device=device)  # type: ignore
    T = torch.tensor(T, dtype=torch_dtype, device=device)  # type: ignore
    x0 = torch.tensor(x0, dtype=torch_dtype, device=device)  # type: ignore

    model = LinODE(input_size=dim, kernel_initialization=A)
    model.to(dtype=torch_dtype, device=device)

    Xhat = model(T, x0)
    Xhat = Xhat.clone().detach().cpu().numpy()

    err = np.abs(X - Xhat)

    if relative_error:
        err /= np.abs(X) + eps

    result = np.array([scaled_norm(err, p=p) for p in (1, 2, np.inf)])
    return result


def test_linode_error(nsamples=100, make_plot=False):
    r"""Compare LinODE against scipy.odeint on linear system."""
    logger.info("Testing LinODE")
    logger.info("Generating %i samples in single precision", nsamples)
    err_single = np.array(
        [linode_error(precision="single") for _ in trange(nsamples)]
    ).T
    logger.info("Generating %i samples in double precision", nsamples)
    err_double = np.array(
        [linode_error(precision="double") for _ in trange(nsamples)]
    ).T

    for err, tol in zip(err_single, (10 ** 0, 10 ** 2, 10 ** 4)):
        q = np.nanquantile(err, 0.99)
        logger.info("99%% quantile %f", q)
        assert q <= tol, f"99% quantile {q=} larger than allowed {tol=}"
    # Note that the matching of the predictions is is 4 order of magnitude better in FP64.
    # Since 10^4 ~ 2^13
    for err, tol in zip(err_double, (10 ** -4, 10 ** -2, 10 ** 0)):
        q = np.nanquantile(err, 0.99)
        logger.info("99%% quantile %f", q)
        assert q <= tol, f"99% quantile {q=} larger than allowed  {tol=}"
    logger.info("LinODE passes test \N{HEAVY CHECK MARK}")

    if not make_plot:
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

    logger.info("LinODE generating figure")
    for i, err in enumerate((err_single, err_double)):
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

    fig.suptitle(
        r"Discrepancy between `x^\text{(LinODE)}` and `x^\text{(odeint)}`, "
        f"{nsamples} samples"
    )
    fig.savefig("LinODE_odeint_comparison.svg")
    logger.info("LinearContraction all done")


if __name__ == "__main__":
    test_linode_error(nsamples=1000, make_plot=True)
