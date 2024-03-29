r"""Test whether the initializations satisfy the advertised properties."""

import logging
import warnings

import matplotlib.pyplot as plt
import psutil
import pytest
import torch

from linodenet.config import PROJECT
from linodenet.initializations import INITIALIZATIONS

RESULT_DIR = PROJECT.RESULTS_DIR[__file__]

__logger__ = logging.getLogger(__name__)


def _make_fig(path, means, stdvs, key):
    with plt.style.context("bmh"):
        fig, ax = plt.subplots(
            ncols=2, figsize=(8, 4), constrained_layout=True, sharey=True
        )
        ax[0].hist(means.cpu().numpy(), bins="auto", density=True, log=True)
        ax[0].set_title("Mean across multiple random inits.")
        ax[1].hist(stdvs.cpu().numpy(), bins="auto", density=True, log=True)
        ax[1].set_title("Std. across multiple random inits.")
        ax[0].set_ylim((10**0, 10**3))
        ax[0].set_xlim((-0.01, +0.01))
        ax[1].set_xlim((0.85, 1.15))
        # ax[1].set_xscale("log", base=2)
        fig.suptitle(f"{key}")
        fig.supylabel("log-odds")
        fig.savefig(path / f"{key}.svg")


@pytest.mark.parametrize("key", INITIALIZATIONS)
@pytest.mark.flaky(reruns=3)
def test_initialization(
    key: str,
    make_plots: bool,
    *,
    num_runs: int = 64,
    num_samples: int = 1024,
    dim: int = 128,
) -> None:
    r"""Test normalization property empirically for all initializations.

    Args:
        key: the initialization to test
        num_runs: Number of repetitions
        num_samples: Number of samples
        dim: Number of dimensions
        make_plots: Whether to plot
    """
    initialization = INITIALIZATIONS[key]
    LOGGER = logging.getLogger(initialization.__name__)
    LOGGER.info("Testing...")

    if psutil.virtual_memory().available < 16 * 1024**3:
        warnings.warn("Requires up to 16GiB of RAM", UserWarning, stacklevel=2)

    ZERO = torch.tensor(0.0)
    ONE = torch.tensor(1.0)
    x = torch.randn(num_runs, num_samples, dim)
    # Batch compute A⋅x for num_samples of x and num_runs many samples of A
    matrices = initialization((num_runs, dim))  # (num_runs, dim, dim)
    y = torch.einsum(
        "...kl, ...nl -> ...nk", matrices, x
    )  # (num_runs, num_samples, dim)
    y = y.flatten(start_dim=1)  # (num_runs, num_samples * dim)
    means = torch.mean(y, dim=-1)  # (num_runs, )
    stdvs = torch.std(y, dim=-1)  # (num_runs, )

    # save results
    if make_plots:
        _make_fig(RESULT_DIR, means, stdvs, key)

    # check if 𝐄[A⋅x] ≈ 0
    valid_mean = torch.isclose(means, ZERO, rtol=1e-8, atol=1e-2).float().mean()
    assert valid_mean > 0.9, f"Only {valid_mean=:.2%} of means were close to 0!"
    LOGGER.info("%s of means are close to 0 ✔ ", f"{valid_mean=:.2%}")

    # check if 𝐕[A⋅x] ≈ 1
    valid_stdv = torch.isclose(stdvs, ONE, rtol=1e-2, atol=1e-2).float().mean()
    assert valid_stdv > 0.9, f"Only {valid_stdv=:.2%} of stdvs were close to 1!"
    LOGGER.info("%s of stdvs are close to 1 ✔ ", f"{valid_stdv=:.2%}")

    # todo: add plot after applying matrix exponential


@pytest.mark.skip
def test_all_initializations(make_plots: bool) -> None:
    r"""Test all initializations."""
    __logger__.info("Testing initializations %s", set(INITIALIZATIONS))
    for key in INITIALIZATIONS:
        test_initialization(key, make_plots=make_plots)
    __logger__.info("All initializations passed! ✔ ")
