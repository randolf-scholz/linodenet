r"""Test whether the initializations satisfy the advertised properties."""

import logging
import warnings

import matplotlib.pyplot as plt
import psutil
import pytest
import torch

from linodenet.config import PROJECT
from linodenet.constants import ONE, ZERO
from linodenet.initializations import INITIALIZATIONS
from linodenet.testing import MATRIX_TESTS

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


@pytest.mark.flaky(reruns=3)
@pytest.mark.parametrize("init_name", INITIALIZATIONS)
def test_normalization_property(
    init_name: str,
    make_plots: bool,
    *,
    num_runs: int = 64,
    num_samples: int = 1024,
    dim: int = 128,
) -> None:
    r"""Test normalization property empirically for all initializations."""
    LOGGER = logging.getLogger(init_name)
    LOGGER.info("Testing...")

    if psutil.virtual_memory().available < 16 * 1024**3:
        warnings.warn("Requires up to 16GiB of RAM", UserWarning, stacklevel=2)

    # initialize matrices
    kwargs: dict = {}
    if init_name == "low_rank":
        kwargs["rank"] = max(1, dim // 2)  # with rank-1, too unstable

    initialization = INITIALIZATIONS[init_name]
    matrices = initialization((num_runs, dim), **kwargs)  # (n_runs, dim, dim)

    # Batch compute A⋅x for num_samples of x and num_runs many samples of A
    x = torch.randn(num_runs, num_samples, dim)
    y = torch.einsum("...kl, ...nl -> ...nk", matrices, x)  # (n_runs, n_samples, dim)
    y = y.flatten(start_dim=1)  # (n_runs, n_samples * dim)
    means = torch.mean(y, dim=-1)  # (n_runs, )
    stdvs = torch.std(y, dim=-1)  # (n_runs, )

    # save results
    if make_plots:
        _make_fig(RESULT_DIR, means, stdvs, init_name)

    # check if 𝐄[A⋅x] ≈ 0
    valid_mean = torch.isclose(means, ZERO, rtol=1e-2, atol=1e-2).float().mean()
    assert valid_mean > 0.9, f"Only {valid_mean=:.2%} of means were close to 0!"
    LOGGER.info("%s of means are close to 0 ✔ ", f"{valid_mean=:.2%}")

    # check if 𝐕[A⋅x] ≈ 1
    valid_stdv = torch.isclose(stdvs, ONE, rtol=1e-2, atol=1e-2).float().mean()
    assert valid_stdv > 0.9, f"Only {valid_stdv=:.2%} of stdvs were close to 1!"
    LOGGER.info("%s of stdvs are close to 1 ✔ ", f"{valid_stdv=:.2%}")


@pytest.mark.repeat(10)
@pytest.mark.parametrize("init_name", INITIALIZATIONS)
def test_validity_initializations(init_name: str) -> None:
    """Validate that the initializations give correct matrix properties."""
    test_name = f"is_{init_name}"
    if test_name not in MATRIX_TESTS:
        pytest.skip(f"Test {test_name} not implemented.")

    initialization = INITIALIZATIONS[init_name]
    matrix_test = MATRIX_TESTS[test_name]

    shape = 4

    matrix = initialization(shape)
    assert matrix_test(matrix), f"{init_name} failed test {test_name}\n{matrix=}."


@pytest.mark.skip
def test_all_initializations(make_plots: bool) -> None:
    r"""Test all initializations."""
    __logger__.info("Testing initializations %s", set(INITIALIZATIONS))
    for key in INITIALIZATIONS:
        test_normalization_property(key, make_plots=make_plots)
    __logger__.info("All initializations passed! ✔ ")
