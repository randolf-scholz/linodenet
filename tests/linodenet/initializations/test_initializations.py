r"""Test whether the initializations satisfy the advertised properties."""

import logging
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import psutil
import pytest
import torch
from torch import Tensor

from linodenet.initializations import INITIALIZATION_FNS
from linodenet.registry import get_registry_entry
from tests.testing import PROJECT

RESULT_DIR = PROJECT.RESULTS_DIR[__file__]


def _make_fig(path: Path, means: Tensor, stdvs: Tensor, key: str) -> None:
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
@pytest.mark.parametrize("num_runs", [64], ids=lambda n_runs: f"{n_runs=}")
@pytest.mark.parametrize("num_samples", [1024], ids=lambda n_samples: f"{n_samples=}")
@pytest.mark.parametrize("dim", [128], ids=lambda dim: f"{dim=}")
@pytest.mark.parametrize("name", INITIALIZATION_FNS)
def test_normalization_property(
    *,
    name: str,
    dim: int,
    num_runs: int,
    num_samples: int,
    make_plots: bool,
) -> None:
    r"""Test normalization property empirically for all initializations."""
    logger = logging.getLogger(name)
    logger.info("Testing...")

    if psutil.virtual_memory().available < 16 * 1024**3:
        warnings.warn("Requires up to 16GiB of RAM", UserWarning, stacklevel=2)

    if name == "thomson":
        pytest.skip("Thomson initialization samples sphere points, not matrices.")

    # initialize matrices
    kwargs: dict = {}
    matrix_dim: int | tuple[int, int] = dim
    if name == "gaussian":
        matrix_dim = (dim, dim)
    if name == "low_rank":
        kwargs["rank"] = max(1, dim // 2)  # with rank-1, too unstable

    initialization = INITIALIZATION_FNS[name]
    matrices = initialization((num_runs,), matrix_dim, **kwargs)  # (n_runs, dim, dim)
    assert matrices.shape == (num_runs, dim, dim)

    # Batch compute A⋅x for num_samples of x and num_runs many samples of A
    x = torch.randn(num_runs, num_samples, dim)
    y = torch.einsum("...kl, ...nl -> ...nk", matrices, x)  # (n_runs, n_samples, dim)
    y = y.flatten(start_dim=1)  # (n_runs, n_samples * dim)
    means = torch.mean(y, dim=-1)  # (n_runs, )
    stdvs = torch.std(y, dim=-1)  # (n_runs, )

    # save results
    if make_plots:
        _make_fig(RESULT_DIR, means, stdvs, name)

    # check if 𝐄[A⋅x] ≈ 0
    zeros = torch.zeros_like(means)
    valid_mean = torch.isclose(means, zeros, rtol=1e-2, atol=1e-2).float().mean()
    assert valid_mean > 0.9, f"Only {valid_mean=:.2%} of means were close to 0!"
    logger.info("%s of means are close to 0 ✔ ", f"{valid_mean=:.2%}")

    # check if 𝐕[A⋅x] ≈ 1
    ones = torch.ones_like(stdvs)
    valid_stdv = torch.isclose(stdvs, ones, rtol=1e-2, atol=1e-2).float().mean()
    threshold = 0.85 if name == "low_rank" else 0.9
    assert valid_stdv > threshold, f"Only {valid_stdv=:.2%} of stdvs were close to 1!"
    logger.info("%s of stdvs are close to 1 ✔ ", f"{valid_stdv=:.2%}")


@pytest.mark.repeat(10)
@pytest.mark.parametrize("name", INITIALIZATION_FNS)
def test_validity_initializations(name: str) -> None:
    r"""Validate that the initializations give correct matrix properties."""
    entry = get_registry_entry(name)
    if not callable(entry.test):
        pytest.skip(f"No registry test registered for {name}.")

    initialization = INITIALIZATION_FNS[name]
    matrix_test = entry.test

    size = 4

    if name == "thomson":
        pytest.skip("Thomson initialization samples sphere points, not matrices.")

    kwargs: dict = {}
    if name == "low_rank":
        kwargs["rank"] = 2

    matrix = initialization((), size, **kwargs)
    assert matrix.shape == (size, size)
    try:
        result = matrix_test(matrix)
    except TypeError as exc:
        raise pytest.skip(
            f"Test {matrix_test.__name__} for {name} requires additional arguments."
        ) from exc

    assert result.item(), f"{name} failed test {matrix_test.__name__}\n{matrix=}."
