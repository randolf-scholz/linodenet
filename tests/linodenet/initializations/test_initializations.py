r"""Test whether the initializations satisfy the advertised properties."""

import logging
import warnings
from collections import defaultdict

import matplotlib.pyplot as plt
import psutil
import pytest
import torch

from linodenet.initializations import INITIALIZATION_FNS
from linodenet.registry import get_registry_entry
from tests.testing import PROJECT

RESULT_DIR = PROJECT.RESULTS_DIR[__file__]


@pytest.mark.parametrize("name", INITIALIZATION_FNS)
class TestCorrectness:
    r"""Validate correctness properties of initialization functions."""

    BATCH_SIZE = 10

    @pytest.mark.parametrize("dim", [128], ids=lambda dim: f"{dim}x{dim}")
    @pytest.mark.parametrize("num_samples", [1024], ids=lambda size: f"{size=}")
    @pytest.mark.parametrize("num_runs", [64], ids=lambda n_runs: f"{n_runs=}")
    def test_normalization_property(
        self,
        *,
        name: str,
        dim: int,
        num_runs: int,
        num_samples: int,
    ) -> None:
        r"""Test normalization property empirically for all initializations."""
        logger = logging.getLogger(name)
        logger.info("Testing...")
        if name == "thomson":
            pytest.skip("Thomson initialization samples sphere points, not matrices.")
        if psutil.virtual_memory().available < 16 * 1024**3:
            warnings.warn("Requires up to 16GiB of RAM", UserWarning, stacklevel=2)

        KWARGS = defaultdict(dict)
        KWARGS["low_rank"] = {"rank": 2}

        matrix_dim: int | tuple[int, int] = dim
        if name == "gaussian":
            matrix_dim = (dim, dim)

        initialization = INITIALIZATION_FNS[name]
        matrices = initialization((num_runs,), matrix_dim, **KWARGS[name])  # (B, D, D)
        assert matrices.shape == (num_runs, dim, dim)

        # Batch compute A⋅x for num_samples of x and num_runs many samples of A
        x = torch.randn(num_runs, num_samples, dim)
        y = torch.einsum("...kl, ...nl -> ...nk", matrices, x)  # (B, N, D)
        y = y.flatten(start_dim=1)  # (B, N * D)
        means = torch.mean(y, dim=-1)  # (B, )
        stdvs = torch.std(y, dim=-1)  # (B, )

        # check if 𝐄[A⋅x] ≈ 0
        zeros = torch.zeros_like(means)
        valid_mean = torch.isclose(means, zeros, rtol=1e-2, atol=1e-2).float().mean()
        assert valid_mean > 0.9, f"Only {valid_mean=:.2%} of means were close to 0!"
        logger.info("%s of means are close to 0 ✔ ", f"{valid_mean=:.2%}")

        # check if 𝐕[A⋅x] ≈ 1
        ones = torch.ones_like(stdvs)
        valid_stdv = torch.isclose(stdvs, ones, rtol=1e-2, atol=1e-2).float().mean()
        threshold = 0.8 if name == "low_rank" else 0.9
        assert valid_stdv > threshold, (
            f"Only {valid_stdv=:.2%} of stdvs were close to 1!"
        )
        logger.info("%s of stdvs are close to 1 ✔ ", f"{valid_stdv=:.2%}")

    @pytest.mark.parametrize("size", [4])
    @pytest.mark.parametrize("batch_size", [10])
    def test_validity_initializations(
        self, name: str, size: int, batch_size: int
    ) -> None:
        r"""Validate that the initializations give correct matrix properties."""
        if name == "thomson":
            pytest.skip("Thomson initialization samples sphere points, not matrices.")
        entry = get_registry_entry(name)
        if not callable(matrix_test := entry.test):
            pytest.skip(f"No registry test registered for {name}.")

        KWARGS = defaultdict(dict)
        KWARGS["low_rank"] = {"rank": 2}
        initialization = INITIALIZATION_FNS[name]

        matrix = initialization(batch_size, (size, size), **KWARGS[name])
        assert matrix.shape == (batch_size, size, size)

        result = matrix_test(matrix, **KWARGS[name])
        assert torch.all(result).item()


class TestVisualization:
    r"""Exercise visualization-only branches for initialization diagnostics."""

    BATCH_SIZE = 256
    NUM_SAMPLES = 256
    INPUT_SIZE = 16

    @pytest.mark.parametrize("name", INITIALIZATION_FNS)
    def test_normalization_property(self, name: str) -> None:
        r"""Cover the plot-generation branch for normalization diagnostics."""
        B = self.BATCH_SIZE
        N = self.NUM_SAMPLES
        D = self.INPUT_SIZE
        KWARGS = defaultdict(dict)
        KWARGS["low_rank"] = {"rank": 2}

        initialization = INITIALIZATION_FNS[name]
        matrices = initialization((B,), D, **KWARGS[name])
        assert matrices.shape == (B, D, D)

        x = torch.randn(B, N, D)
        y = torch.einsum("...kl, ...nl -> ...nk", matrices, x)
        y = y.flatten(start_dim=1)
        means = torch.mean(y, dim=-1)
        stdvs = torch.std(y, dim=-1)

        with plt.style.context("bmh"):
            fig, ax = plt.subplots(
                ncols=2,
                figsize=(8, 4),
                constrained_layout=True,
            )
            ax[0].hist(means, density=True, log=True)
            ax[0].set_title("Mean across multiple random inits.")
            ax[1].hist(stdvs, density=True, log=True)
            ax[1].set_title("Std. across multiple random inits.")
            ax[0].set_xlim(-0.11, +0.11)
            ax[1].set_xlim(0.1, 1.9)
            ax[0].set_ylim(10**-1, 10**2)
            ax[1].set_ylim(10**-1, 10**2)
            fig.suptitle(f"{name}")
            fig.supylabel("log-odds")
            fig.savefig(RESULT_DIR / f"{name}.svg")
