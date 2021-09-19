r"""Test whether the initializations satisfy the advertised properties."""

import logging

import numpy as np
import torch

from linodenet.initializations import INITIALIZATIONS

logger = logging.getLogger(__name__)

if torch.cuda.is_available():
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
else:
    torch.set_default_tensor_type(torch.FloatTensor)

ZERO = torch.tensor(0.0)
ONE  = torch.tensor(1.0)

def test_all_initializations(num_runs: int = 10000, num_samples: int = 1000, dim: int = 100):
    r"""Test normalization property empirically for all initializations.

    Parameters
    ----------
    num_runs: int, default=10000
        Number of repetitions
    num_samples: int: default=1000
        Number of samples
    dim: int, default=100
        Number of dimensions

    .. warning::
        Requires up to 16 GB RAM with default settings.
    """
    logger.info("Testing all available initializations %s", set(INITIALIZATIONS))

    x = torch.randn(num_runs, num_samples, dim)

    for key, initialization in INITIALIZATIONS.items():

        logger.info("Testing %s", key)

        # Batch compute A⋅x for num_samples of x and num_runs many samples of A
        matrices = initialization((num_runs, dim))  # num_runs many dim×dim matrices.
        y = torch.einsum("rkl, rnl -> rnk", matrices, x)
        y = y.flatten(start_dim=1)
        means = torch.mean(y, dim=-1)
        stdvs = torch.std(y, dim=-1)

        # Check if A⋅x ∼ 𝓝(0,1)
        assert (  # check if 𝐄[A⋅x] ≈ 0
            torch.isclose(means, ZERO, rtol=1e-2, atol=1e-2).float().mean() > 0.9
        ), f"{means=} far from zero!"
        assert (  # check if 𝐕[A⋅x] ≈ 1
            torch.isclose(stdvs, ONE, rtol=1e-2, atol=1e-2).float().mean() > 0.9
        ), f"{stdvs=} far from one!"

        logger.info("%s passed \N{HEAVY CHECK MARK}", key)

    # todo: add plot
    # todo: add experiment after applying matrix exponential

    logger.info(
        "All initializations %s passed \N{HEAVY CHECK MARK}", set(INITIALIZATIONS)
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    LOGGER.info("Testing INITIALIZATIONS started!")
    test_all_initializations()
    LOGGER.info("Testing INITIALIZATIONS finished!")
