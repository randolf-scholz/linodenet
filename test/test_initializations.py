r"""Test whether the initializations satisfy the advertised properties."""

import logging

import numpy as np
import torch

from linodenet.initializations import INITIALIZATIONS

logger = logging.getLogger(__name__)


def test_all_initializations(rep: int = 10000, num: int = 1000, dim: int = 100):
    r"""Test normalization property empirically for all initializations.

    Parameters
    ----------
    rep: int, default=10000
        Number of repetitions
    num: int: default=1000
        Number of samples
    dim: int, default=100
        Number of dimensions

    .. warning::
        Requires up to 16 GB RAM with default settings.
    """
    logger.info("Testing all available initializations %s", set(INITIALIZATIONS))

    x = torch.randn(rep, num, dim)

    for key, method in INITIALIZATIONS.items():

        logger.info("Testing %s", key)

        mats = method((rep, dim))
        y = torch.einsum("rkl, rnl -> rnk", mats, x)
        y = y.flatten(start_dim=1).numpy()
        means = np.mean(y, axis=-1)
        stdvs = np.std(y, axis=-1)

        assert (
            np.isclose(means, 0, rtol=1e-2, atol=1e-2).mean() > 0.9
        ), f"{means=} far from zero!"
        assert (
            np.isclose(stdvs, 1, rtol=1e-2, atol=1e-2).mean() > 0.9
        ), f"{stdvs=} far from one!"

        logger.info("%s passed!", key)

    # todo: add plot
    # todo: add experiment after applying matrix exponential

    logger.info(
        "All initializations %s passed \N{HEAVY CHECK MARK}", set(INITIALIZATIONS)
    )


if __name__ == "__main__":
    test_all_initializations()
