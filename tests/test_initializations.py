import logging

import numpy as np
import torch

from linodenet.init import INITS

logger = logging.getLogger(__name__)


def test_all_initializations(rep: int = 10000, num: int = 1000, dim: int = 100):
    r"""

    Note: requires up to 16 GB RAM with default settings.

    Parameters
    ----------
    rep: int, default=10000
        Number of repetitions
    num: int: default=1000
        Number of samples
    dim: int, default=100
        Number of dimensions

    Returns
    -------
    """

    logger.info(F"Testing all available initializations {set(INITS)}")

    x = torch.randn(rep, num, dim)

    for key, method in INITS.items():

        logger.info(F"Testing {key}")

        mats = method((rep, dim))
        y = torch.einsum('rkl, rnl -> rnk', mats, x)
        y = y.flatten(start_dim=1).numpy()
        means = np.mean(y, axis=-1)
        stdvs = np.std(y, axis=-1)

        assert np.isclose(means, 0, rtol=1e-2, atol=1e-2).mean() > 0.9, F"{means=} not close to zero!"
        assert np.isclose(stdvs, 1, rtol=1e-2, atol=1e-2).mean() > 0.9, F"{stdvs=} not close to one!"

        logger.info(F"{key} passed!")

    print(F"All initializations {set(INITS)} passes!")


if __name__ == "__main__":
    test_all_initializations()
