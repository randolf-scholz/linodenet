r"""Test whether the initializations satisfy the advertised properties."""

import logging

import torch

from linodenet.initializations import INITIALIZATIONS

LOGGER = logging.getLogger(__name__)


def test_all_initializations(
    num_runs: int = 1000, num_samples: int = 1000, dim: int = 100
):
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
    LOGGER.info("Testing all available initializations %s", set(INITIALIZATIONS))

    if torch.cuda.is_available():
        torch.set_default_tensor_type(torch.cuda.FloatTensor)  # type: ignore
    else:
        torch.set_default_tensor_type(torch.FloatTensor)  # type: ignore

    ZERO = torch.tensor(0.0)
    ONE = torch.tensor(1.0)
    x = torch.randn(num_runs, num_samples, dim)

    for key, initialization in INITIALIZATIONS.items():

        LOGGER.info("Testing %s", key)

        # Batch compute A⋅x for num_samples of x and num_runs many samples of A
        matrices = initialization((num_runs, dim))  # num_runs many dim×dim matrices.
        y = torch.einsum("rkl, rnl -> rnk", matrices, x)
        y = y.flatten(start_dim=1)
        means = torch.mean(y, dim=-1)
        stdvs = torch.std(y, dim=-1)

        # check if 𝐄[A⋅x] ≈ 0
        valid_mean = torch.isclose(means, ZERO, rtol=1e-2, atol=1e-2).float().mean()
        assert valid_mean > 0.9, f"Only {valid_mean=:.2%} of means were clsoe to 0!"
        LOGGER.info("%s of means are close to 0 ✔ ", f"{valid_mean=:.2%}")

        # check if 𝐕[A⋅x] ≈ 1
        valid_stdv = torch.isclose(stdvs, ONE, rtol=1e-2, atol=1e-2).float().mean()
        assert valid_stdv > 0.9, f"Only {valid_mean=:.2%} of stdvs were clsoe to 1!"
        LOGGER.info("%s of stdvs are close to 1 ✔ ", f"{valid_stdv=:.2%}")

    # todo: add plot
    # todo: add experiment after applying matrix exponential

    LOGGER.info("All initializations passed! ✔ ")


def __main__():
    logging.basicConfig(level=logging.INFO)
    LOGGER.info("Testing INITIALIZATIONS started!")
    test_all_initializations()
    LOGGER.info("Testing INITIALIZATIONS finished!")


if __name__ == "__main__":
    __main__()
