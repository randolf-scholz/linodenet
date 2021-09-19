r"""Test the iResNet components.

1. is the LinearContraction layer really a linear contraction?
2. is the iResNet really invertible via fixed-point iteration?
"""

import logging
import random
from typing import Optional

import matplotlib.pyplot as plt
import torch
from torch import Tensor

from linodenet.models import LinearContraction, iResNetBlock
from tsdm.plot import visualize_distribution
from tsdm.util import scaled_norm

LOGGER = logging.getLogger(__name__)  # noqa: E402


def test_LinearContraction(
    num_sample: Optional[int] = None,
    dim_inputs: Optional[int] = None,
    dim_output: Optional[int] = None,
    make_plot: bool = False,
):
    r"""Test empirically if the LinearContraction really is a linear contraction.

    Parameters
    ----------
    num_sample: Optional[int] = None
        default: sample randomly from [1000, 2000, ..., 5000]
    dim_inputs: Optional[int] = None
        default: sample randomly from [2, 4, 8, , .., 128]
    dim_output: Optional[int] = None
            default: sample randomly from [2, 4, 8, , .., 128]
    make_plot: bool
    """
    if torch.cuda.is_available():
        torch.set_default_tensor_type(torch.cuda.FloatTensor)  # type: ignore
        LOGGER.info("Using CUDA")
    else:
        torch.set_default_tensor_type(torch.FloatTensor)  # type: ignore

    LOGGER.info(">>> Testing LinearContraction <<<")
    num_sample = num_sample or random.choice([1000 * k for k in range(1, 6)])
    dim_inputs = dim_inputs or random.choice([2 ** k for k in range(2, 8)])
    dim_output = dim_output or random.choice([2 ** k for k in range(2, 8)])
    extra_stats = {
        "Samples": f"{num_sample}",
        "Dim-in": f"{dim_inputs}",
        "Dim-out": f"{dim_output}",
    }
    LOGGER.info("Configuration: %s", extra_stats)

    x = torch.randn(num_sample, dim_inputs)
    y = torch.randn(num_sample, dim_inputs)
    distances: Tensor = torch.cdist(x, y)

    model = LinearContraction(dim_inputs, dim_output)
    xhat = model(x)
    yhat = model(y)
    latent_distances: Tensor = torch.cdist(xhat, yhat)

    # Test whether contraction property holds
    assert torch.all(latent_distances <= distances)
    LOGGER.info("LinearContraction passes test \N{HEAVY CHECK MARK}")

    if not make_plot:
        return

    LOGGER.info("LinearContraction generating figure")
    scaling_factor = (latent_distances / distances).flatten()

    # TODO: Switch to PGF backend with matplotlib 3.5 release
    fig, ax = plt.subplots(figsize=(5.5, 3.4), tight_layout=True)
    ax.set_title(r"LinearContraction -- Scaling Factor Distribution")
    ax.set_xlabel(
        r"$s(x, y) = \frac{\|\phi(x)-\phi(y)\|}{\|x-y\|}$ where "
        r"$x_i, y_j \overset{\text{i.i.d}}{\sim} \mathcal N(0, 1)$"
    )
    ax.set_ylabel(r"density $p(s \mid x, y)$")

    visualize_distribution(scaling_factor, ax=ax, extra_stats=extra_stats)

    fig.savefig("LinearContraction_ScalingFactor.pdf")
    LOGGER.info("LinearContraction all done")


def test_iResNetBlock(
    num_sample: Optional[int] = None,
    dim_inputs: Optional[int] = None,
    dim_output: Optional[int] = None,
    maxiter: int = 20,
    make_plot: bool = False,
    quantiles: tuple[float, ...] = (0.5, 0.68, 0.95, 0.997),
    targets: tuple[float, ...] = (0.005, 0.005, 0.01, 0.01),
):
    r"""Test empirically whether the iResNetBlock is indeed invertible.

    Parameters
    ----------
    num_sample: Optional[int] = None
        default: sample randomly from [1000, 2000, ..., 10000]
    dim_inputs: Optional[int] = None
        default: sample randomly from [2, 4, 8, , .., 128]
    dim_output: Optional[int] = None
        default: sample randomly from [2, 4, 8, , .., 128]
    make_plot: bool
    quantiles: tuple[float, ...]
        The quantiles of the error distribution
    targets: tuple[float, ...]
        The target values for the quantiles of the error distribution
    """
    LOGGER.info(">>> Testing iResNetBlock <<<")
    num_sample = num_sample or random.choice([1000 * k for k in range(1, 11)])
    dim_inputs = dim_inputs or random.choice([2 ** k for k in range(2, 8)])
    dim_output = dim_output or random.choice([2 ** k for k in range(2, 8)])
    extra_stats = {
        "Samples": f"{num_sample}",
        "Dim-in": f"{dim_inputs}",
        "Dim-out": f"{dim_output}",
        "maxiter": f"{maxiter}",
    }
    if torch.cuda.is_available():
        torch.set_default_tensor_type(torch.cuda.FloatTensor)  # type: ignore
        LOGGER.info("Using CUDA")
    else:
        torch.set_default_tensor_type(torch.FloatTensor)  # type: ignore

    QUANTILES = torch.tensor(quantiles)
    TARGETS = torch.tensor(targets)

    LOGGER.info("Configuration: %s", extra_stats)
    LOGGER.info("QUANTILES: %s", QUANTILES)
    LOGGER.info("TARGETS  : %s", TARGETS)

    with torch.no_grad():
        model = iResNetBlock(dim_inputs, hidden_size=dim_output, maxiter=maxiter)
        x = torch.randn(num_sample, dim_inputs)
        y = torch.randn(num_sample, dim_inputs)
        fx = model(x)
        xhat = model.inverse(fx)
        ify = model.inverse(y)
        yhat = model(ify)

    # Test if ϕ⁻¹∘ϕ=id, i.e. the right inverse is working
    forward_inverse_error = scaled_norm(x - xhat, axis=-1)
    forward_inverse_quantiles = torch.quantile(forward_inverse_error, QUANTILES)
    assert forward_inverse_error.shape == (num_sample,)
    assert (forward_inverse_quantiles <= TARGETS).all(), f"{forward_inverse_quantiles=}"
    LOGGER.info("iResNetBlock satisfies ϕ⁻¹∘ϕ≈id \N{HEAVY CHECK MARK}")
    LOGGER.info("Quantiles: %s", forward_inverse_quantiles)

    # Test if ϕ∘ϕ⁻¹=id, i.e. the right inverse is working
    inverse_forward_error = scaled_norm(y - yhat, axis=-1)
    inverse_forward_quantiles = torch.quantile(forward_inverse_error, QUANTILES)
    assert inverse_forward_error.shape == (num_sample,)
    assert (inverse_forward_quantiles <= TARGETS).all(), f"{inverse_forward_quantiles=}"
    LOGGER.info("iResNetBlock satisfies ϕ∘ϕ⁻¹≈id \N{HEAVY CHECK MARK}")
    LOGGER.info("Quantiles: %s", inverse_forward_quantiles)

    # Test if ϕ≠id, i.e. the forward map is different from the identity
    forward_difference = scaled_norm(x - fx, axis=-1)
    forward_quantiles = torch.quantile(forward_difference, 1 - QUANTILES)
    assert forward_difference.shape == (num_sample,)
    assert (forward_quantiles >= TARGETS).all(), f"{forward_quantiles}"
    LOGGER.info("iResNetBlock satisfies ϕ≉id \N{HEAVY CHECK MARK}")
    LOGGER.info("Quantiles: %s", forward_quantiles)

    # Test if ϕ⁻¹≠id, i.e. the inverse map is different from an identity
    inverse_difference = scaled_norm(y - ify, axis=-1)
    inverse_quantiles = torch.quantile(inverse_difference, 1 - QUANTILES)
    assert inverse_difference.shape == (num_sample,)
    assert (inverse_quantiles >= TARGETS).all(), f"{inverse_quantiles}"
    LOGGER.info("iResNetBlock satisfies ϕ⁻¹≉id \N{HEAVY CHECK MARK}")
    LOGGER.info("Quantiles: %s", inverse_quantiles)

    if not make_plot:
        return

    LOGGER.info("iResNetBlock generating figure")
    fig, ax = plt.subplots(
        ncols=2, nrows=2, figsize=(8, 5), tight_layout=True, sharex="row", sharey="row"
    )

    visualize_distribution(forward_inverse_error, ax=ax[0, 0], extra_stats=extra_stats)
    visualize_distribution(inverse_forward_error, ax=ax[0, 1], extra_stats=extra_stats)
    visualize_distribution(forward_difference, ax=ax[1, 0], extra_stats=extra_stats)
    visualize_distribution(inverse_difference, ax=ax[1, 1], extra_stats=extra_stats)

    ax[0, 0].set_xlabel(
        r"$r_\text{left}(x) = \|x - \phi^{-1}(\phi(x))\|$  where $x_i \sim \mathcal N(0,1)$"
    )
    ax[0, 0].set_ylabel(r"density $p(r_\text{left} \mid x)$")
    ax[0, 1].set_xlabel(
        r"$r_\text{right}(y) = \|y - \phi(\phi^{-1}(y))\|$ where $y_j \sim \mathcal N(0,1)$"
    )
    ax[0, 1].set_ylabel(r"denisty $p(r_\text{right}\mid y)$")

    ax[1, 0].set_xlabel(
        r"$d_\text{left}(x) = \|x - \phi(x)\|$ where $x_i \sim \mathcal N(0,1)$"
    )
    ax[1, 0].set_ylabel(r"density $p(d_\text{left} \mid x)$")
    ax[1, 1].set_xlabel(
        r"$d_\text{right}(y) = \|y - \phi^{-1}(y)\|$ where $y_j \sim \mathcal N(0,1)$"
    )
    ax[1, 1].set_ylabel(r"density $p(d_\text{right} \mid y)$")
    fig.suptitle("iResNetBlock -- Inversion Property", fontsize=16)
    fig.savefig("iResNetBlock_inversion.pdf")
    LOGGER.info("iResNetBlock all done")


def __main__():
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.basicConfig(level=logging.INFO)

    LOGGER.info("Testing LinearContraction started!")
    test_LinearContraction(make_plot=True)
    LOGGER.info("Testing LinearContraction finished!")

    LOGGER.info("Testing iResNetBlock started!")
    test_iResNetBlock(make_plot=True)
    LOGGER.info("Testing iResNetBlock finished!")


if __name__ == "__main__":
    __main__()
