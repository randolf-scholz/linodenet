"""
Test the iResNet components:

1. is the LinearContraction layer really a linear contraction?
2. is the iResNet really invertible via fixed-point iteration?
"""

import logging

import matplotlib.pyplot as plt
import numpy as np
import torch

from tsdm.plot import visualize_distribution
from tsdm.util import scaled_norm

from linodenet.models import LinearContraction, iResNetBlock

logger = logging.getLogger(__name__)
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('PIL').setLevel(logging.WARNING)


def test_LinearContraction(n_samples: int = 10_000, dim_in: int = None, dim_out: int = None):
    """
    Tests empirically whether the LinearContraction module is a contraction.
    """
    logger.info(">>> Testing LinearContraction <<<")
    n_samples = n_samples or np.random.randint(low=1000, high=10_000)
    dim_in = dim_in or np.random.randint(low=2, high=100)
    dim_out = dim_out or np.random.randint(low=2, high=100)
    logger.info("nsamples=%i, dim_in=%i, dim_out=%i", n_samples, dim_in, dim_out)

    x = torch.randn(n_samples, dim_in)
    y = torch.randn(n_samples, dim_in)
    distances = torch.cdist(x, y)

    model = LinearContraction(dim_in, dim_out)
    xhat = model(x)
    yhat = model(y)
    latent_distances = torch.cdist(xhat, yhat)

    assert torch.all(latent_distances <= distances)
    logger.info("LinearContraction passes test \N{HEAVY CHECK MARK}")

    logger.info("LinearContraction generating figure")
    scaling_factor = (latent_distances / distances).flatten()
    fig, ax = plt.subplots(figsize=(8, 4), tight_layout=True)
    visualize_distribution(scaling_factor, ax=ax)
    ax.set_title(F"LinearContraction -- Scaling Factor Distribution"
                 F" (samples:{n_samples}, dim-in:{dim_in}, dim-out:{dim_out})")
    ax.set_xlabel(r"$s(X, y) = \frac{\|\phi(X)-\phi(y)\|}{\|X-y\|}$")
    ax.set_ylabel(r"density $p(s\mid X, y)$ where $x_i,y_i\sim \mathcal N(0,1)$")
    fig.savefig("LinearContraction_ScalingFactor.png")
    logger.info("LinearContraction all done")


def test_iResNetBlock(n_samples: int = 10_000, input_size: int = None, hidden_size: int = None):
    """
    Tests empirically whether the iResNetBlock is indeed invertible.
    """

    logger.info(">>> Testing iResNetBlock <<<")
    n_samples = n_samples or np.random.randint(low=1000, high=10_000)
    input_size = input_size or np.random.randint(low=2, high=100)
    hidden_size = hidden_size or np.random.randint(low=2, high=100)
    logger.info("nsamples=%i, input_size=%i, hidden_size=%i", n_samples, input_size, hidden_size)

    # HP = {'hidden_size' : hidden_size}
    model = iResNetBlock(input_size, hidden_size=hidden_size)
    x = torch.randn(n_samples, input_size)
    y = torch.randn(n_samples, input_size)
    fx = model(x)
    xhat = model.inverse(fx)
    ify = model.inverse(y)
    yhat = model(ify)

    dist_lmap    = scaled_norm(x - fx, axis=-1)
    dist_rmap    = scaled_norm(y - ify, axis=-1)
    err_linverse = scaled_norm(x - xhat, axis=-1)
    err_rinverse = scaled_norm(y - yhat, axis=-1)

    assert torch.quantile(err_linverse, 0.99) <= 10 ** -6
    assert torch.quantile(err_rinverse, 0.99) <= 10 ** -6
    logger.info("iResNetBlock passes test \N{HEAVY CHECK MARK}")

    logger.info("iResNetBlock generating figure")
    fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(10, 5), tight_layout=True,
                           sharex='row', sharey='row')

    visualize_distribution(err_linverse, ax=ax[0, 0])
    visualize_distribution(err_rinverse, ax=ax[0, 1])
    visualize_distribution(dist_lmap, ax=ax[1, 0])
    visualize_distribution(dist_rmap, ax=ax[1, 1])

    ax[0, 0].set_xlabel(r"$r_\text{left}(X) = \|X - \phi^{-1}(\phi(X))\|$")
    ax[0, 0].set_ylabel(r"$p(r_\text{left} \mid X)$ where $x_i \sim \mathcal N(0,1)$")
    ax[0, 1].set_xlabel(r"$r_\text{right}(y) = \|y - \phi(\phi^{-1}(y))\|$")
    ax[0, 1].set_ylabel(r"$p(r_\text{right}\mid y)$ where $y_j \sim \mathcal N(0,1)$")

    ax[1, 0].set_xlabel(r"$d_\text{left}(X) = \|X - \phi(X)\|$")
    ax[1, 0].set_ylabel(r"$p(d_\text{left} \mid X)$ where $x_i \sim \mathcal N(0,1)$")
    ax[1, 1].set_xlabel(r"$d_\text{right}(y) = \|y - \phi^{-1}(y)\|$")
    ax[1, 1].set_ylabel(r"$p(d_\text{right} \mid y)$ where $y_j \sim \mathcal N(0,1)$")
    fig.suptitle(F"iResNetBlock -- Inversion property "
                 F"(samples:{n_samples}, dim-in:{input_size}, dim-hidden:{hidden_size})",
                 fontsize=16)
    fig.savefig("iResNetBlock_inversion.png")
    logger.info("iResNetBlock all done")


if __name__ == "__main__":
    test_LinearContraction()
    test_iResNetBlock()
