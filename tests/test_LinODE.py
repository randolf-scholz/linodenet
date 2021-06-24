"""
Test error of linear ODE against odeint
"""

import logging

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.integrate import odeint
from tqdm.auto import trange

from tsdm.plot import visualize_distribution
from tsdm.util import scaled_norm

from linodenet.models import LinODE

logger = logging.getLogger(__name__)
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('PIL').setLevel(logging.WARNING)


def linode_error(dim=None, num=None, precision="single", relative_error=True,
                 device: torch.device = torch.device('cpu')):
    """Compare LinODE against scipy.odeint on linear system"""

    if precision == "single":
        eps = 2 ** -24
        numpy_dtype = np.float32
        torch_dtype = torch.float32
    elif precision == "double":
        eps = 2 ** -53
        numpy_dtype = np.float64
        torch_dtype = torch.float64
    else:
        raise ValueError

    num = np.random.randint(low=20, high=1000) or num
    dim = np.random.randint(low=2, high=100) or dim
    t0, t1 = np.random.uniform(low=-10, high=10, size=(2,)).astype(numpy_dtype)
    A = np.random.randn(dim, dim).astype(numpy_dtype)
    x0 = np.random.randn(dim).astype(numpy_dtype)
    T = np.random.uniform(low=t0, high=t1, size=num - 2).astype(numpy_dtype)
    T = np.sort([t0, *T, t1]).astype(numpy_dtype)

    def func(t, x):
        return A @ x

    X = np.array(odeint(func, x0, T, tfirst=True))
    T = torch.tensor(T, dtype=torch_dtype, device=device)
    x0 = torch.tensor(x0, dtype=torch_dtype, device=device)
    model = LinODE(input_size=dim, kernel_initialization=A)
    model.to(dtype=torch_dtype, device=device)

    Xhat = model(torch.tensor(T), torch.tensor(x0))
    Xhat = Xhat.clone().detach().cpu().numpy()

    err = np.abs(X - Xhat)

    if relative_error:
        err /= np.abs(X) + eps

    return np.array([scaled_norm(err, p=p) for p in (1, 2, np.inf)])


def test_linode_error():
    """Compare LinODE against scipy.odeint on linear system"""
    NSAMPLES = 100
    logger.info("Testing LinODE")
    logger.info("Generating %i samples in single precision", NSAMPLES)
    err_single = np.array([linode_error(precision="single") for _ in trange(NSAMPLES)]).T
    logger.info("Generating %i samples in double precision", NSAMPLES)
    err_double = np.array([linode_error(precision="double") for _ in trange(NSAMPLES)]).T

    with plt.style.context('bmh'):
        fig, ax = plt.subplots(ncols=3, nrows=2, figsize=(10, 5), tight_layout=True,
                               sharey='all', sharex='all')

    logger.info("LinODE generating figure")
    for i, err in enumerate((err_single, err_double)):
        for j, p in enumerate((1, 2, np.inf)):
            visualize_distribution(err[j], log=True, ax=ax[i, j])
            if j == 0:
                ax[i, 0].annotate(
                    F"FP{32 * (i + 1)}", xy=(0, 0.5), xytext=(-ax[i, 0].yaxis.labelpad - 5, 0),
                    xycoords=ax[i, 0].yaxis.label, textcoords='offset points', size='xx-large',
                    ha='right', va='center')
            if i == 1:
                ax[i, j].set_xlabel(F"scaled, relative L{p} distance")

    fig.suptitle(r"Discrepancy between $x^\text{(LinODE)}$ and $x^\text{(odeint)}$, "
                 F"{NSAMPLES} samples")
    fig.savefig('LinODE_odeint_comparison.png')
    logger.info("LinearContraction all done")


if __name__ == "__main__":
    test_linode_error()
