import numpy as np
import torch
from tqdm.auto import trange
from tsdm.plot import visualize_distribution
from tsdm.util import scaled_norm
from linodenet.models import LinODE
import matplotlib.pyplot as plt


def test_linode(dim=None, num=None, tol=1e-3, precision="single", relative_error=True, device='cpu'):
    from scipy.integrate import odeint

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

    X = odeint(func, x0, T, tfirst=True)

    model = LinODE(input_size=dim, kernel_initialization=A)
    model.to(dtype=torch_dtype, device=torch.device(device))
    ΔT = torch.diff(torch.tensor(T))
    Xhat = torch.empty(num, dim, dtype=torch_dtype)
    Xhat[0] = torch.tensor(x0)

    for i, Δt in enumerate(ΔT):
        Xhat[i + 1] = model(Δt, Xhat[i])

    Xhat = Xhat.detach().cpu().numpy()

    err = np.abs(X - Xhat)

    if relative_error:
        err /= np.abs(X) + eps

    return np.array([scaled_norm(err, p=p) for p in (1, 2, np.inf)])


def test_linode_error():
    err_single = np.array([test_linode() for _ in trange(1_000)]).T
    err_double = np.array([test_linode(precision="double") for _ in trange(1_000)]).T

    with plt.style.context('bmh'):
        fig, ax = plt.subplots(ncols=3, nrows=2, figsize=(10, 5), tight_layout=True, sharey='all', sharex='all')

    for i, err in enumerate((err_single, err_double)):
        for j, p in enumerate((1, 2, np.inf)):
            visualize_distribution(err[j], log=True, ax=ax[i, j])
            if j == 0:
                ax[i, 0].annotate(
                    F"FP{32 * (i + 1)}", xy=(0, 0.5), xytext=(-ax[i, 0].yaxis.labelpad - 5, 0),
                    xycoords=ax[i, 0].yaxis.label, textcoords='offset points', size='xx-large', ha='right', va='center')
            if i == 1:
                ax[i, j].set_xlabel(F"scaled, relative L{p} error")

    fig.savefig('linode_error_plot.svg')
