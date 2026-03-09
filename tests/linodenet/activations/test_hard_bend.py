import matplotlib.pyplot as plt
import torch
from torch import Tensor

from tests.utils.project import PROJECT

RESULT_DIR = PROJECT.RESULTS_DIR[__file__]


def bend(x: Tensor, a: Tensor | float, c: Tensor | float) -> Tensor:
    if a == 1:
        return x
    return torch.where(
        x > c / (a - 1),
        x + c,
        torch.where(x < -c / (a - 1), x - c, a * x),
    )


def test_bend_histogram_grid() -> None:
    torch.manual_seed(0)
    x = torch.randn(20000)
    a_values = (1.0, 2.0, 4.0, 8.0)
    c_values = (0.5, 1.0, 2.0, 4.0)
    xmin = -5
    xmax = 5

    with plt.style.context("bmh"):
        fig, axes = plt.subplots(
            nrows=4,
            ncols=4,
            figsize=(12, 12),
            constrained_layout=True,
            squeeze=False,
            sharex=True,
            sharey=True,
        )
        for col, a in enumerate(a_values):
            for row, c in enumerate(c_values):
                ax = axes[row][col]
                ax.set_xlim(xmin, xmax)
                y = bend(x, a, c)
                ax.hist(y.numpy(), bins=50, density=True)
                ax.set_title(f"bend(x, a={a}, c={c})")
        fig.savefig(RESULT_DIR / "bend_hist_grid.png", dpi=200)
        plt.close(fig)
