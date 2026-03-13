from collections.abc import Callable

import matplotlib.pyplot as plt
import pytest
import torch
from matplotlib.axes import Axes
from torch import Tensor

from linodenet_special.hard_contract import hard_contract, hard_expand
from tests.utils.project import PROJECT

RESULT_DIR = PROJECT.RESULTS_DIR[__file__]


@pytest.mark.parametrize("a", [0.125, 0.25, 0.5, 1.0], ids=lambda a: f"a={a}")
@pytest.mark.parametrize("c", [0.25, 1.0, 3.0], ids=lambda c: f"c={c}")
def test_hard_contract_reversible(a: float, c: float) -> None:
    x = make_test_grid(a, c)
    y = hard_contract(x, a=a, c=c)
    x_recovered = hard_expand(y, a=1 / a, c=c)

    torch.testing.assert_close(x_recovered, x, rtol=1e-14, atol=1e-14)


@pytest.mark.parametrize("a", [1.0, 2.0, 4.0, 8.0], ids=lambda a: f"a={a}")
@pytest.mark.parametrize("c", [0.25, 1.0, 3.0], ids=lambda c: f"c={c}")
def test_hard_expand_reversible(a: float, c: float) -> None:
    x = make_test_grid(a, c)
    y = hard_expand(x, a=a, c=c)
    x_recovered = hard_contract(y, a=1 / a, c=c)

    torch.testing.assert_close(x_recovered, x, rtol=1e-14, atol=1e-14)


def bend(x: Tensor, a: Tensor | float, c: Tensor | float) -> Tensor:
    if a == 1:
        return x
    return torch.where(
        x > c / (a - 1),
        x + c,
        torch.where(x < -c / (a - 1), x - c, a * x),
    )


def make_test_grid(a: float, c: float, /) -> Tensor:
    if a == 1:
        return torch.linspace(-8 * c, 8 * c, steps=257, dtype=torch.float64)

    threshold = c / abs(a - 1)
    points = torch.tensor(
        [
            -threshold - 2 * c,
            -threshold - c,
            -threshold,
            -0.5 * threshold,
            0.0,
            0.5 * threshold,
            threshold,
            threshold + c,
            threshold + 2 * c,
        ],
        dtype=torch.float64,
    )
    return torch.cat(
        (torch.linspace(-4 * threshold, 4 * threshold, steps=257), points)
    ).unique()


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


class TestVisual:
    X_MIN, X_MAX = -8.0, 8.0
    Y_MIN, Y_MAX = -8.0, 8.0

    def _format_centered_axes(self, ax: Axes, /) -> None:
        ax.set_xlim(self.X_MIN, self.X_MAX)
        ax.set_ylim(self.Y_MIN, self.Y_MAX)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_position(("data", 0))
        ax.spines["left"].set_position(("data", 0))
        ax.xaxis.set_ticks_position("bottom")
        ax.yaxis.set_ticks_position("left")

    def _plot_hard_bend_grid(
        self,
        func: Callable[..., Tensor],
        /,
        *,
        a_values: tuple[float, float, float],
        c_values: tuple[float, float, float],
        filename: str,
    ) -> None:
        x = torch.linspace(self.X_MIN, self.X_MAX, steps=1000)
        ys: dict[tuple[float, float], Tensor] = {}

        for a in a_values:
            for c in c_values:
                ys[a, c] = func(x, a=a, c=c)

        with plt.style.context("bmh"):
            fig, axes = plt.subplots(
                nrows=3,
                ncols=3,
                figsize=(10, 10),
                constrained_layout=True,
                squeeze=False,
                sharex=True,
                sharey=True,
            )
            for row, c in enumerate(c_values):
                for col, a in enumerate(a_values):
                    ax = axes[row][col]
                    ax.plot(x, ys[a, c], zorder=2, lw=3)
                    ax.plot(x, x + c, linestyle="--", color="black", lw=1)
                    ax.plot(x, x - c, linestyle="--", color="black", lw=1)
                    ax.set_title(f"a={a}, c={c}")
                    self._format_centered_axes(ax)

            fig.savefig(RESULT_DIR / filename, dpi=200)
            plt.close(fig)

    def test_plot_hard_contract(self) -> None:
        self._plot_hard_bend_grid(
            hard_contract,
            a_values=(1.0, 0.5, 0.25),
            c_values=(0.25, 1.0, 3.0),
            filename="hard_contract_grid.png",
        )

    def test_plot_hard_expand(self) -> None:
        self._plot_hard_bend_grid(
            hard_expand,
            a_values=(1.0, 2.0, 4.0),
            c_values=(0.25, 1.0, 3.0),
            filename="hard_expand_grid.png",
        )
