r"""Plot of all activation functions in a gallery."""

import math

import matplotlib.pyplot as plt
import pytest
import torch
from matplotlib.axes import Axes
from torch import Tensor

from linodenet.nn.activations import ACTIVATION_FUNCTIONS
from tests.utils.project import PROJECT

RESULT_DIR = PROJECT.RESULTS_DIR[__file__]


class TestActivationGallery:
    r"""Test suite for visualizing all activation functions in a gallery."""

    num = 1000
    xmax = 3.5
    xmin = -3.5
    ymin = -2.5
    ymax = 2.5
    x = torch.linspace(xmin, xmax, num)
    aspect_ratio = (xmax - xmin) / (ymax - ymin)
    w = 3  # width of each subplot in inches
    h = w / aspect_ratio
    xticks = list(range(math.ceil(xmin), math.floor(xmax) + 1))
    yticks = list(range(math.ceil(ymin), math.floor(ymax) + 1))

    def plot(self, ax: Axes, name: str) -> None:
        y = self.compute_activation(name)
        with plt.style.context("bmh"):
            ax.plot(self.x, y)
            ax.set_title(name)

    def fmt_axes(self, ax: Axes) -> None:
        ax.set_xlim(self.xmin, self.xmax)
        ax.set_ylim(self.ymin, self.ymax)
        # hide top and right spines
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        # move bottom and left spines to the center (data coordinate 0)
        ax.spines["bottom"].set_position(("data", 0))
        ax.spines["left"].set_position(("data", 0))
        # ax.set_aspect(self.aspect_ratio, adjustable="box")
        ax.set_xticks(self.xticks, [])
        ax.set_yticks(self.yticks, [])

    def compute_activation(self, name: str) -> Tensor:
        activation = ACTIVATION_FUNCTIONS[name]
        torch.manual_seed(0)
        y = activation(self.x)
        assert y.shape == self.x.shape
        return y

    @pytest.mark.parametrize("name", ACTIVATION_FUNCTIONS)
    def test_activation_gallery(
        self,
        name: str,
        # make_plots: bool,
    ) -> None:
        fig, ax = plt.subplots(figsize=(self.w, self.h), constrained_layout=True)
        self.plot(ax, name)
        self.fmt_axes(ax)
        fig.savefig(RESULT_DIR / f"{name}.png", dpi=200)
        plt.close(fig)

    def test_all_jointly(self) -> None:
        num_activations = len(ACTIVATION_FUNCTIONS)
        cols = 3
        rows = (num_activations + cols - 1) // cols
        fig, axes = plt.subplots(
            rows,
            cols,
            figsize=(self.w * cols, self.h * rows),
            constrained_layout=True,
            sharex=True,
            sharey=True,
        )
        for ax in axes.flatten():
            self.fmt_axes(ax)
        for ax, name in zip(axes.flatten(), ACTIVATION_FUNCTIONS, strict=False):
            try:
                self.plot(ax, name)
            except Exception:
                ax.set_title(f"{name} (error)")

        fig.savefig(RESULT_DIR / "all_activations.png", dpi=200)
        plt.close(fig)
