r"""Interactive plot of the optimal transport from a Gaussian to a mixture of Gaussians."""
# mypy: disable-error-code="no-untyped-def"

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
import pytest
import torch
from matplotlib.widgets import Slider
from scipy import stats
from torch import Tensor

from linodenet_special.fallbacks import (
    bimodal_to_gaussian,
    bimodal_to_gaussian_value_and_grad,
    gaussian_to_bimodal,
    gaussian_to_bimodal_value_and_grad,
    hard_bend,
    mixture_to_gaussian,
)
from tests.testing import PROJECT

RESULT_DIR = PROJECT.RESULTS_DIR[__file__]


@pytest.mark.interactive
def test_gaussian_to_bimodal_interactive() -> None:
    gaussian_to_bimodal_interactive()


@pytest.mark.interactive
def test_bimodal_to_gaussian_interactive() -> None:
    bimodal_to_gaussian_interactive()


@pytest.mark.interactive
def test_2_components_interactive() -> None:
    two_components_interactive()


@pytest.mark.interactive
def test_3_components_interactive() -> None:
    three_components_interactive()


PDF_MAX = 2.0
X_MIN = -8
X_MAX = +8


def bimodal_to_gaussian_approximation(
    x: Tensor,
    /,
    mu: Tensor,
    sigma: Tensor,
) -> Tensor:
    lam = math.exp(-0.5 * (mu / sigma) ** 2) / sigma
    return hard_bend(x, lam, mu / sigma, 1 / sigma)


def gaussian_to_bimodal_approximation(
    x: Tensor,
    /,
    mu: Tensor,
    sigma: Tensor,
) -> Tensor:
    lam = sigma * math.exp(0.5 * (mu / sigma) ** 2)
    return hard_bend(x, lam, mu, sigma)


def asymptotic_line(
    x,
    /,
    mu,
    sigma,
    weight_k,
    mu_k,
    sigma_k,
    *,
    use_correction: bool = False,
):
    z = (x - mu_k) / sigma_k
    y = mu + sigma * z

    if use_correction:
        # non-linear O(1/x) correction term.
        # TODO: possibly incorrect, need to double check
        y = y - torch.log(weight_k) * (sigma / z)
    return y


def asymptotic_inverse(
    y,
    /,
    mu,
    sigma,
    weight_k,
    mu_k,
    sigma_k,
    *,
    use_correction: bool = False,
):
    z = (y - mu) / sigma
    x = sigma_k * z + mu_k

    if use_correction:
        # non-linear O(1/x) correction term.
        # TODO: possibly incorrect, need to double check
        x = x + torch.log(weight_k) * (sigma_k / z)
    return x


def make_input_mixture(
    mu: Tensor,
    sigma: Tensor,
    omegas: Tensor,
    mus: Tensor,
    sigmas: Tensor,
    /,
) -> stats.Mixture:
    components = [
        stats.Normal(
            mu=float(mu + sigma * mus[k]),
            sigma=float(sigma * sigmas[k]),
        )
        for k in range(mus.numel())
    ]
    return stats.Mixture(components, weights=omegas.detach().cpu().tolist())


def make_bimodal_distribution(mu: Tensor, sigma: Tensor, /) -> stats.Mixture:
    components = [
        stats.Normal(mu=float(-mu), sigma=float(sigma)),
        stats.Normal(mu=float(mu), sigma=float(sigma)),
    ]
    return stats.Mixture(components, weights=[0.5, 0.5])


def test_plot_2_components() -> None:
    dtype = torch.float64

    x = torch.linspace(X_MIN, X_MAX, 100, dtype=dtype)

    mu = torch.tensor(0, dtype=dtype)
    sigma = torch.tensor(1, dtype=dtype)
    omegas = torch.tensor([2, 1], dtype=dtype)
    omegas = omegas / omegas.sum()
    mus = torch.tensor([-3, 3], dtype=dtype)
    sigmas = torch.tensor([0.5, 0.5], dtype=dtype)

    y = mixture_to_gaussian((x - mu) / sigma, omegas, mus, sigmas)

    with plt.style.context("bmh"):
        fig, ax = plt.subplots()
        ax.plot(x, y, lw=5)
        for k in range(mus.numel()):
            ax.plot(
                x, asymptotic_line(x, mu, sigma, omegas[k], mus[k], sigmas[k]), "k--"
            )

        ax.set_xlim(X_MIN, X_MAX)
        ax.set_title("Optimal Transport from N(0, 1) to mixture of 2 Gaussians")
        result_dir = RESULT_DIR / "transport_plots"
        result_dir.mkdir(exist_ok=True)
        file = result_dir / "transport_2.png"
        fig.savefig(file, dpi=300)
        print(f"Saved transport plot to {file!s}")


def test_plot_3_components() -> None:
    dtype = torch.float64

    x = torch.linspace(X_MIN, X_MAX, 100, dtype=dtype)

    mu = torch.tensor(0, dtype=dtype)
    sigma = torch.tensor(1, dtype=dtype)
    omegas = torch.tensor([2, 1, 2], dtype=dtype)
    omegas = omegas / omegas.sum()
    mus = torch.tensor([-3, -6, 3], dtype=dtype)
    sigmas = torch.tensor([0.5, 0.5, 0.5], dtype=dtype)

    y = mixture_to_gaussian((x - mu) / sigma, omegas, mus, sigmas)

    with plt.style.context("bmh"):
        fig, ax = plt.subplots()
        ax.plot(x, y, lw=5)
        for k in range(mus.numel()):
            ax.plot(
                x, asymptotic_line(x, mu, sigma, omegas[k], mus[k], sigmas[k]), "k--"
            )
        ax.set_xlim(X_MIN, X_MAX)
        ax.set_title("Optimal Transport from N(0, 1) to mixture of 3 Gaussians")
        result_dir = RESULT_DIR / "transport_plots"
        result_dir.mkdir(exist_ok=True)
        file = result_dir / "transport_2.png"
        fig.savefig(file, dpi=300)
        print(f"Saved transport plot to {file!s}")


def _slider_positions(
    count: int,
    /,
    *,
    bottom: float = 0.02,
    height: float = 0.025,
    gap: float = 0.005,
) -> list[tuple[float, float, float, float]]:
    return [
        (0.12, bottom + i * (height + gap), 0.76, height)
        for i in range(count - 1, -1, -1)
    ]


@dataclass
class ComplementarySliderCallback:
    target: Slider

    def __call__(self, value: float) -> None:
        target_value = min(1.0, max(0.0, 1.0 - value))
        if math.isclose(self.target.val, target_value):
            return

        target_eventson = self.target.eventson
        self.target.eventson = False
        try:
            self.target.set_val(target_value)
        finally:
            self.target.eventson = target_eventson


def _couple_complementary_sliders(slider_1: Slider, slider_2: Slider, /) -> None:
    slider_1.on_changed(ComplementarySliderCallback(slider_2))
    slider_2.on_changed(ComplementarySliderCallback(slider_1))


@dataclass
class CoupledThreeSliderState:
    sliders: Mapping[str, Slider]
    values: dict[str, float]
    active: bool = False


@dataclass
class CoupledThreeSliderCallback:
    source: str
    targets: tuple[str, str]
    state: CoupledThreeSliderState

    def __call__(self, value: float) -> None:
        if self.state.active:
            return

        delta = value - self.state.values[self.source]
        if math.isclose(delta, 0.0):
            return

        target_1_key, target_2_key = self.targets
        target_1_old = self.state.values[target_1_key]
        target_2_old = self.state.values[target_2_key]
        total = target_1_old + target_2_old

        if math.isclose(total, 0.0):
            target_1_value = -delta / 2
            target_2_value = -delta / 2
        else:
            target_1_value = target_1_old - delta * (target_1_old / total)
            target_2_value = target_2_old - delta * (target_2_old / total)

        source_value = min(1.0, max(0.0, value))
        target_1_value = min(1.0, max(0.0, target_1_value))
        target_2_value = min(1.0, max(0.0, target_2_value))

        self.state.active = True
        try:
            for key, target_value in (
                (self.source, source_value),
                (target_1_key, target_1_value),
                (target_2_key, target_2_value),
            ):
                slider = self.state.sliders[key]
                self.state.values[key] = target_value
                if math.isclose(slider.val, target_value):
                    continue

                slider_eventson = slider.eventson
                slider.eventson = False
                try:
                    slider.set_val(target_value)
                finally:
                    slider.eventson = slider_eventson
        finally:
            self.state.active = False


def _couple_three_weight_sliders(
    slider_1: Slider, slider_2: Slider, slider_3: Slider, /
) -> None:
    sliders = {"omega_1": slider_1, "omega_2": slider_2, "omega_3": slider_3}
    state = CoupledThreeSliderState(
        sliders=sliders,
        values={key: slider.val for key, slider in sliders.items()},
    )
    slider_1.on_changed(
        CoupledThreeSliderCallback("omega_1", ("omega_2", "omega_3"), state)
    )
    slider_2.on_changed(
        CoupledThreeSliderCallback("omega_2", ("omega_1", "omega_3"), state)
    )
    slider_3.on_changed(
        CoupledThreeSliderCallback("omega_3", ("omega_1", "omega_2"), state)
    )


@dataclass
class TransportPlotState2:
    x: Tensor
    line: Any
    component_lines: list[Any]
    input_pdf_line: Any
    target_pdf_line: Any
    twin_ax: Any
    input_hist_container: Any
    output_hist_container: Any
    sliders: Mapping[str, Slider]

    def update(self, _value: float) -> None:
        dtype = self.x.dtype

        mu = torch.tensor(self.sliders["mu"].val, dtype=dtype)
        sigma = torch.tensor(self.sliders["sigma"].val, dtype=dtype)

        omegas = torch.tensor(
            [self.sliders["omega_1"].val, self.sliders["omega_2"].val],
            dtype=dtype,
        )
        omegas = omegas / omegas.sum()

        mus = torch.tensor(
            [self.sliders["mu_1"].val, self.sliders["mu_2"].val], dtype=dtype
        )

        sigmas = torch.tensor(
            [self.sliders["sigma_1"].val, self.sliders["sigma_2"].val], dtype=dtype
        )

        y = mixture_to_gaussian((self.x - mu) / sigma, omegas, mus, sigmas)
        mixture = make_input_mixture(mu, sigma, omegas, mus, sigmas)
        target = stats.Normal(mu=0.0, sigma=1.0)
        x_samples = torch.tensor(mixture.sample(shape=1_000, rng=0), dtype=dtype)
        y_samples = mixture_to_gaussian((x_samples - mu) / sigma, omegas, mus, sigmas)

        self.line.set_ydata(y)
        self.input_pdf_line.set_xdata(self.x)
        self.input_pdf_line.set_ydata(mixture.pdf(self.x))
        self.target_pdf_line.set_xdata(self.x)
        self.target_pdf_line.set_ydata(target.pdf(self.x))
        for k, line in enumerate(self.component_lines):
            line.set_ydata(
                asymptotic_line(
                    (self.x - mu) / sigma,
                    mus[k],
                    sigmas[k],
                    omegas[k],
                    mus[k],
                    sigmas[k],
                )
            )

        self.input_hist_container.remove()
        self.input_hist_container = self.twin_ax.hist(
            x_samples,
            bins=50,
            density=True,
            alpha=0.2,
            color="green",
        )[2]
        self.output_hist_container.remove()
        self.output_hist_container = self.twin_ax.hist(
            y_samples,
            bins=50,
            density=True,
            alpha=0.2,
            color="red",
        )[2]
        self.twin_ax.set_xlabel("density")
        self.twin_ax.grid(False)
        self.twin_ax.patch.set_alpha(0)
        self.twin_ax.set_ylim(0, PDF_MAX)
        self.line.figure.canvas.draw_idle()


@dataclass
class TransportPlotStateBimodal:
    x: Tensor
    line: Any
    jac_line: Any
    component_lines: list[Any]
    input_pdf_line: Any
    target_pdf_line: Any
    twin_ax: Any
    jac_ax: Any
    input_hist_container: Any
    output_hist_container: Any
    sliders: Mapping[str, Slider]

    def update(self, _value: float) -> None:
        dtype = self.x.dtype

        mu = torch.tensor(self.sliders["mu"].val, dtype=dtype)
        sigma = torch.tensor(self.sliders["sigma"].val, dtype=dtype)

        y, jac = bimodal_to_gaussian_value_and_grad(self.x, mu, sigma)
        twin = make_bimodal_distribution(mu, sigma)
        target = stats.Normal(mu=0.0, sigma=1.0)
        x_samples = torch.tensor(twin.sample(shape=1_000, rng=0), dtype=dtype)
        y_samples = bimodal_to_gaussian(x_samples, mu, sigma)
        approximation = bimodal_to_gaussian_approximation(self.x, mu, sigma)

        self.line.set_ydata(y)
        self.jac_line.set_ydata(jac)
        self.component_lines[0].set_ydata(approximation)
        self.input_pdf_line.set_xdata(self.x)
        self.input_pdf_line.set_ydata(twin.pdf(self.x))
        self.target_pdf_line.set_xdata(self.x)
        self.target_pdf_line.set_ydata(target.pdf(self.x))

        self.input_hist_container.remove()
        self.input_hist_container = self.twin_ax.hist(
            x_samples,
            bins=50,
            density=True,
            alpha=0.2,
            color="green",
        )[2]
        self.output_hist_container.remove()
        self.output_hist_container = self.twin_ax.hist(
            y_samples,
            bins=50,
            density=True,
            alpha=0.2,
            color="red",
        )[2]
        self.twin_ax.set_xlabel("density")
        self.twin_ax.grid(False)
        self.twin_ax.patch.set_alpha(0)
        self.twin_ax.set_ylim(0, PDF_MAX)
        self.jac_ax.set_ylim(0, 10)
        self.line.figure.canvas.draw_idle()


@dataclass
class TransportPlotStateGaussian:
    y: Tensor
    line: Any
    jac_line: Any
    component_lines: list[Any]
    input_pdf_line: Any
    target_pdf_line: Any
    twin_ax: Any
    jac_ax: Any
    input_hist_container: Any
    output_hist_container: Any
    sliders: Mapping[str, Slider]

    def update(self, _value: float) -> None:
        dtype = self.y.dtype

        mu = torch.tensor(self.sliders["mu"].val, dtype=dtype)
        sigma = torch.tensor(self.sliders["sigma"].val, dtype=dtype)

        x, jac = gaussian_to_bimodal_value_and_grad(self.y, mu, sigma)
        source = stats.Normal(mu=0.0, sigma=1.0)
        twin = make_bimodal_distribution(mu, sigma)
        y_samples = torch.tensor(source.sample(shape=1_000, rng=0), dtype=dtype)
        x_samples = gaussian_to_bimodal(y_samples, mu, sigma)
        approximation = gaussian_to_bimodal_approximation(self.y, mu, sigma)

        self.line.set_ydata(x)
        self.jac_line.set_ydata(jac)
        self.component_lines[0].set_ydata(approximation)
        self.input_pdf_line.set_xdata(self.y)
        self.input_pdf_line.set_ydata(source.pdf(self.y))
        self.target_pdf_line.set_xdata(self.y)
        self.target_pdf_line.set_ydata(twin.pdf(self.y))

        self.input_hist_container.remove()
        self.input_hist_container = self.twin_ax.hist(
            y_samples,
            bins=50,
            density=True,
            alpha=0.2,
            color="green",
        )[2]
        self.output_hist_container.remove()
        self.output_hist_container = self.twin_ax.hist(
            x_samples,
            bins=50,
            density=True,
            alpha=0.2,
            color="red",
        )[2]
        self.twin_ax.set_xlabel("density")
        self.twin_ax.grid(False)
        self.twin_ax.patch.set_alpha(0)
        self.twin_ax.set_ylim(0, PDF_MAX)
        self.jac_ax.set_ylim(0, 10)
        self.line.figure.canvas.draw_idle()


@dataclass
class TransportPlotState3Components:
    x: Tensor
    line: Any
    component_lines: list[Any]
    input_pdf_line: Any
    target_pdf_line: Any
    twin_ax: Any
    input_hist_container: Any
    output_hist_container: Any
    sliders: Mapping[str, Slider]

    def update(self, _value: float) -> None:
        dtype = self.x.dtype

        mu = torch.tensor(self.sliders["mu"].val, dtype=dtype)
        sigma = torch.tensor(self.sliders["sigma"].val, dtype=dtype)

        omegas = torch.tensor(
            [
                self.sliders["omega_1"].val,
                self.sliders["omega_2"].val,
                self.sliders["omega_3"].val,
            ],
            dtype=dtype,
        )
        omegas = omegas / omegas.sum()

        mus = torch.tensor(
            [
                self.sliders["mu_1"].val,
                self.sliders["mu_2"].val,
                self.sliders["mu_3"].val,
            ],
            dtype=dtype,
        )

        sigmas = torch.tensor(
            [
                self.sliders["sigma_1"].val,
                self.sliders["sigma_2"].val,
                self.sliders["sigma_3"].val,
            ],
            dtype=dtype,
        )

        y = mixture_to_gaussian((self.x - mu) / sigma, omegas, mus, sigmas)
        source = make_input_mixture(mu, sigma, omegas, mus, sigmas)
        target = stats.Normal(mu=0.0, sigma=1.0)
        x_samples = torch.tensor(source.sample(shape=1_000, rng=0), dtype=dtype)
        y_samples = mixture_to_gaussian((x_samples - mu) / sigma, omegas, mus, sigmas)

        self.line.set_ydata(y)
        self.input_pdf_line.set_xdata(self.x)
        self.input_pdf_line.set_ydata(source.pdf(self.x))
        self.target_pdf_line.set_xdata(self.x)
        self.target_pdf_line.set_ydata(target.pdf(self.x))
        for k, line in enumerate(self.component_lines):
            line.set_ydata(
                asymptotic_line(
                    (self.x - mu) / sigma,
                    mus[k],
                    sigmas[k],
                    omegas[k],
                    mus[k],
                    sigmas[k],
                )
            )

        self.input_hist_container.remove()
        self.input_hist_container = self.twin_ax.hist(
            x_samples,
            bins=50,
            density=True,
            alpha=0.2,
            color="green",
        )[2]
        self.output_hist_container.remove()
        self.output_hist_container = self.twin_ax.hist(
            y_samples,
            bins=50,
            density=True,
            alpha=0.2,
            color="red",
        )[2]
        self.twin_ax.set_xlabel("density")
        self.twin_ax.grid(False)
        self.twin_ax.patch.set_alpha(0)
        self.twin_ax.set_ylim(0, PDF_MAX)
        self.line.figure.canvas.draw_idle()


def gaussian_to_bimodal_interactive() -> None:
    dtype = torch.float64

    y = torch.linspace(X_MIN, X_MAX, 200, dtype=dtype)

    mu = torch.tensor(3.0, dtype=dtype)
    sigma = torch.tensor(0.5, dtype=dtype)

    x, jac = gaussian_to_bimodal_value_and_grad(y, mu, sigma)
    source = stats.Normal(mu=0.0, sigma=1.0)
    target = make_bimodal_distribution(mu, sigma)
    y_samples = torch.tensor(source.sample(shape=1_000, rng=0), dtype=dtype)
    x_samples = gaussian_to_bimodal(y_samples, mu, sigma)

    with plt.style.context("bmh"):
        fig, ax = plt.subplots(figsize=(10, 5))
        ax_twin = ax.twinx()
        ax_jac = ax.twinx()
        ax_jac.spines.right.set_position(("axes", 1.12))
        ax.patch.set_alpha(0)
        ax_twin.patch.set_alpha(0)
        ax_jac.patch.set_alpha(0)
        ax_twin.grid(False)
        ax_jac.grid(False)
        ax.set_zorder(2)
        ax_twin.set_zorder(1)
        ax_jac.set_zorder(0)

        (line,) = ax.plot(y, x, label="transport", lw=5)
        (jac_line,) = ax_jac.plot(
            y,
            jac,
            color="tab:purple",
            alpha=0.8,
            lw=2,
            label="derivative",
        )
        component_lines = [
            ax.plot(y, gaussian_to_bimodal_approximation(y, mu, sigma), "k--")[0],
        ]
        (input_pdf_line,) = ax_twin.plot(
            y,
            source.pdf(y),
            color="green",
            alpha=0.3,
            lw=2,
            zorder=0,
        )
        (target_pdf_line,) = ax_twin.plot(
            y,
            target.pdf(y),
            color="red",
            alpha=0.5,
            lw=2,
            zorder=0,
        )
        _, _, input_hist_container = ax_twin.hist(
            y_samples,
            bins=50,
            density=True,
            alpha=0.2,
            color="green",
        )
        _, _, output_hist_container = ax_twin.hist(
            x_samples,
            bins=50,
            density=True,
            alpha=0.2,
            color="red",
        )
        ax.set_xlim(X_MIN, X_MAX)
        ax_twin.set_xlim(X_MIN, X_MAX)
        ax_jac.set_xlim(X_MIN, X_MAX)
        ax_twin.set_ylabel("density")
        ax_twin.set_ylim(0, PDF_MAX)
        ax_jac.set_ylabel("derivative")
        ax_jac.set_ylim(0, 10)
        ax.set_title("Interactive Transport via gaussian_to_bimodal")
        ax.legend([line, jac_line], ["transport", "derivative"], loc="upper left")

        slider_specs = [
            ("mu", "μ", -5.0, 5.0, float(mu)),
            ("sigma", "σ", 0.1, 3.0, float(sigma)),
        ]

        fig.subplots_adjust(bottom=0.14)
        axes = [fig.add_axes(pos) for pos in _slider_positions(len(slider_specs))]
        sliders = {
            key: Slider(ax_, label, min_, max_, valinit=init)
            for ax_, (key, label, min_, max_, init) in zip(
                axes, slider_specs, strict=True
            )
        }

        state = TransportPlotStateGaussian(
            y=y,
            line=line,
            jac_line=jac_line,
            component_lines=component_lines,
            input_pdf_line=input_pdf_line,
            target_pdf_line=target_pdf_line,
            twin_ax=ax_twin,
            jac_ax=ax_jac,
            input_hist_container=input_hist_container,
            output_hist_container=output_hist_container,
            sliders=sliders,
        )
        for slider in sliders.values():
            slider.on_changed(state.update)

        plt.show()


def bimodal_to_gaussian_interactive() -> None:
    dtype = torch.float64

    x = torch.linspace(X_MIN, X_MAX, 200, dtype=dtype)

    mu = torch.tensor(3.0, dtype=dtype)
    sigma = torch.tensor(0.5, dtype=dtype)

    y, jac = bimodal_to_gaussian_value_and_grad(x, mu, sigma)
    source = make_bimodal_distribution(mu, sigma)
    target = stats.Normal(mu=0.0, sigma=1.0)
    x_samples = torch.tensor(source.sample(shape=1_000, rng=0), dtype=dtype)
    y_samples = bimodal_to_gaussian(x_samples, mu, sigma)

    with plt.style.context("bmh"):
        fig, ax = plt.subplots(figsize=(10, 5))
        ax_twin = ax.twinx()
        ax_jac = ax.twinx()
        ax_jac.spines.right.set_position(("axes", 1.12))
        ax.patch.set_alpha(0)
        ax_twin.patch.set_alpha(0)
        ax_jac.patch.set_alpha(0)
        ax_twin.grid(False)
        ax_jac.grid(False)
        ax.set_zorder(2)
        ax_twin.set_zorder(1)
        ax_jac.set_zorder(0)

        (line,) = ax.plot(x, y, label="transport", lw=5)
        (jac_line,) = ax_jac.plot(
            x,
            jac,
            color="tab:purple",
            alpha=0.8,
            lw=2,
            label="derivative",
        )
        asymptote = bimodal_to_gaussian_approximation(x, mu, sigma)
        component_lines = [ax.plot(x, asymptote, "k--")[0]]
        (input_pdf_line,) = ax_twin.plot(
            x,
            source.pdf(x),
            color="green",
            alpha=0.3,
            lw=2,
            zorder=0,
        )
        (target_pdf_line,) = ax_twin.plot(
            x,
            target.pdf(x),
            color="red",
            alpha=0.5,
            lw=2,
            zorder=0,
        )
        _, _, input_hist_container = ax_twin.hist(
            x_samples,
            bins=50,
            density=True,
            alpha=0.2,
            color="green",
        )
        _, _, output_hist_container = ax_twin.hist(
            y_samples,
            bins=50,
            density=True,
            alpha=0.2,
            color="red",
        )
        ax.set_xlim(X_MIN, X_MAX)
        ax_twin.set_xlim(X_MIN, X_MAX)
        ax_jac.set_xlim(X_MIN, X_MAX)
        ax_twin.set_ylabel("density")
        ax_twin.set_ylim(0, PDF_MAX)
        ax_jac.set_ylabel("derivative")
        ax_jac.set_ylim(0, 10)
        ax.set_title("Interactive Transport via bimodal_to_gaussian")
        ax.legend([line, jac_line], ["transport", "derivative"], loc="upper left")

        slider_specs = [
            ("mu", "μ", -5.0, 5.0, float(mu)),
            ("sigma", "σ", 0.1, 3.0, float(sigma)),
        ]

        fig.subplots_adjust(bottom=0.14)
        axes = [fig.add_axes(pos) for pos in _slider_positions(len(slider_specs))]
        sliders = {
            key: Slider(ax_, label, min_, max_, valinit=init)
            for ax_, (key, label, min_, max_, init) in zip(
                axes, slider_specs, strict=True
            )
        }

        state = TransportPlotStateBimodal(
            x=x,
            line=line,
            jac_line=jac_line,
            component_lines=component_lines,
            input_pdf_line=input_pdf_line,
            target_pdf_line=target_pdf_line,
            twin_ax=ax_twin,
            jac_ax=ax_jac,
            input_hist_container=input_hist_container,
            output_hist_container=output_hist_container,
            sliders=sliders,
        )
        for slider in sliders.values():
            slider.on_changed(state.update)

        plt.show()


def two_components_interactive() -> None:
    dtype = torch.float64

    x = torch.linspace(X_MIN, X_MAX, 200, dtype=dtype)

    mu = torch.tensor(0, dtype=dtype)
    sigma = torch.tensor(1, dtype=dtype)
    omegas = torch.tensor([2, 1], dtype=dtype)
    omegas = omegas / omegas.sum()
    mus = torch.tensor([-3, 3], dtype=dtype)
    sigmas = torch.tensor([0.5, 0.5], dtype=dtype)

    y = mixture_to_gaussian((x - mu) / sigma, omegas, mus, sigmas)
    source = make_input_mixture(mu, sigma, omegas, mus, sigmas)
    target = stats.Normal(mu=0.0, sigma=1.0)
    x_samples = torch.tensor(source.sample(shape=1_000, rng=0), dtype=dtype)
    y_samples = mixture_to_gaussian((x_samples - mu) / sigma, omegas, mus, sigmas)

    with plt.style.context("bmh"):
        fig, ax = plt.subplots(figsize=(10, 6))
        ax_twin = ax.twinx()
        ax.patch.set_alpha(0)
        ax_twin.patch.set_alpha(0)
        ax_twin.grid(False)
        ax.set_zorder(2)
        ax_twin.set_zorder(1)

        (line,) = ax.plot(x, y, label="transport", lw=5)
        component_lines = [
            ax.plot(
                x,
                asymptotic_line(
                    (x - mu) / sigma, mus[k], sigmas[k], omegas[k], mus[k], sigmas[k]
                ),
                "k--",
            )[0]
            for k in range(mus.numel())
        ]

        (input_pdf_line,) = ax_twin.plot(
            x,
            source.pdf(x),
            color="green",
            alpha=0.3,
            lw=2,
            zorder=0,
        )
        (target_pdf_line,) = ax_twin.plot(
            x,
            target.pdf(x),
            color="red",
            alpha=0.5,
            lw=2,
            zorder=0,
        )
        _, _, input_hist_container = ax_twin.hist(
            x_samples,
            bins=50,
            density=True,
            alpha=0.2,
            color="green",
        )
        _, _, output_hist_container = ax_twin.hist(
            y_samples,
            bins=50,
            density=True,
            alpha=0.2,
            color="red",
        )
        ax.set_xlim(X_MIN, X_MAX)
        ax_twin.set_xlim(X_MIN, X_MAX)
        ax_twin.set_ylabel("density")
        ax_twin.set_ylim(0, PDF_MAX)
        ax.set_title("Interactive Transport via gaussian_to_mixture")
        ax.legend(loc="upper left")

        slider_specs = [
            ("mu", "μ", -5.0, 5.0, float(mu)),
            ("sigma", "σ", 0.2, 5.0, float(sigma)),
            ("omega_1", "ω₁", 0.0, 1.0, float(omegas[0])),
            ("omega_2", "ω₂", 0.0, 1.0, float(omegas[1])),
            ("mu_1", "μ₁", -8.0, 8.0, float(mus[0])),
            ("mu_2", "μ₂", -8.0, 8.0, float(mus[1])),
            ("sigma_1", "σ₁", 0.1, 3.0, float(sigmas[0])),
            ("sigma_2", "σ₂", 0.1, 3.0, float(sigmas[1])),
        ]

        fig.subplots_adjust(bottom=0.32)
        axes = [fig.add_axes(pos) for pos in _slider_positions(len(slider_specs))]
        sliders = {
            key: Slider(ax_, label, min_, max_, valinit=init)
            for ax_, (key, label, min_, max_, init) in zip(
                axes, slider_specs, strict=True
            )
        }
        _couple_complementary_sliders(sliders["omega_1"], sliders["omega_2"])

        state = TransportPlotState2(
            x=x,
            line=line,
            component_lines=component_lines,
            input_pdf_line=input_pdf_line,
            target_pdf_line=target_pdf_line,
            twin_ax=ax_twin,
            input_hist_container=input_hist_container,
            output_hist_container=output_hist_container,
            sliders=sliders,
        )
        for slider in sliders.values():
            slider.on_changed(state.update)

        plt.show()


def three_components_interactive() -> None:
    dtype = torch.float64

    x = torch.linspace(X_MIN, X_MAX, 200, dtype=dtype)

    mu = torch.tensor(0, dtype=dtype)
    sigma = torch.tensor(1, dtype=dtype)
    omegas = torch.tensor([2, 1, 2], dtype=dtype)
    omegas = omegas / omegas.sum()
    mus = torch.tensor([-3, 0, 3], dtype=dtype)
    sigmas = torch.tensor([0.5, 0.5, 0.5], dtype=dtype)

    y = mixture_to_gaussian((x - mu) / sigma, omegas, mus, sigmas)
    source = make_input_mixture(mu, sigma, omegas, mus, sigmas)
    target = stats.Normal(mu=0.0, sigma=1.0)
    x_samples = torch.tensor(source.sample(shape=1_000, rng=0), dtype=dtype)
    y_samples = mixture_to_gaussian((x_samples - mu) / sigma, omegas, mus, sigmas)

    with plt.style.context("bmh"):
        fig, ax = plt.subplots(figsize=(10, 7))
        ax_twin = ax.twinx()
        ax.patch.set_alpha(0)
        ax_twin.patch.set_alpha(0)
        ax_twin.grid(False)
        ax.set_zorder(2)
        ax_twin.set_zorder(1)

        (line,) = ax.plot(x, y, label="transport", lw=5)
        component_lines = [
            ax.plot(
                x,
                asymptotic_line(
                    (x - mu) / sigma, mus[k], sigmas[k], omegas[k], mus[k], sigmas[k]
                ),
                "k--",
            )[0]
            for k in range(mus.numel())
        ]

        (input_pdf_line,) = ax_twin.plot(
            x,
            source.pdf(x),
            color="green",
            alpha=0.3,
            lw=2,
            zorder=0,
        )
        (target_pdf_line,) = ax_twin.plot(
            x,
            target.pdf(x),
            color="red",
            alpha=0.5,
            lw=2,
            zorder=0,
        )
        _, _, input_hist_container = ax_twin.hist(
            x_samples,
            bins=50,
            density=True,
            alpha=0.2,
            color="green",
        )
        _, _, output_hist_container = ax_twin.hist(
            y_samples,
            bins=50,
            density=True,
            alpha=0.2,
            color="red",
        )
        ax.set_xlim(X_MIN, X_MAX)
        ax_twin.set_xlim(X_MIN, X_MAX)
        ax_twin.set_ylabel("density")
        ax_twin.set_ylim(0, PDF_MAX)
        ax.set_title("Interactive Transport to mixture of 3 Gaussians")
        ax.legend(loc="upper left")

        slider_specs = [
            ("mu", "μ", -5.0, 5.0, float(mu)),
            ("sigma", "σ", 0.2, 5.0, float(sigma)),
            ("omega_1", "ω₁", 0.0, 1.0, float(omegas[0])),
            ("omega_2", "ω₂", 0.0, 1.0, float(omegas[1])),
            ("omega_3", "ω₃", 0.0, 1.0, float(omegas[2])),
            ("mu_1", "μ₁", -8.0, 8.0, float(mus[0])),
            ("mu_2", "μ₂", -8.0, 8.0, float(mus[1])),
            ("mu_3", "μ₃", -8.0, 8.0, float(mus[2])),
            ("sigma_1", "σ₁", 0.1, 3.0, float(sigmas[0])),
            ("sigma_2", "σ₂", 0.1, 3.0, float(sigmas[1])),
            ("sigma_3", "σ₃", 0.1, 3.0, float(sigmas[2])),
        ]

        fig.subplots_adjust(bottom=0.4)
        axes = [fig.add_axes(pos) for pos in _slider_positions(len(slider_specs))]
        sliders = {
            key: Slider(ax_, label, min_, max_, valinit=init)
            for ax_, (key, label, min_, max_, init) in zip(
                axes, slider_specs, strict=True
            )
        }
        _couple_three_weight_sliders(
            sliders["omega_1"], sliders["omega_2"], sliders["omega_3"]
        )

        state = TransportPlotState3Components(
            x=x,
            line=line,
            component_lines=component_lines,
            input_pdf_line=input_pdf_line,
            target_pdf_line=target_pdf_line,
            twin_ax=ax_twin,
            input_hist_container=input_hist_container,
            output_hist_container=output_hist_container,
            sliders=sliders,
        )
        for slider in sliders.values():
            slider.on_changed(state.update)

        plt.show()
