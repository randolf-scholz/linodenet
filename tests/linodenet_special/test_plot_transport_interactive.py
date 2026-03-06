r"""Interactive plot of the optimal transport from a Gaussian to a mixture of Gaussians."""
# mypy: disable-error-code="no-untyped-def"

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Final

import matplotlib.pyplot as plt
import torch
from matplotlib.widgets import Slider
from torch import Tensor

from linodenet_special import ndtri_exp_naive as ndtri_exp
from tests.utils.project import PROJECT

SQRT_2: Final[float] = math.sqrt(2)
type Context = Any  # torch offers no type hint


def psi_old(x, mu, sigma, omegas, mus, sigmas):
    # x.shape = (batch_size,)
    # mu.shape = (,)
    # sigma.shape = (,)
    # omegas.shape = (num_components,)
    # mus.shape = (num_components,)
    # sigmas.shape = (num_components,)

    EPS = 8 * torch.finfo(x.dtype).eps

    # compute (x - μₖ)/(σₖ√2) for all k.
    # x has shape (batch_size, d)
    # mus and sigmas have shape (num_components,)
    # we want z to have shape (batch_size, num_components,)
    z = (x.unsqueeze(-1) - mus) / (sigmas * SQRT_2)

    # compute ∑ₖωₖ\erf(zₖ)
    mix = torch.einsum("k, ...k -> ...", omegas, torch.erf(z))
    mix = torch.clamp(mix, -1 + EPS, 1 - EPS)
    mask = mix.abs() < (1 - EPS)

    # compute y = μ + √2σ * erfinv(mix)
    y_exact = mu + sigma * SQRT_2 * torch.erfinv(mix)
    mask = mask & y_exact.isfinite()

    # in the far field, the result is asymptotically linear
    # with y ≍ (x-μ⁎)⋅(σ/σ⁎) + μ, where μ⁎ and σ⁎ are the mean and std
    # of the component with the largest variance.
    # This is because the tails of the mixture are dominated by the component
    # with the largest variance, and the optimal transport to a Gaussian is
    # linear in the tails.
    # however, this only happens before the line for the N(μₘᵢₙ, σₘᵢₙ)
    # and N(μ⁎, σ⁎) intersect, which happens at xₘᵢₙ = (σₘᵢₙμ⁎ - σ⁎μₘᵢₙ) / (σₘᵢₙ - σ⁎).
    # and likewise after the line for the N(μₘₐₓ, σₘₐₓ) and N(μ⁎, σ⁎) intersect,
    # which happens at xₘₐₓ = (σₘₐₓμ⁎ - σ⁎μₘₐₓ) / (σₘₐₓ - σ⁎).
    # So, on [xₘₐₓ, ∞), far field holds, but on
    # [x₀, xₘₐₓ], when the exact is numericallwe use N(μₘₐₓ, σₘₐₓ)
    k_left = torch.argmin(mus)
    k_right = torch.argmax(mus)
    mu_left = mus[k_left]
    mu_right = mus[k_right]
    sigma_left = sigmas[k_left]
    sigma_right = sigmas[k_right]

    # far field
    k_star = torch.argmax(sigmas)
    mu_star = mus[k_star]
    sigma_star = sigmas[k_star]
    y_tail = (x - mu_star) * (sigma / sigma_star) + mu

    # restrict the exact formula to the region where it is numerically stable
    # and better than the PL approximation.
    x_left = mu_left - 5 * sigma_left
    x_right = mu_right + 5 * sigma_right
    y_left = (x - mu_left) * (sigma / sigma_left) + mu
    y_right = (x - mu_right) * (sigma / sigma_right) + mu
    x_middle = (x_left + x_right) / 2
    # mask = mask & (x >= x_left) & (x <= x_right)

    if k_star == k_left:
        x_min = x_left
    else:
        x_min = (sigma_left * mu_star - sigma_star * mu_left) / (
            sigma_left - sigma_star
        )
        x_min = (
            x_min if x_min.isfinite() else torch.tensor(float("-inf"), dtype=x.dtype)
        )

    if k_star == k_right:
        x_max = mu_right
    else:
        x_max = (sigma_right * mu_star - sigma_star * mu_right) / (
            sigma_right - sigma_star
        )
        x_max = x_max if x_max.isfinite() else torch.tensor(float("inf"), dtype=x.dtype)

    mask = mask & (x >= x_min) & (x <= x_max)

    return torch.where(
        mask,
        y_exact,
        torch.where(
            (x <= x_min) | (x >= x_max),
            y_tail,
            torch.where(x < x_middle, y_left, y_right),
        ),
    )


LOG_HALF: Final[float] = math.log(0.5)
r"""CONST: log(0.5) is used in the tail handling of the erfinv computation."""


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
        y = y - torch.log(weight_k) * (sigma / z)
    return y


USE_NAIVE = False


def psi(x, mu, sigma, omegas, mus, sigmas):
    # compute zₖ=(x - μₖ)/(σₖ√2) for all k.
    # x has shape (batch_size, d)
    # mus and sigmas have shape (num_components,)
    # we want z to have shape (batch_size, num_components)
    z = (x.unsqueeze(-1) - mus) / sigmas
    assert z.shape[-1] == mus.shape[0] == sigmas.shape[0] == omegas.shape[0]

    log_w = torch.log(omegas)

    # compute ℓₖ=log Φ(zₖ) using log_ndtr for numerical stability.
    log_phi = torch.special.log_ndtr(z)
    log_iphi = torch.special.log_ndtr(-z)

    # compute log ∑ₖωₖΦ(zₖ) using logsumexp for numerical stability.
    log_p = torch.logsumexp(log_w + log_phi, dim=-1)
    log_q = torch.logsumexp(log_w + log_iphi, dim=-1)

    # use either branch depending on the sign
    u = torch.where(
        log_p < LOG_HALF,
        ndtri_exp(log_p),
        -ndtri_exp(log_q),
    )
    assert u.isfinite().all()

    return u * sigma + mu


def test_plot_transport_3():
    dtype = torch.float64

    x = torch.linspace(-10, 10, 100, dtype=dtype)

    mu = torch.tensor(0, dtype=dtype)
    sigma = torch.tensor(1, dtype=dtype)
    omegas = torch.tensor([2, 1, 2], dtype=dtype)
    omegas = omegas / omegas.sum()
    mus = torch.tensor([-3, -6, 3], dtype=dtype)
    sigmas = torch.tensor([0.5, 0.5, 0.5], dtype=dtype)

    y = psi(x, mu, sigma, omegas, mus, sigmas)

    with plt.style.context("bmh"):
        fig, ax = plt.subplots()
        ax.plot(x, y, lw=5)
        for k in range(mus.numel()):
            ax.plot(
                x, asymptotic_line(x, mu, sigma, omegas[k], mus[k], sigmas[k]), "k--"
            )
        ax.set_title("Optimal Transport from N(0, 1) to mixture of 3 Gaussians")
        fig.show()


def test_plot_transport_2():
    dtype = torch.float64

    x = torch.linspace(-10, 10, 100, dtype=dtype)

    mu = torch.tensor(0, dtype=dtype)
    sigma = torch.tensor(1, dtype=dtype)
    omegas = torch.tensor([2, 1], dtype=dtype)
    omegas = omegas / omegas.sum()
    mus = torch.tensor([-3, 3], dtype=dtype)
    sigmas = torch.tensor([0.5, 0.5], dtype=dtype)

    y = psi(x, mu, sigma, omegas, mus, sigmas)

    result_dir = PROJECT.RESULTS_DIR[__file__] / "transport_plots"
    result_dir.mkdir(exist_ok=True)

    with plt.style.context("bmh"):
        fig, ax = plt.subplots()
        ax.plot(x, y, lw=5)
        for k in range(mus.numel()):
            ax.plot(
                x, asymptotic_line(x, mu, sigma, omegas[k], mus[k], sigmas[k]), "k--"
            )

        file = result_dir / "transport_2.png"
        ax.set_title("Optimal Transport from N(0, 1) to mixture of 2 Gaussians")
        fig.savefig(file, dpi=300)
        print(f"Saved transport plot to {file!s}")


def _slider_positions(
    count: int, /, *, bottom: float = 0.02, height: float = 0.025, gap: float = 0.005
) -> list[list[float]]:
    return [[0.12, bottom + i * (height + gap), 0.76, height] for i in range(count)]


@dataclass
class TransportPlotState:
    x: Tensor
    line: Any
    component_lines: list[Any]
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

        y = psi(self.x, mu, sigma, omegas, mus, sigmas)

        self.line.set_ydata(y)
        for k, line in enumerate(self.component_lines):
            line.set_ydata(
                asymptotic_line(self.x, mu, sigma, omegas[k], mus[k], sigmas[k])
            )
        self.line.figure.canvas.draw_idle()


@dataclass
class TransportPlotState2:
    x: Tensor
    line: Any
    component_lines: list[Any]
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

        y = psi(self.x, mu, sigma, omegas, mus, sigmas)

        self.line.set_ydata(y)
        for k, line in enumerate(self.component_lines):
            line.set_ydata(
                asymptotic_line(self.x, mu, sigma, omegas[k], mus[k], sigmas[k])
            )
        self.line.figure.canvas.draw_idle()


def test_plot_transport_3_interactive():
    dtype = torch.float64

    x = torch.linspace(-10, 10, 200, dtype=dtype)

    mu = torch.tensor(0, dtype=dtype)
    sigma = torch.tensor(1, dtype=dtype)
    omegas = torch.tensor([2, 1, 2], dtype=dtype)
    omegas = omegas / omegas.sum()
    mus = torch.tensor([-3, 0, 3], dtype=dtype)
    sigmas = torch.tensor([0.5, 0.5, 0.5], dtype=dtype)

    y = psi(x, mu, sigma, omegas, mus, sigmas)

    with plt.style.context("bmh"):
        fig, ax = plt.subplots(figsize=(10, 7))
        (line,) = ax.plot(x, y, label="transport", lw=5)
        component_lines = [
            ax.plot(
                x, asymptotic_line(x, mu, sigma, omegas[k], mus[k], sigmas[k]), "k--"
            )[0]
            for k in range(mus.numel())
        ]
        ax.set_title("Interactive Transport to mixture of 3 Gaussians")
        ax.legend(loc="upper left")

        slider_specs = [
            ("mu", "mu", -5.0, 5.0, float(mu)),
            ("sigma", "sigma", 0.2, 5.0, float(sigma)),
            ("omega_1", "omega_1", 0.1, 5.0, float(omegas[0])),
            ("omega_2", "omega_2", 0.1, 5.0, float(omegas[1])),
            ("omega_3", "omega_3", 0.1, 5.0, float(omegas[2])),
            ("mu_1", "mu_1", -8.0, 8.0, float(mus[0])),
            ("mu_2", "mu_2", -8.0, 8.0, float(mus[1])),
            ("mu_3", "mu_3", -8.0, 8.0, float(mus[2])),
            ("sigma_1", "sigma_1", 0.1, 3.0, float(sigmas[0])),
            ("sigma_2", "sigma_2", 0.1, 3.0, float(sigmas[1])),
            ("sigma_3", "sigma_3", 0.1, 3.0, float(sigmas[2])),
        ]

        fig.subplots_adjust(bottom=0.4)
        axes = [fig.add_axes(pos) for pos in _slider_positions(len(slider_specs))]
        sliders = {
            key: Slider(ax_, label, min_, max_, valinit=init)
            for ax_, (key, label, min_, max_, init) in zip(
                axes, slider_specs, strict=True
            )
        }

        state = TransportPlotState(
            x=x, line=line, component_lines=component_lines, sliders=sliders
        )
        for slider in sliders.values():
            slider.on_changed(state.update)

        plt.show()


def test_plot_transport_2_interactive():
    dtype = torch.float64

    x = torch.linspace(-10, 10, 200, dtype=dtype)

    mu = torch.tensor(0, dtype=dtype)
    sigma = torch.tensor(1, dtype=dtype)
    omegas = torch.tensor([2, 1], dtype=dtype)
    omegas = omegas / omegas.sum()
    mus = torch.tensor([-3, 3], dtype=dtype)
    sigmas = torch.tensor([0.5, 0.5], dtype=dtype)

    y = psi(x, mu, sigma, omegas, mus, sigmas)

    with plt.style.context("bmh"):
        fig, ax = plt.subplots(figsize=(10, 6))
        (line,) = ax.plot(x, y, label="transport", lw=5)
        component_lines = [
            ax.plot(
                x, asymptotic_line(x, mu, sigma, omegas[k], mus[k], sigmas[k]), "k--"
            )[0]
            for k in range(mus.numel())
        ]
        ax.set_title("Interactive Transport to mixture of 2 Gaussians")
        ax.legend(loc="upper left")

        slider_specs = [
            ("mu", "mu", -5.0, 5.0, float(mu)),
            ("sigma", "sigma", 0.2, 5.0, float(sigma)),
            ("omega_1", "omega_1", 0.1, 5.0, float(omegas[0])),
            ("omega_2", "omega_2", 0.1, 5.0, float(omegas[1])),
            ("mu_1", "mu_1", -8.0, 8.0, float(mus[0])),
            ("mu_2", "mu_2", -8.0, 8.0, float(mus[1])),
            ("sigma_1", "sigma_1", 0.1, 3.0, float(sigmas[0])),
            ("sigma_2", "sigma_2", 0.1, 3.0, float(sigmas[1])),
        ]

        fig.subplots_adjust(bottom=0.32)
        axes = [fig.add_axes(pos) for pos in _slider_positions(len(slider_specs))]
        sliders = {
            key: Slider(ax_, label, min_, max_, valinit=init)
            for ax_, (key, label, min_, max_, init) in zip(
                axes, slider_specs, strict=True
            )
        }

        state = TransportPlotState2(
            x=x, line=line, component_lines=component_lines, sliders=sliders
        )
        for slider in sliders.values():
            slider.on_changed(state.update)

        plt.show()
