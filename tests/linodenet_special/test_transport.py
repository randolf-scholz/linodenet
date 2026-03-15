r"""Tests for transport maps."""

import math

import pytest
import torch
from torch.autograd import gradcheck

from linodenet_special import (
    bimodal_to_gaussian,
    gaussian_to_bimodal,
    gaussian_to_mixture,
    hard_bend,
    mixture_to_gaussian,
)
from linodenet_special.compiled import gaussian_to_mixture as gaussian_to_mixture_cpp

from .fixtures import DEVICES, DTYPES, Fixture


class TestBimodalToGaussian(Fixture):
    X_MIN = -20
    X_MAX = 20
    N = 1000
    N_FEW = 32

    STDVS = [0.1, 0.5, 1, 2, 10]
    MEANS = [0.1, 0.5, 1, 2, 10]

    @staticmethod
    def get_x_star(mean: float, stdv: float) -> float:
        """Critical point of the piecewise-linear approximation.

        Given λ=Ψ'(0)=exp(-½μ²/σ²)/σ, it's λx = (x±μ)/σ ⟺ x = ±μ/(1-λσ)
        """
        lam = math.exp(-0.5 * (mean / stdv) ** 2) / stdv
        return mean * min(1.0, abs(1 / (1 - lam * stdv)))

    @pytest.mark.parametrize("device", DEVICES, ids=str)
    @pytest.mark.parametrize("stdv", STDVS, ids="stdv={}".format)
    @pytest.mark.parametrize("mean", MEANS, ids="mean={}".format)
    @pytest.mark.parametrize("dtype", DTYPES, ids=str)
    def test_hard_contract_approximation(
        self, dtype: torch.dtype, mean: float, stdv: float, device: str
    ) -> None:
        r"""When the gaussians are well separated, we can approximate with hard_bend."""
        x = torch.linspace(
            self.X_MIN, self.X_MAX, steps=self.N, dtype=dtype, device=device
        )
        μ = torch.tensor(mean, dtype=dtype, device=device)
        σ = torch.tensor(stdv, dtype=dtype, device=device)
        λ = torch.exp(-0.5 * (μ / σ) ** 2) / σ

        y = bimodal_to_gaussian(x, μ, σ)
        assert y.dtype == dtype
        assert y.isfinite().all(), (
            "bimodal_to_gaussian should produce finite outputs for finite inputs"
        )

        y_approx = hard_bend(x, λ, μ / σ, 1 / σ)
        assert y_approx.dtype == dtype
        assert y_approx.isfinite().all(), (
            "Hard-contract approximation should produce finite outputs"
        )
        self.assert_upper_bounded(y - y_approx, μ / σ)

    @pytest.mark.parametrize("device", DEVICES, ids=str)
    @pytest.mark.parametrize("stdv", STDVS, ids="stdv={}".format)
    @pytest.mark.parametrize("mean", MEANS, ids="mean={}".format)
    @pytest.mark.parametrize("dtype", DTYPES, ids=str)
    def test_bimodal_to_gaussian_forward(
        self, dtype: torch.dtype, mean: float, stdv: float, device: str
    ) -> None:
        μ = torch.tensor(mean, dtype=dtype, device=device)
        σ = torch.tensor(stdv, dtype=dtype, device=device)
        λ = (torch.exp(-0.5 * (μ / σ) ** 2) / σ).item()

        zero = torch.tensor(0, dtype=dtype, device=device)
        y_zero = bimodal_to_gaussian(zero, μ, σ)
        self.assert_close(y_zero, zero)

        x1 = torch.linspace(0, self.X_MAX, steps=self.N, dtype=dtype, device=device)
        y1 = bimodal_to_gaussian(x1, μ, σ)
        assert y1.dtype == dtype
        assert y1.isfinite().all()

        x2 = torch.linspace(0, self.X_MIN, steps=self.N, dtype=dtype, device=device)
        y2 = bimodal_to_gaussian(x2, μ, σ)
        assert y2.dtype == dtype
        assert y2.isfinite().all()

        self.assert_close(y1, -y2)

        x_tail = max(100.0, μ.item() * max(1, 1 / (1 - λ)))
        assert x_tail > 0
        x1 = torch.linspace(
            100 * x_tail, 1000 * x_tail, steps=self.N, dtype=dtype, device=device
        )
        x2 = -x1
        tail1 = (x1 - torch.sign(x1) * μ) / σ
        tail2 = (x2 - torch.sign(x2) * μ) / σ
        y1 = bimodal_to_gaussian(x1, μ, σ)
        y2 = bimodal_to_gaussian(x2, μ, σ)
        assert y1.isfinite().all()
        assert y2.isfinite().all()
        self.assert_close(y1, tail1)
        self.assert_close(y2, tail2)

    @pytest.mark.parametrize("device", DEVICES, ids=str)
    @pytest.mark.parametrize("stdv", STDVS, ids="stdv={}".format)
    @pytest.mark.parametrize("mean", MEANS, ids="mean={}".format)
    @pytest.mark.parametrize("dtype", DTYPES, ids=str)
    def test_bimodal_to_gaussian_backward(
        self, dtype: torch.dtype, mean: float, stdv: float, device: str
    ) -> None:
        μ = torch.tensor(mean, dtype=dtype, device=device)
        σ = torch.tensor(stdv, dtype=dtype, device=device)
        λ = torch.exp(-0.5 * (μ / σ) ** 2) / stdv
        g_rtol = 2**-4
        lower_grad_bound = max(0, λ.item() * (1 - g_rtol))
        upper_grad_bound = 1 / stdv

        x1 = torch.linspace(
            0,
            self.X_MAX,
            steps=self.N,
            dtype=dtype,
            device=device,
            requires_grad=True,
        )
        y1 = bimodal_to_gaussian(x1, μ, σ)
        y1.sum().backward()
        assert x1.grad is not None
        assert x1.grad.isfinite().all()
        assert x1.grad.max() <= upper_grad_bound
        assert x1.grad.min() >= lower_grad_bound
        self.assert_close(x1.grad[0], λ, rtol=g_rtol)

        x2 = torch.linspace(
            0,
            self.X_MIN,
            steps=self.N,
            dtype=dtype,
            device=device,
            requires_grad=True,
        )
        y2 = bimodal_to_gaussian(x2, μ, σ)
        y2.sum().backward()
        assert x2.grad is not None
        assert x2.grad.isfinite().all()
        assert x2.grad.max() <= upper_grad_bound
        assert x2.grad.min() >= lower_grad_bound
        self.assert_close(x2.grad[0], λ, rtol=g_rtol)
        self.assert_close(x1.grad, x2.grad)

        x_tail = self.get_x_star(mean, stdv)
        assert x_tail > 0
        tail_values = torch.linspace(
            10 * x_tail, 100 * x_tail, steps=self.N, dtype=dtype, device=device
        )
        tail = torch.cat([tail_values, tail_values.neg()]).requires_grad_()
        y_tail = bimodal_to_gaussian(tail, μ, σ)
        assert y_tail.isfinite().all()
        y_tail.sum().backward()
        assert tail.grad is not None
        assert tail.grad.isfinite().all()
        self.assert_close(tail.grad, upper_grad_bound, rtol=0.5)

    @pytest.mark.parametrize("device", DEVICES, ids=str)
    @pytest.mark.parametrize("stdv", STDVS, ids="stdv={}".format)
    @pytest.mark.parametrize("mean", MEANS, ids="mean={}".format)
    @pytest.mark.parametrize("dtype", DTYPES, ids=str)
    def test_bimodal_to_gaussian_gradcheck(
        self, dtype: torch.dtype, mean: float, stdv: float, device: str
    ) -> None:
        μ = torch.tensor(mean, dtype=dtype, device=device, requires_grad=True)
        σ = torch.tensor(stdv, dtype=dtype, device=device, requires_grad=True)
        x_star = self.get_x_star(mean, stdv)
        x_narrow = torch.linspace(
            -x_star,
            x_star,
            steps=self.N_FEW,
            dtype=dtype,
            device=device,
            requires_grad=True,
        )

        match dtype:
            case torch.float32:
                atol, rtol, eps = 1e-2, 1e-2, 1e-4
            case torch.float64:
                atol, rtol, eps = 1e-6, 1e-6, 1e-8
            case _:
                raise ValueError(f"Unsupported dtype: {dtype}")

        gradcheck(bimodal_to_gaussian, (x_narrow, μ, σ), atol=atol, rtol=rtol, eps=eps)

    @pytest.mark.parametrize("device", DEVICES, ids=str)
    @pytest.mark.parametrize("stdv", [0.5, 1, 2, 10], ids="stdv={}".format)
    @pytest.mark.parametrize("mean", [0.1, 0.5, 1, 2], ids="mean={}".format)
    @pytest.mark.parametrize("dtype", DTYPES, ids=str)
    def test_reversible(
        self, dtype: torch.dtype, mean: float, stdv: float, device: str
    ) -> None:
        μ = torch.tensor(mean, dtype=dtype, device=device)
        σ = torch.tensor(stdv, dtype=dtype, device=device)
        λ = torch.exp(-0.5 * (μ / σ) ** 2) / σ
        x_star = μ * min(1, 1 / (1 - λ.item()))
        x = torch.linspace(
            -x_star,
            x_star,
            steps=self.N_FEW,
            dtype=dtype,
            device=device,
            requires_grad=True,
        )
        y = bimodal_to_gaussian(x, μ, σ)
        x_inv = gaussian_to_bimodal(y, μ, σ)
        z = x_inv.sum()
        z.backward()
        assert x.grad is not None
        self.assert_close(x_inv, x, rtol=1e-4, atol=1e-4)
        self.assert_close(x.grad, 1.0, rtol=1e-4, atol=1e-4)


class TestGaussianToBimodal(Fixture):
    X_MIN = -20
    X_MAX = 20
    N = 1000
    N_FEW = 32
    STDVS = [1, 2, 3]
    MEANS = [0.5, 1, 2]

    @staticmethod
    def get_x_star(mean: float, stdv: float) -> float:
        """Critical point of the piecewise-linear approximation.

        Given λ=Ψ⁻¹'(0)=σ⋅exp(½μ²/σ²), it's λx = σx±μ ⟺ x = ±μ/(λ-σ),
        """
        lam = stdv * math.exp(0.5 * (mean / stdv) ** 2)
        return abs(mean / (lam - stdv))

    @pytest.mark.parametrize("device", DEVICES, ids=str)
    @pytest.mark.parametrize("stdv", STDVS, ids="stdv={}".format)
    @pytest.mark.parametrize("mean", MEANS, ids="mean={}".format)
    @pytest.mark.parametrize("dtype", DTYPES, ids=str)
    def test_hard_expand_approximation(
        self, dtype: torch.dtype, mean, stdv, device: str
    ) -> None:
        y = torch.linspace(
            self.X_MIN, self.X_MAX, steps=self.N, dtype=dtype, device=device
        )
        μ = torch.tensor(mean, dtype=dtype, device=device)
        σ = torch.tensor(stdv, dtype=dtype, device=device)
        λ = (torch.exp(-0.5 * (μ / σ) ** 2) / σ).item()

        x = gaussian_to_bimodal(y, μ, σ)
        assert x.dtype == dtype
        assert x.isfinite().all(), (
            "gaussian_to_bimodal should produce finite outputs for finite inputs"
        )

        x_approx = hard_bend(y, 1 / λ, μ, σ)
        assert x_approx.dtype == dtype
        assert x_approx.isfinite().all(), (
            "Hard-expand approximation should produce finite outputs"
        )
        self.assert_upper_bounded(x - x_approx, μ * σ, atol=1e-1, rtol=1e-1)

    @pytest.mark.parametrize("device", DEVICES, ids=str)
    @pytest.mark.parametrize("stdv", STDVS, ids="stdv={}".format)
    @pytest.mark.parametrize("mean", MEANS, ids="mean={}".format)
    @pytest.mark.parametrize("dtype", DTYPES, ids=str)
    def test_gaussian_to_bimodal_forward(
        self, dtype: torch.dtype, mean: float, stdv: float, device: str
    ) -> None:
        μ = torch.tensor(mean, dtype=dtype, device=device)
        σ = torch.tensor(stdv, dtype=dtype, device=device)
        λ = (torch.exp(-0.5 * (μ / σ) ** 2) / σ).item()

        zero = torch.tensor(0, dtype=dtype, device=device)
        x_zero = gaussian_to_bimodal(zero, μ, σ)
        self.assert_close(x_zero, zero)

        y1 = torch.linspace(0, self.X_MAX, steps=self.N_FEW, dtype=dtype, device=device)
        x1 = gaussian_to_bimodal(y1, μ, σ)
        assert x1.dtype == dtype
        assert x1.isfinite().all()

        y2 = torch.linspace(0, self.X_MIN, steps=self.N_FEW, dtype=dtype, device=device)
        x2 = gaussian_to_bimodal(y2, μ, σ)
        assert x2.dtype == dtype
        assert x2.isfinite().all()

        self.assert_close(x1, -x2)

        y_tail = max(100.0, μ.abs().item() / (λ - 1))
        assert y_tail > 0
        y1 = torch.linspace(
            100 * y_tail, 1000 * y_tail, steps=self.N_FEW, dtype=dtype, device=device
        )
        y2 = -y1
        tail1 = σ * y1 - μ
        tail2 = σ * y2 + μ
        x1 = gaussian_to_bimodal(y1, μ, σ)
        x2 = gaussian_to_bimodal(y2, μ, σ)
        assert x1.isfinite().all()
        assert x2.isfinite().all()
        self.assert_close(x1, tail1, rtol=1e-3)
        self.assert_close(x2, tail2, rtol=1e-3)

    @pytest.mark.parametrize("device", DEVICES, ids=str)
    @pytest.mark.parametrize("stdv", STDVS, ids="stdv={}".format)
    @pytest.mark.parametrize("mean", MEANS, ids="mean={}".format)
    @pytest.mark.parametrize("dtype", DTYPES, ids=str)
    def test_gaussian_to_bimodal_backward(
        self, dtype: torch.dtype, mean: float, stdv: float, device: str
    ) -> None:
        μ = torch.tensor(mean, dtype=dtype, device=device)
        σ = torch.tensor(stdv, dtype=dtype, device=device)
        λ_log = 0.5 * (μ / σ) ** 2 + σ.log()
        g_rtol = 2**-4
        log_tol = math.log(1 + g_rtol)
        log_grad_bound = λ_log + log_tol

        y1 = torch.linspace(
            0,
            self.X_MAX,
            steps=self.N,
            dtype=dtype,
            device=device,
            requires_grad=True,
        )
        x1 = gaussian_to_bimodal(y1, μ, σ)
        x1.sum().backward()
        assert y1.grad is not None
        assert y1.grad.isfinite().all()
        assert y1.grad.min() >= min(1, σ.item())
        assert y1.grad.log().max() <= log_grad_bound

        y2 = torch.linspace(
            0,
            self.X_MIN,
            steps=self.N,
            dtype=dtype,
            device=device,
            requires_grad=True,
        )
        x2 = gaussian_to_bimodal(y2, μ, σ)
        x2.sum().backward()
        assert y2.grad is not None
        assert y2.grad.isfinite().all()
        assert y2.grad.min() >= min(1, σ.item())
        assert y2.grad.log().max() <= log_grad_bound

        self.assert_close(y1.grad, y2.grad)

        y_tail = self.get_x_star(mean, stdv)
        assert y_tail > 0
        tail_values = torch.linspace(
            10 * y_tail, 100 * y_tail, steps=self.N, dtype=dtype, device=device
        )
        tail = torch.cat([tail_values, tail_values.neg()]).requires_grad_()
        x_tail = gaussian_to_bimodal(tail, μ, σ)
        assert x_tail.isfinite().all()
        x_tail.sum().backward()
        assert tail.grad is not None
        assert tail.grad.isfinite().all()
        # FIXME: huge rtol needed?!
        self.assert_close(tail.grad, σ, atol=1e-3, rtol=1e-1)

    @pytest.mark.parametrize("device", DEVICES, ids=str)
    @pytest.mark.parametrize("stdv", STDVS, ids="stdv={}".format)
    @pytest.mark.parametrize("mean", MEANS, ids="mean={}".format)
    @pytest.mark.parametrize("dtype", DTYPES, ids=str)
    def test_gaussian_to_bimodal_gradcheck(
        self, dtype: torch.dtype, mean: float, stdv: float, device: str
    ) -> None:
        μ = torch.tensor(mean, dtype=dtype, device=device, requires_grad=True)
        σ = torch.tensor(stdv, dtype=dtype, device=device, requires_grad=True)
        y_star = self.get_x_star(mean, stdv)
        y_narrow = torch.linspace(
            -y_star / 2,
            y_star / 2,
            steps=self.N_FEW,
            dtype=dtype,
            device=device,
            requires_grad=True,
        )

        match dtype:
            case torch.float32:
                atol, rtol, eps = 1e-2, 1e-2, 1e-4
            case torch.float64:
                atol, rtol, eps = 1e-6, 1e-6, 1e-8
            case _:
                raise ValueError(f"Unsupported dtype: {dtype}")

        gradcheck(gaussian_to_bimodal, (y_narrow, μ, σ), atol=atol, rtol=rtol, eps=eps)

    @pytest.mark.parametrize("device", DEVICES, ids=str)
    @pytest.mark.parametrize("stdv", [0.5, 1, 2, 10], ids="stdv={}".format)
    @pytest.mark.parametrize("mean", [0.1, 0.5, 1, 2], ids="mean={}".format)
    @pytest.mark.parametrize("dtype", DTYPES, ids=str)
    def test_reversible(
        self, dtype: torch.dtype, mean: float, stdv: float, device: str
    ) -> None:
        μ = torch.tensor(mean, dtype=dtype, device=device)
        σ = torch.tensor(stdv, dtype=dtype, device=device)
        y = torch.linspace(
            self.X_MIN,
            self.X_MAX,
            steps=self.N,
            dtype=dtype,
            device=device,
            requires_grad=True,
        )
        x_inv = gaussian_to_bimodal(y, μ, σ)
        y_inv = bimodal_to_gaussian(x_inv, μ, σ)
        z = y_inv.sum()
        z.backward()
        assert y.grad is not None
        self.assert_close(y_inv, y, rtol=1e-4, atol=1e-4)
        self.assert_close(y.grad, 1.0, rtol=1e-4, atol=1e-4)

    @pytest.mark.parametrize("device", DEVICES, ids=str)
    @pytest.mark.parametrize("stdv", STDVS, ids="stdv={}".format)
    @pytest.mark.parametrize("mean", MEANS, ids="mean={}".format)
    @pytest.mark.parametrize("dtype", DTYPES, ids=str)
    def test_negative_mu_matches_positive_mu(
        self, dtype: torch.dtype, mean: float, stdv: float, device: str
    ) -> None:
        y = torch.linspace(
            self.X_MIN, self.X_MAX, steps=self.N_FEW, dtype=dtype, device=device
        )
        x = torch.linspace(
            self.X_MIN, self.X_MAX, steps=self.N_FEW, dtype=dtype, device=device
        )
        μ_pos = torch.tensor(mean, dtype=dtype, device=device)
        μ_neg = torch.tensor(-mean, dtype=dtype, device=device)
        σ = torch.tensor(stdv, dtype=dtype, device=device)

        self.assert_close(
            gaussian_to_bimodal(y, μ_pos, σ),
            gaussian_to_bimodal(y, μ_neg, σ),
        )
        self.assert_close(
            bimodal_to_gaussian(x, μ_pos, σ),
            bimodal_to_gaussian(x, μ_neg, σ),
        )


class TestMixtureToGaussian(Fixture):
    N = 64

    @pytest.mark.parametrize("device", DEVICES, ids=str)
    @pytest.mark.parametrize("dtype", DTYPES, ids=str)
    @pytest.mark.parametrize(
        ("weights", "means", "stdvs"),
        [
            pytest.param(
                [0.4, 0.25, 0.35],
                [-1.0, 0.5, 1.5],
                [0.8, 1.1, 0.9],
                id="asymmetric",
            ),
            pytest.param(
                [0.2, 0.5, 0.3],
                [-1.5, -0.5, 1.0],
                [1.0, 0.8, 1.2],
                id="shifted",
            ),
        ],
    )
    def test_reversible(
        self,
        dtype: torch.dtype,
        weights: list[float],
        means: list[float],
        stdvs: list[float],
        device: str,
    ) -> None:
        omegas = torch.tensor(weights, dtype=dtype, device=device)
        mus = torch.tensor(means, dtype=dtype, device=device)
        sigmas = torch.tensor(stdvs, dtype=dtype, device=device)
        x_min = torch.min(mus - 3 * sigmas).item()
        x_max = torch.max(mus + 3 * sigmas).item()
        x = torch.linspace(
            x_min,
            x_max,
            steps=self.N,
            dtype=dtype,
            device=device,
            requires_grad=True,
        )

        y = mixture_to_gaussian(x, omegas, mus, sigmas)
        x_inv = gaussian_to_mixture(y, omegas, mus, sigmas)
        x_inv.sum().backward()
        assert x.grad is not None
        self.assert_close(x_inv, x, rtol=1e-4, atol=1e-4)
        self.assert_close(x.grad, 1.0, rtol=1e-4, atol=1e-4)

    @pytest.mark.parametrize("device", DEVICES, ids=str)
    @pytest.mark.parametrize("dtype", DTYPES, ids=str)
    @pytest.mark.parametrize(
        ("weights", "means", "stdvs"),
        [
            pytest.param(
                [0.4, 0.25, 0.35],
                [-1.0, 0.5, 1.5],
                [0.8, 1.1, 0.9],
                id="asymmetric",
            ),
            pytest.param(
                [0.2, 0.5, 0.3],
                [-1.5, -0.5, 1.0],
                [1.0, 0.8, 1.2],
                id="shifted",
            ),
        ],
    )
    @pytest.mark.parametrize(
        "values",
        [
            pytest.param([[-1.25, -0.5], [0.25, 1.75]], id="batch"),
            pytest.param(0.375, id="scalar"),
            pytest.param([-3.0, -2.25, -1.5, -0.5, -0.1], id="p_branch"),
            pytest.param([0.1, 0.5, 1.5, 2.25, 3.0], id="q_branch"),
        ],
    )
    def test_gradcheck(
        self,
        values: list[float] | float,
        weights: list[float],
        means: list[float],
        stdvs: list[float],
        device: str,
        dtype: torch.dtype,
    ) -> None:
        x = torch.tensor(values, dtype=dtype, device=device, requires_grad=True)
        omegas = torch.tensor(weights, dtype=dtype, device=device, requires_grad=True)
        mus = torch.tensor(means, dtype=dtype, device=device, requires_grad=True)
        sigmas = torch.tensor(stdvs, dtype=dtype, device=device, requires_grad=True)

        if dtype is torch.float32:
            eps = 1e-4
            atol = 1e-2
            rtol = 1e-3
        else:
            eps = 1e-6
            atol = 1e-6
            rtol = 1e-6

        gradcheck(
            lambda z, ω, μ, σ: mixture_to_gaussian(z, ω / ω.sum(), μ, σ),
            (x, omegas, mus, sigmas),
            eps=eps,
            atol=atol,
            rtol=rtol,
        )


class TestGaussianToMixture(Fixture):
    N = 64

    @pytest.mark.parametrize("device", DEVICES, ids=str)
    @pytest.mark.parametrize("dtype", DTYPES, ids=str)
    @pytest.mark.parametrize(
        ("weights", "means", "stdvs"),
        [
            pytest.param(
                [0.4, 0.25, 0.35],
                [-1.0, 0.5, 1.5],
                [0.8, 1.1, 0.9],
                id="asymmetric",
            ),
            pytest.param(
                [0.2, 0.5, 0.3],
                [-1.5, -0.5, 1.0],
                [1.0, 0.8, 1.2],
                id="shifted",
            ),
        ],
    )
    def test_reversible(
        self,
        dtype: torch.dtype,
        weights: list[float],
        means: list[float],
        stdvs: list[float],
        device: str,
    ) -> None:
        omegas = torch.tensor(weights, dtype=dtype, device=device)
        mus = torch.tensor(means, dtype=dtype, device=device)
        sigmas = torch.tensor(stdvs, dtype=dtype, device=device)
        y = torch.linspace(
            -4,
            4,
            steps=self.N,
            dtype=dtype,
            device=device,
            requires_grad=True,
        )

        x = gaussian_to_mixture(y, omegas, mus, sigmas)
        y_inv = mixture_to_gaussian(x, omegas, mus, sigmas)
        y_inv.sum().backward()
        assert y.grad is not None
        self.assert_close(y_inv, y, rtol=1e-4, atol=1e-4)
        self.assert_close(y.grad, 1.0, rtol=1e-4, atol=1e-4)

    @pytest.mark.parametrize("device", DEVICES, ids=str)
    @pytest.mark.parametrize("dtype", DTYPES, ids=str)
    @pytest.mark.parametrize(
        ("weights", "means", "stdvs"),
        [
            pytest.param(
                [0.4, 0.25, 0.35],
                [-1.0, 0.5, 1.5],
                [0.8, 1.1, 0.9],
                id="asymmetric",
            ),
            pytest.param(
                [0.2, 0.5, 0.3],
                [-1.5, -0.5, 1.0],
                [1.0, 0.8, 1.2],
                id="shifted",
            ),
        ],
    )
    @pytest.mark.parametrize(
        "values",
        [
            pytest.param([[-2.0, -0.75], [0.25, 1.5]], id="batch"),
            pytest.param(-0.375, id="scalar"),
            pytest.param([-3.0, -1.5, -0.5, 0.0, 0.5], id="left"),
            pytest.param([0.0, 0.5, 1.5, 2.25, 3.0], id="right"),
        ],
    )
    def test_gradcheck(
        self,
        values: list[float] | float,
        weights: list[float],
        means: list[float],
        stdvs: list[float],
        device: str,
        dtype: torch.dtype,
    ) -> None:
        y = torch.tensor(values, dtype=dtype, device=device, requires_grad=True)
        omegas = torch.tensor(weights, dtype=dtype, device=device, requires_grad=True)
        mus = torch.tensor(means, dtype=dtype, device=device, requires_grad=True)
        sigmas = torch.tensor(stdvs, dtype=dtype, device=device, requires_grad=True)

        if dtype is torch.float32:
            eps = 1e-4
            atol = 1e-2
            rtol = 1e-2
        else:
            eps = 1e-6
            atol = 1e-6
            rtol = 1e-6

        gradcheck(
            lambda z, ω, μ, σ: gaussian_to_mixture(z, ω / ω.sum(), μ, σ),
            (y, omegas, mus, sigmas),
            eps=eps,
            atol=atol,
            rtol=rtol,
        )

    @pytest.mark.parametrize("device", DEVICES, ids=str)
    @pytest.mark.parametrize("dtype", DTYPES, ids=str)
    def test_compiled_scalar_backward_shapes(
        self,
        device: str,
        dtype: torch.dtype,
    ) -> None:
        y = torch.tensor(-0.375, dtype=dtype, device=device, requires_grad=True)
        omegas = torch.tensor(
            [0.2, 0.5, 0.3], dtype=dtype, device=device, requires_grad=True
        )
        mus = torch.tensor(
            [-1.5, -0.5, 1.0], dtype=dtype, device=device, requires_grad=True
        )
        sigmas = torch.tensor(
            [1.0, 0.8, 1.2], dtype=dtype, device=device, requires_grad=True
        )

        x = gaussian_to_mixture_cpp(y, omegas, mus, sigmas)
        x.backward()

        assert y.grad is not None
        assert omegas.grad is not None
        assert mus.grad is not None
        assert sigmas.grad is not None
