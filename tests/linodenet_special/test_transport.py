r"""Tests for transport maps."""

import math

import pytest
import torch
from torch.autograd import gradcheck

from linodenet_special.fallbacks.transport import (
    gaussian_to_twin,
    mixture_to_gaussian,
    twin_to_gaussian,
)
from linodenet_special.hard_contract import hard_contract, hard_expand
from tests.linodenet_special.fixtures import DEVICES, DTYPES

from .fixtures import Fixture


@pytest.mark.parametrize("device", DEVICES, ids=str)
@pytest.mark.parametrize("dtype", DTYPES, ids=str)
@pytest.mark.parametrize(
    ("weights", "means", "sigmas"),
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
        pytest.param(torch.randn(8, 2), id="batch"),
        pytest.param(torch.randn(()), id="scalar"),
        pytest.param([-3.0, -2.25, -1.5, -0.5, -0.1], id="p_branch"),
        pytest.param([0.1, 0.5, 1.5, 2.25, 3.0], id="q_branch"),
    ],
)
def test_mixture_to_gaussian_gradcheck(
    values: list[float],
    weights: list[float],
    means: list[float],
    sigmas: list[float],
    device: str,
    dtype: torch.dtype,
) -> None:
    x = torch.tensor(values, dtype=dtype, device=device, requires_grad=True)
    w = torch.tensor(weights, dtype=dtype, device=device, requires_grad=True)
    mu = torch.tensor(means, dtype=dtype, device=device, requires_grad=True)
    sigma = torch.tensor(sigmas, dtype=dtype, device=device, requires_grad=True)

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
        (x, w, mu, sigma),
        eps=eps,
        atol=atol,
        rtol=rtol,
    )


class TestTwinToGaussian(Fixture):
    X_MIN = -20
    X_MAX = 20
    N = 1000
    N_FEW = 32

    @pytest.mark.parametrize("device", DEVICES, ids=str)
    @pytest.mark.parametrize("sigma", [0.01, 0.1, 1, 10], ids=lambda x: f"s={x}")
    @pytest.mark.parametrize("mu", [0.5, 1, 2, 10], ids=lambda x: f"mu={x}")
    @pytest.mark.parametrize("dtype", DTYPES, ids=str)
    def test_hard_contract_approximation(
        self, dtype: torch.dtype, mu: float, sigma: float, device: str
    ) -> None:
        r"""When the gaussians are well separated, we can approximate with hard_bend."""
        x = torch.linspace(
            self.X_MIN, self.X_MAX, steps=self.N, dtype=dtype, device=device
        )
        μ = torch.tensor(mu, dtype=dtype, device=device)
        σ = torch.tensor(sigma, dtype=dtype, device=device)
        λ = torch.exp(-0.5 * (μ / σ) ** 2) / sigma

        y = twin_to_gaussian(x, μ, σ)
        assert y.dtype == dtype
        assert y.isfinite().all(), (
            "twin_to_gaussian should produce finite outputs for finite inputs"
        )

        y_approx = hard_contract(y, a=λ, c=mu)

        assert y_approx.dtype == dtype
        assert y_approx.isfinite().all(), (
            "Hard-contract approximation should produce finite outputs"
        )
        atol = torch.finfo(dtype).resolution
        rtol = torch.finfo(dtype).resolution
        assert (y_approx - y).abs().max() <= (1 + rtol) * μ.abs() + atol

    @pytest.mark.parametrize("device", DEVICES, ids=str)
    @pytest.mark.parametrize("sigma", [0.01, 0.1, 1, 10], ids=lambda x: f"s={x}")
    @pytest.mark.parametrize("mu", [0.1, 0.5, 1, 2, 10], ids=lambda x: f"mu={x}")
    @pytest.mark.parametrize("dtype", DTYPES, ids=str)
    def test_hard_expand_approximation(
        self, dtype: torch.dtype, mu: float, sigma: float, device: str
    ) -> None:
        y = torch.linspace(
            self.X_MIN, self.X_MAX, steps=self.N, dtype=dtype, device=device
        )
        μ = torch.tensor(mu, dtype=dtype, device=device)
        σ = torch.tensor(sigma, dtype=dtype, device=device)
        x = gaussian_to_twin(y, μ, σ)
        assert x.dtype == dtype
        assert x.isfinite().all(), (
            "gaussian_to_twin should produce finite outputs for finite inputs"
        )

    @pytest.mark.parametrize("device", DEVICES, ids=str)
    @pytest.mark.parametrize("sigma", [0.1, 0.5, 1, 2, 10], ids=lambda x: f"s={x}")
    @pytest.mark.parametrize("mu", [0.1, 0.5, 1, 2, 10], ids=lambda x: f"mu={x}")
    @pytest.mark.parametrize("dtype", DTYPES, ids=str)
    def test_twin_to_gaussian_forward(
        self, dtype: torch.dtype, mu: float, sigma: float, device: str
    ) -> None:
        μ = torch.tensor(mu, dtype=dtype, device=device)
        σ = torch.tensor(sigma, dtype=dtype, device=device)
        λ = torch.exp(-0.5 * (μ / σ) ** 2) / sigma

        zero = torch.tensor(0, dtype=dtype, device=device)
        y_zero = twin_to_gaussian(zero, μ, σ)
        self.assert_close(y_zero, zero)

        x1 = torch.linspace(0, self.X_MAX, steps=self.N, dtype=dtype, device=device)
        y1 = twin_to_gaussian(x1, μ, σ)
        assert y1.dtype == dtype
        assert y1.isfinite().all()

        x2 = torch.linspace(0, self.X_MIN, steps=self.N, dtype=dtype, device=device)
        y2 = twin_to_gaussian(x2, μ, σ)
        assert y2.dtype == dtype
        assert y2.isfinite().all()

        self.assert_close(y1, -y2)

        x_tail = max(100.0, μ.item() * max(1, 1 / (1 - λ.item())))
        assert x_tail > 0
        x1 = torch.linspace(
            100 * x_tail, 1000 * x_tail, steps=self.N, dtype=dtype, device=device
        )
        x2 = -x1
        tail1 = (x1 - torch.sign(x1) * μ) / sigma
        tail2 = (x2 - torch.sign(x2) * μ) / sigma
        y1 = twin_to_gaussian(x1, μ, σ)
        y2 = twin_to_gaussian(x2, μ, σ)
        self.assert_close(y1, tail1)
        self.assert_close(y2, tail2)

    @pytest.mark.parametrize("device", DEVICES, ids=str)
    @pytest.mark.parametrize("sigma", [0.1, 0.5, 1, 2, 10], ids=lambda x: f"s={x}")
    @pytest.mark.parametrize("mu", [0.1, 0.5, 1, 2, 10], ids=lambda x: f"mu={x}")
    @pytest.mark.parametrize("dtype", DTYPES, ids=str)
    def test_twin_to_gaussian_backward(
        self, dtype: torch.dtype, mu: float, sigma: float, device: str
    ) -> None:
        μ = torch.tensor(mu, dtype=dtype, device=device)
        σ = torch.tensor(sigma, dtype=dtype, device=device)
        λ = torch.exp(-0.5 * (μ / σ) ** 2) / sigma
        g_rtol = 2**-4
        lower_grad_bound = max(0, λ.item() * (1 - g_rtol))
        upper_grad_bound = 1 / sigma

        x1 = torch.linspace(
            0,
            self.X_MAX,
            steps=self.N,
            dtype=dtype,
            device=device,
            requires_grad=True,
        )
        y1 = twin_to_gaussian(x1, μ, σ)
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
        y2 = twin_to_gaussian(x2, μ, σ)
        y2.sum().backward()
        assert x2.grad is not None
        assert x2.grad.isfinite().all()
        assert x2.grad.max() <= upper_grad_bound
        assert x2.grad.min() >= lower_grad_bound
        self.assert_close(x2.grad[0], λ, rtol=g_rtol)
        self.assert_close(x1.grad, x2.grad)

        x_tail = max(10.0, μ.item() * max(1, 1 / (1 - λ.item())))
        assert x_tail > 0
        tail_values = torch.linspace(
            10 * x_tail, 100 * x_tail, steps=self.N, dtype=dtype, device=device
        )
        tail = torch.cat([tail_values, tail_values.neg()]).requires_grad_()
        y_tail = twin_to_gaussian(tail, μ, σ)
        assert y_tail.isfinite().all()
        y_tail.sum().backward()
        assert tail.grad is not None
        assert tail.grad.isfinite().all()
        self.assert_close(tail.grad, upper_grad_bound, rtol=0.5)

    @pytest.mark.parametrize("device", DEVICES, ids=str)
    @pytest.mark.parametrize("sigma", [0.1, 0.5, 1, 2, 10], ids=lambda x: f"s={x}")
    @pytest.mark.parametrize("mu", [0.1, 0.5, 1, 2, 10], ids=lambda x: f"mu={x}")
    @pytest.mark.parametrize("dtype", DTYPES, ids=str)
    def test_twin_to_gaussian_gradcheck(
        self, dtype: torch.dtype, mu: float, sigma: float, device: str
    ) -> None:
        μ = torch.tensor(mu, dtype=dtype, device=device, requires_grad=True)
        σ = torch.tensor(sigma, dtype=dtype, device=device, requires_grad=True)
        λ = torch.exp(-0.5 * (μ / σ) ** 2) / sigma
        x_star = μ * min(1, 1 / (1 - λ.item()))
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
                atol, rtol, eps = 1e-3, 1e-3, 1e-3
            case torch.float64:
                atol, rtol, eps = 1e-6, 1e-6, 1e-6
            case _:
                raise ValueError(f"Unsupported dtype: {dtype}")

        gradcheck(twin_to_gaussian, (x_narrow, μ, σ), atol=atol, rtol=rtol, eps=eps)

    @pytest.mark.parametrize("device", DEVICES, ids=str)
    @pytest.mark.parametrize("sigma", [0.1, 0.5, 1, 2, 10], ids=lambda x: f"s={x}")
    @pytest.mark.parametrize("mu", [0.1, 0.5, 1, 2, 10], ids=lambda x: f"mu={x}")
    @pytest.mark.parametrize("dtype", DTYPES, ids=str)
    def test_reversible(
        self, dtype: torch.dtype, mu: float, sigma: float, device: str
    ) -> None:
        μ = torch.tensor(mu, dtype=dtype, device=device)
        σ = torch.tensor(sigma, dtype=dtype, device=device)
        λ = torch.exp(-0.5 * (μ / σ) ** 2) / sigma
        x_star = μ * min(1, 1 / (1 - λ.item()))
        x = torch.linspace(
            -x_star,
            x_star,
            steps=self.N_FEW,
            dtype=dtype,
            device=device,
            requires_grad=True,
        )
        y = twin_to_gaussian(x, μ, σ)
        x_inv = gaussian_to_twin(y, μ, σ)
        z = x_inv.sum()
        z.backward()
        assert x.grad is not None
        self.assert_close(x_inv, x)
        self.assert_close(x.grad, torch.ones_like(x), rtol=1e-3, atol=1e-3)


class TestGaussianToTwin(Fixture):
    X_MIN = -20
    X_MAX = 20
    N = 1000
    N_FEW = 32

    @pytest.mark.parametrize("device", DEVICES, ids=str)
    @pytest.mark.parametrize("sigma", [0.01, 0.1, 1, 10], ids=lambda x: f"s={x}")
    @pytest.mark.parametrize("mu", [0.1, 0.5, 1, 2, 10], ids=lambda x: f"mu={x}")
    @pytest.mark.parametrize("dtype", DTYPES, ids=str)
    def test_hard_expand_approximation(
        self, dtype: torch.dtype, mu: float, sigma: float, device: str
    ) -> None:
        y = torch.linspace(
            self.X_MIN, self.X_MAX, steps=self.N, dtype=dtype, device=device
        )
        μ = torch.tensor(mu, dtype=dtype, device=device)
        σ = torch.tensor(sigma, dtype=dtype, device=device)
        lam = torch.exp(0.5 * (μ / σ) ** 2).item()

        x = gaussian_to_twin(y, μ, σ)
        assert x.dtype == dtype
        assert x.isfinite().all(), (
            "gaussian_to_twin should produce finite outputs for finite inputs"
        )

        x_approx = hard_expand(y, a=lam, c=mu)
        assert x_approx.dtype == dtype
        assert x_approx.isfinite().all(), (
            "Hard-expand approximation should produce finite outputs"
        )
        atol = torch.finfo(dtype).resolution
        rtol = torch.finfo(dtype).resolution
        assert (x_approx - x).abs().max() <= (1 + rtol) * μ.abs() + atol

    @pytest.mark.parametrize("device", DEVICES, ids=str)
    @pytest.mark.parametrize("sigma", [0.1, 0.5, 1, 2, 10], ids=lambda x: f"s={x}")
    @pytest.mark.parametrize("mu", [0.1, 0.5, 1, 2, 10], ids=lambda x: f"mu={x}")
    @pytest.mark.parametrize("dtype", DTYPES, ids=str)
    def test_gaussian_to_twin_forward(
        self, dtype: torch.dtype, mu: float, sigma: float, device: str
    ) -> None:
        μ = torch.tensor(mu, dtype=dtype, device=device)
        σ = torch.tensor(sigma, dtype=dtype, device=device)

        zero = torch.tensor(0, dtype=dtype, device=device)
        assert torch.allclose(gaussian_to_twin(zero, μ, σ), zero)

        y1 = torch.linspace(0, self.X_MAX, steps=self.N_FEW, dtype=dtype, device=device)
        x1 = gaussian_to_twin(y1, μ, σ)
        assert x1.dtype == dtype
        assert x1.isfinite().all()

        y2 = torch.linspace(0, self.X_MIN, steps=self.N_FEW, dtype=dtype, device=device)
        x2 = gaussian_to_twin(y2, μ, σ)
        assert x2.dtype == dtype
        assert x2.isfinite().all()

        assert torch.allclose(x1, -x2)

        lam = torch.exp(-0.5 * (μ / σ) ** 2).item()
        y_star = μ * max(1, lam / (1 - lam))
        assert y_star.item() > 0
        y1 = torch.linspace(
            10 * y_star, 100 * y_star, steps=self.N_FEW, dtype=dtype, device=device
        )
        y2 = torch.linspace(
            -10 * y_star, -100 * y_star, steps=self.N_FEW, dtype=dtype, device=device
        )
        tail1 = y1 + μ
        tail2 = y2 - μ
        x1 = gaussian_to_twin(y1, μ, σ)
        x2 = gaussian_to_twin(y2, μ, σ)
        assert x1.isfinite().all()
        assert x2.isfinite().all()
        assert torch.allclose(x1, tail1)
        assert torch.allclose(x2, tail2)

    @pytest.mark.parametrize("device", DEVICES, ids=str)
    @pytest.mark.parametrize("sigma", [0.1, 0.5, 1, 2, 10], ids=lambda x: f"s={x}")
    @pytest.mark.parametrize("mu", [0.1, 0.5, 1, 2, 10], ids=lambda x: f"mu={x}")
    @pytest.mark.parametrize("dtype", DTYPES, ids=str)
    def test_gaussian_to_twin_backward(
        self, dtype: torch.dtype, mu: float, sigma: float, device: str
    ) -> None:
        μ = torch.tensor(mu, dtype=dtype, device=device)
        σ = torch.tensor(sigma, dtype=dtype, device=device)
        lam = torch.exp(-0.5 * (μ / σ) ** 2).item()
        lam_inv_log = 0.5 * (μ / σ) ** 2
        g_rtol = 2**-4
        log_tol = math.log2(1 + g_rtol)
        log_grad_bound = lam_inv_log + log_tol

        y1 = torch.linspace(
            0,
            self.X_MAX,
            steps=self.N,
            dtype=dtype,
            device=device,
            requires_grad=True,
        )
        x1 = gaussian_to_twin(y1, μ, σ)
        x1.sum().backward()
        assert y1.grad is not None
        assert y1.grad.isfinite().all()
        assert y1.grad.min() >= 1
        assert y1.grad.log().max() <= log_grad_bound

        y2 = torch.linspace(
            0,
            self.X_MIN,
            steps=self.N,
            dtype=dtype,
            device=device,
            requires_grad=True,
        )
        x2 = gaussian_to_twin(y2, μ, σ)
        x2.sum().backward()
        assert y2.grad is not None
        assert y2.grad.isfinite().all()
        assert y2.grad.min() >= 1
        assert y2.grad.log().max() <= log_grad_bound

        assert torch.allclose(y1.grad, y2.grad)

        y_star = μ * max(1, lam / (1 - lam))
        assert y_star.item() > 0
        tail_values = torch.linspace(
            10 * y_star, 100 * y_star, steps=self.N, dtype=dtype, device=device
        )
        tail = torch.cat([tail_values, tail_values.neg()]).requires_grad_()
        x_tail = gaussian_to_twin(tail, μ, σ)
        assert x_tail.isfinite().all()
        x_tail.sum().backward()
        assert tail.grad is not None
        assert tail.grad.isfinite().all()
        assert torch.allclose(tail.grad, torch.ones_like(tail.grad))

    @pytest.mark.parametrize("device", DEVICES, ids=str)
    @pytest.mark.parametrize("sigma", [0.5, 1, 2, 10], ids=lambda x: f"s={x}")
    @pytest.mark.parametrize("mu", [0.1, 0.5, 1, 1.5], ids=lambda x: f"mu={x}")
    @pytest.mark.parametrize("dtype", DTYPES, ids=str)
    def test_gaussian_to_twin_gradcheck(
        self, dtype: torch.dtype, mu: float, sigma: float, device: str
    ) -> None:
        μ = torch.tensor(mu, dtype=dtype, device=device, requires_grad=True)
        σ = torch.tensor(sigma, dtype=dtype, device=device, requires_grad=True)

        lam = torch.exp(-0.5 * (μ / σ) ** 2).item()
        y_star = μ * min(1, lam / (1 - lam))
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
                atol, rtol, eps = 1e-6, 1e-6, 1e-6
            case _:
                raise ValueError(f"Unsupported dtype: {dtype}")

        gradcheck(gaussian_to_twin, (y_narrow, μ, σ), atol=atol, rtol=rtol, eps=eps)

    @pytest.mark.parametrize("device", DEVICES, ids=str)
    @pytest.mark.parametrize("sigma", [0.1, 0.5, 1, 2, 10], ids=lambda x: f"s={x}")
    @pytest.mark.parametrize("mu", [0.1, 0.5, 1, 2, 10], ids=lambda x: f"mu={x}")
    @pytest.mark.parametrize("dtype", DTYPES, ids=str)
    def test_reversible(
        self, dtype: torch.dtype, mu: float, sigma: float, device: str
    ) -> None:
        μ = torch.tensor(mu, dtype=dtype, device=device)
        σ = torch.tensor(sigma, dtype=dtype, device=device)
        y = torch.linspace(
            self.X_MIN,
            self.X_MAX,
            steps=self.N,
            dtype=dtype,
            device=device,
            requires_grad=True,
        )
        x_inv = gaussian_to_twin(y, μ, σ)
        y_inv = twin_to_gaussian(x_inv, μ, σ)
        z = y_inv.sum()
        z.backward()
        assert ((y_inv - y) / y).abs().mean() <= 1e-4, "Mean relative error too large"
        assert ((y_inv - y) / y).abs().max() <= 1e-2, "Max relative error too large"
        assert y.grad is not None
        assert torch.allclose(y.grad, torch.ones_like(y), atol=1e-3)
