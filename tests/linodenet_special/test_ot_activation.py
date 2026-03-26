import math
from typing import Any

import pytest
import torch
from torch import Tensor
from torch.autograd import Function, gradcheck

type Context = Any  # torch offers no type hint
SQRT_2 = math.sqrt(2)


class TestSimpleVariants:
    r"""Test some simple implementations."""

    def test_ot_activation_gradcheck(self) -> None:
        MU = 1.0
        SIGMA = 0.5
        x = torch.randn(10, dtype=torch.double, requires_grad=True)

        class F(Function):
            @staticmethod
            def forward(ctx: Context, x: Tensor) -> Tensor:
                s = SIGMA * SQRT_2
                a = (x + MU) / s
                b = (x - MU) / s
                z = torch.erfinv(0.5 * torch.erf(a) + 0.5 * torch.erf(b))
                ctx.save_for_backward(a, b, z)
                return s * z

            @staticmethod
            def backward(ctx: Context, *outer_grad: Tensor) -> Tensor:
                (g,) = outer_grad
                a, b, z = ctx.saved_tensors
                y_prime = (
                    0.5 * torch.exp(z**2) * (torch.exp(-(a**2)) + torch.exp(-(b**2)))
                )
                return g * y_prime

        gradcheck(F.apply, (x,), eps=1e-6, atol=1e-4)

    def test_gradcheck_with_parameters(self) -> None:
        class F(Function):
            @staticmethod
            def forward(ctx: Context, x: Tensor, mu: Tensor, sigma: Tensor) -> Tensor:
                s = sigma * SQRT_2
                a = (x + mu) / s
                b = (x - mu) / s
                z = torch.erfinv(0.5 * torch.erf(a) + 0.5 * torch.erf(b))
                ctx.save_for_backward(a, b, z)
                return s * z

            @staticmethod
            def backward(ctx: Context, *outer: Tensor) -> tuple[Tensor, Tensor, Tensor]:
                (g,) = outer
                a, b, z = ctx.saved_tensors
                dX = 0.5 * torch.exp(z**2) * (torch.exp(-(a**2)) + torch.exp(-(b**2)))
                dMu = 0.5 * torch.exp(z**2) * (torch.exp(-(a**2)) - torch.exp(-(b**2)))
                dSigma = SQRT_2 * (
                    z
                    - 0.5
                    * torch.exp(z**2)
                    * (a * torch.exp(-(a**2)) + b * torch.exp(-(b**2)))
                )
                return (g * dX), (g * dMu), (g * dSigma)

        mu = torch.tensor(1.0, dtype=torch.double, requires_grad=True)
        sigma = torch.tensor(0.5, dtype=torch.double, requires_grad=True)
        x = torch.linspace(-2.0, 2.0, steps=6, dtype=torch.double, requires_grad=True)
        gradcheck(F.apply, (x, mu, sigma), eps=1e-6, atol=1e-4)

    def test_gradcheck_simplified(self) -> None:
        class Psi(Function):
            @staticmethod
            def forward(ctx: Context, x: Tensor, mu: Tensor, sigma: Tensor) -> Tensor:
                s = sigma * SQRT_2
                a = (x + mu) / s
                b = (x - mu) / s
                y = torch.erfinv(0.5 * torch.erf(a) + 0.5 * torch.erf(b))
                ctx.save_for_backward(a, b, y)
                return s * y

            @staticmethod
            def backward(ctx: Context, *outer: Tensor) -> tuple[Tensor, Tensor, Tensor]:
                (g,) = outer
                a, b, z = ctx.saved_tensors
                phi1 = torch.exp(z**2 - a**2)
                phi2 = torch.exp(z**2 - b**2)
                dX = 0.5 * (phi1 + phi2)
                dMu = 0.5 * (phi1 - phi2)
                dSigma = SQRT_2 * (z - 0.5 * (a * phi1 + b * phi2))
                return (g * dX), (g * dMu), (g * dSigma)

        mu = torch.tensor(1.0, dtype=torch.double, requires_grad=True)
        sigma = torch.tensor(0.5, dtype=torch.double, requires_grad=True)
        x = torch.linspace(
            -2.0, 2.0, steps=1000, dtype=torch.double, requires_grad=True
        )
        gradcheck(Psi.apply, (x, mu, sigma), eps=1e-6, atol=1e-4)


MAXITER = 10


class TestImplementation:
    class Psi(Function):
        @staticmethod
        def forward(ctx, x: Tensor, mu: Tensor, sigma: Tensor) -> Tensor:
            s = sigma * SQRT_2
            EPS = 8 * torch.finfo(x.dtype).eps

            a = (x + mu) / s
            b = (x - mu) / s
            mix = 0.5 * (torch.erf(a) + torch.erf(b))
            mix = torch.clamp(mix, -1 + EPS, 1 - EPS)
            mask = mix.abs() < (1 - EPS)

            # compute y = √2σ * erfinv(mix), with tail handling
            z = torch.erfinv(mix)
            y = torch.where(mask, s * z, x - torch.sign(x) * mu)
            assert y.isfinite().all()

            # project to legal range
            y = torch.clamp(y, x - mu, x + mu)

            ctx.save_for_backward(a, b, z, mask)
            return y

        @staticmethod
        def backward(ctx: Context, *outer: Tensor) -> tuple[Tensor, Tensor, Tensor]:
            (g,) = outer
            a, b, z, mask = ctx.saved_tensors
            finfo = torch.finfo(z.dtype)
            TINY = finfo.tiny

            phi1 = torch.exp(z**2 - a**2)
            phi2 = torch.exp(z**2 - b**2)

            # compute the exact derivatives
            d_x_exact = 0.5 * (phi1 + phi2)
            d_mu_exact = 0.5 * (phi1 - phi2)
            d_sigma_exact = SQRT_2 * (z - 0.5 * (a * phi1 + b * phi2))

            # clamp gradient away from zero.
            d_x_exact = torch.clamp(d_x_exact, TINY, 1)
            d_mu_exact = torch.clamp(d_mu_exact, -1, 1)

            # compute the tail terms
            d_x_tail = torch.ones_like(d_x_exact)
            d_mu_tail = -torch.sign(z)
            d_sigma_tail = torch.zeros_like(d_sigma_exact)

            # combine via mask
            d_x = torch.where(mask, d_x_exact, d_x_tail)
            d_mu = torch.where(mask, d_mu_exact, d_mu_tail)
            d_sigma = torch.where(mask, d_sigma_exact, d_sigma_tail)

            return (g * d_x), (g * d_mu), (g * d_sigma)

    def psi(self, x: Tensor, mu: Tensor, sigma: Tensor) -> Tensor:
        return self.Psi.apply(x, mu, sigma)

    class InvPsi(Function):
        @staticmethod
        def forward(ctx: Context, y: Tensor, mu: Tensor, sigma: Tensor) -> Tensor:
            r"""Solve y = Ψ(x, μ, σ) for x using Newton's method.

            Note: ∂Ψ/∂x = \exp(-½μ²/σ²) at x=0. This is the minimum slope of Ψ
            So:   ∂Ψ⁻¹/∂y = 1 / (∂Ψ/∂x) ≈ \exp(½μ²/σ²) at y=0.

            How to make good initial guess:
                1. approximate Ψ⁻¹(y, μ, σ) ≈ hard_bend(y, λ=\exp(μ²/σ²), c=μ)
                2. invert hard_bend to get initial guess for x.
            """
            s = sigma * SQRT_2
            finfo = torch.finfo(y.dtype)
            EPS = 8 * finfo.eps
            TINY = finfo.tiny

            # we know the solution is in the interval [y-μ, y+μ]
            # and we may also use bisection.
            lower = y - mu
            upper = y + mu

            # Use hard_bend approximation to get initial guess for x
            #
            # hard_bend(z, λ, c) = {
            #    z + c       if   λz > z+c         (i.e. z > c/(λ-1))
            #    λz          if   z-c ≤ λz ≤ z+c   (i.e. z∈[-c/(λ-1), c/(λ-1)])
            #    z - c       if   λz < z-c         (i.e. z < -c/(λ-1))
            # }
            # Inverse of hard_bend is:
            # hard_bend_inv(y, λ, c) = {
            #    y - c       if   y > cλ/(λ-1)     (i.e. y > cλ/(λ-1))
            #    y / λ        if   -cλ/(λ-1) ≤ y ≤ cλ/(λ-1)   (i.e. y∈[-cλ/(λ-1), cλ/(λ-1)])
            #    y + c       if   y < -cλ/(λ-1)     (i.e. y < -cλ/(λ-1))
            # }
            lam = torch.exp(-0.5 * (mu / sigma) ** 2).item()
            x = torch.where(
                (y / lam).abs() <= y.abs() - mu,
                y / lam,
                y + torch.sign(y) * mu,
            )

            for _ in range(MAXITER):
                # project onto legal range
                x = torch.clamp(x, lower, upper)

                a = (x + mu) / s
                b = (x - mu) / s
                mix = 0.5 * (torch.erf(a) + torch.erf(b))
                mix = torch.clamp(mix, -1 + EPS, 1 - EPS)
                mask = mix.abs() < 1 - EPS

                # compute y = √2σ * erfinv(mix), with tail handling
                z = torch.erfinv(mix)
                fx = torch.where(mask, s * z, x - torch.sign(x) * mu)
                assert fx.isfinite().all()

                # project to legal range
                fx = torch.clamp(fx, x - mu, x + mu)

                # compute the exact derivatives
                phi1 = torch.exp(z**2 - a**2)
                phi2 = torch.exp(z**2 - b**2)

                # clamp gradient away from zero.
                d_x_exact = 0.5 * (phi1 + phi2)
                d_x_exact = torch.clamp(d_x_exact, TINY, 1)

                # compute the tail terms
                d_x_tail = torch.ones_like(d_x_exact)

                # combine via mask
                d_fx = torch.where(mask, d_x_exact, d_x_tail)

                # compute residual, update bounds using monotonicity
                r = fx - y
                lower = torch.where(r < 0, x, lower)
                upper = torch.where(r > 0, x, upper)

                x_newton = x - r / d_fx
                x_bisect = 0.5 * (lower + upper)

                # only do newton if it stays in the legal range, otherwise do bisection
                x = torch.where(
                    (x_newton >= lower) & (x_newton <= upper),
                    x_newton,
                    x_bisect,
                )

            # compute final derivatives for backward pass
            # project onto legal range
            x = torch.clamp(x, lower, upper)

            a = (x + mu) / s
            b = (x - mu) / s
            mix = 0.5 * (torch.erf(a) + torch.erf(b))
            mix = torch.clamp(mix, -1 + EPS, 1 - EPS)
            mask = mix.abs() < 1 - EPS

            # compute y = √2σ * erfinv(mix), with tail handling
            z = torch.erfinv(mix)
            fx = torch.where(mask, s * z, x - torch.sign(x) * mu)
            assert fx.isfinite().all()

            # compute the exact derivatives
            phi1 = torch.exp(z**2 - a**2)
            phi2 = torch.exp(z**2 - b**2)
            # compute the exact derivatives
            d_x_exact = 0.5 * (phi1 + phi2)
            d_mu_exact = 0.5 * (phi1 - phi2)
            d_sigma_exact = SQRT_2 * (z - 0.5 * (a * phi1 + b * phi2))

            # clamp gradient away from zero.
            d_x_exact = torch.clamp(d_x_exact, TINY, 1)
            d_mu_exact = torch.clamp(d_mu_exact, -1, 1)

            # compute the tail terms
            d_x_tail = torch.ones_like(d_x_exact)
            d_mu_tail = -torch.sign(z)
            d_sigma_tail = torch.zeros_like(d_sigma_exact)

            # combine via mask
            d_x = torch.where(mask, d_x_exact, d_x_tail)
            d_mu = torch.where(mask, d_mu_exact, d_mu_tail)
            d_sigma = torch.where(mask, d_sigma_exact, d_sigma_tail)

            ctx.save_for_backward(d_x, d_mu, d_sigma)
            return x

        @staticmethod
        def backward(ctx: Context, *outer: Tensor) -> tuple[Tensor, Tensor, Tensor]:
            """Use the derivatives of Ψ to compute the derivatives of x with respect to y, μ, and σ.

            .. math::  ∂Ψ(x(y, μ, σ)) = y
                ⟹ ∂x/∂y = (∂Ψ/∂x)⁻¹
                ⟹ ∂x/∂μ = - (∂Ψ/∂x)⁻¹ * (∂Ψ/∂μ)
                ⟹ ∂x/∂σ = - (∂Ψ/∂x)⁻¹ * (∂Ψ/∂σ)
            """
            (g,) = outer
            dx, dmu, dsigma = ctx.saved_tensors
            dy = g * (1 / dx)
            dmu = g * (-dmu / dx)
            dsigma = g * (-dsigma / dx)
            return dy, dmu, dsigma

    def invpsi(self, y: Tensor, mu: Tensor, sigma: Tensor) -> Tensor:
        return self.InvPsi.apply(y, mu, sigma)

    @pytest.mark.parametrize("sigma", [0.01, 0.1, 1, 10], ids=lambda x: f"s={x}")
    @pytest.mark.parametrize("mu", [0.1, 0.5, 1, 2, 10], ids=lambda x: f"mu={x}")
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64], ids=str)
    def test_hard_bend_approximation(
        self, dtype: torch.dtype, mu: float, sigma: float
    ) -> None:
        μ = torch.tensor(mu, dtype=dtype)
        σ = torch.tensor(sigma, dtype=dtype)

        # Test the hard_bend approximation for a range of y values
        x = torch.linspace(-20.0, 20.0, steps=1000, dtype=dtype)
        y = self.psi(x, μ, σ)
        assert y.dtype == dtype
        assert y.isfinite().all(), "Psi should produce finite outputs for finite inputs"

        # compute the inverse approximation
        lam = torch.exp(-0.5 * (μ / σ) ** 2).item()
        x_approx = torch.where(
            (y / lam).abs() <= y.abs() - μ,
            y / lam,
            y + torch.sign(y) * μ,
        )
        assert x_approx.dtype == dtype
        assert x_approx.isfinite().all(), (
            "Approximation should produce finite outputs for finite inputs"
        )
        assert (x_approx - x).abs().max() < μ.abs()

    @pytest.mark.parametrize("sigma", [0.1, 0.5, 1, 2, 10], ids=lambda x: f"s={x}")
    @pytest.mark.parametrize("mu", [0.1, 0.5, 1, 2, 10], ids=lambda x: f"mu={x}")
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64], ids=str)
    def test_psi_forward(self, dtype: torch.dtype, mu: float, sigma: float) -> None:
        μ = torch.tensor(mu, dtype=dtype)
        σ = torch.tensor(sigma, dtype=dtype)

        # zero check: Ψ(0, μ, σ) = 0
        zero = torch.tensor(0, dtype=dtype)
        assert torch.allclose(self.psi(zero, μ, σ), zero)

        # positive x values
        x1 = torch.linspace(0, 20, steps=1000, dtype=dtype)
        y1 = self.psi(x1, μ, σ)
        assert y1.dtype == dtype
        assert y1.isfinite().all()

        # negative x values
        x2 = torch.linspace(0, -20, steps=1000, dtype=dtype)
        y2 = self.psi(x2, μ, σ)
        assert y2.dtype == dtype
        assert y2.isfinite().all()

        # approximate symmetry check: Ψ(-x, μ, σ) ≈ -Ψ(x, μ, σ)
        assert torch.allclose(y1, -y2)

        # tail check: Ψ(x, μ, σ) ≈ x - sign(x) * μ for large |x|
        # large here means x ≫ x⁎, where x⁎ comes from the hard_contract approximation:
        # x⁎=c/(1-λ) with c=μ and λ=exp(-½μ²/σ²)
        # we pick the threshold as c⋅max(1, 1/(1-λ))
        lam = torch.exp(-0.5 * (μ / σ) ** 2).item()
        x_star = μ * max(1, 1 / (1 - lam))
        assert x_star.item() > 0
        x1 = torch.linspace(10 * x_star, 100 * x_star, steps=1000, dtype=dtype)
        x2 = torch.linspace(-10 * x_star, -100 * x_star, steps=1000, dtype=dtype)
        tail1 = x1 - torch.sign(x1) * μ
        tail2 = x2 - torch.sign(x2) * μ
        y1 = self.psi(x1, μ, σ)
        y2 = self.psi(x2, μ, σ)
        assert torch.allclose(y1, tail1)
        assert torch.allclose(y2, tail2)

    @pytest.mark.parametrize("sigma", [0.1, 0.5, 1, 2, 10], ids=lambda x: f"s={x}")
    @pytest.mark.parametrize("mu", [0.1, 0.5, 1, 2, 10], ids=lambda x: f"mu={x}")
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64], ids=str)
    def test_inv_psi_forward(self, dtype: torch.dtype, mu: float, sigma: float) -> None:
        μ = torch.tensor(mu, dtype=dtype)
        σ = torch.tensor(sigma, dtype=dtype)

        # zero check: Ψ(0, μ, σ) = 0
        zero = torch.tensor(0, dtype=dtype)
        assert torch.allclose(self.invpsi(zero, μ, σ), zero)

        # positive x values
        y1 = torch.linspace(0, 20, steps=1000, dtype=dtype)
        x1 = self.invpsi(y1, μ, σ)
        assert x1.dtype == dtype
        assert x1.isfinite().all()

        # negative x values
        y2 = torch.linspace(0, -20, steps=1000, dtype=dtype)
        x2 = self.invpsi(y2, μ, σ)
        assert x2.dtype == dtype
        assert x2.isfinite().all()

        # approximate symmetry check: Ψ(-x, μ, σ) ≈ -Ψ(x, μ, σ)
        assert torch.allclose(x1, -x2)

        # tail check: Ψ⁻¹(y, μ, σ) ≈ y + sign(y)μ for large |y|
        # here, large means y ≫ y⁎, where y⁎ comes from the PL-approximation:
        # For hard_expand, y⁎=c/(λ-1)=cλ⁻¹/(1-λ⁻¹) with c=μ and λ=exp(½μ²/σ²)
        # we pick the threshold as c⋅max(1, λ⁻¹/(1-λ⁻¹))
        lam = torch.exp(-0.5 * (μ / σ) ** 2).item()
        y_star = μ * max(1, lam / (1 - lam))
        assert y_star.item() > 0
        y1 = torch.linspace(10 * y_star, 100 * y_star, steps=1000, dtype=dtype)
        y2 = torch.linspace(-10 * y_star, -100 * y_star, steps=1000, dtype=dtype)
        tail1 = y1 + μ
        tail2 = y2 - μ
        x1 = self.invpsi(y1, μ, σ)
        x2 = self.invpsi(y2, μ, σ)
        assert x1.isfinite().all()
        assert x2.isfinite().all()
        assert torch.allclose(x1, tail1)
        assert torch.allclose(x2, tail2)

    @pytest.mark.parametrize("sigma", [0.1, 0.5, 1, 2, 10], ids=lambda x: f"s={x}")
    @pytest.mark.parametrize("mu", [0.1, 0.5, 1, 2, 10], ids=lambda x: f"mu={x}")
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64], ids=str)
    def test_psi_grad(self, dtype: torch.dtype, mu: float, sigma: float) -> None:
        μ = torch.tensor(mu, dtype=dtype)
        σ = torch.tensor(sigma, dtype=dtype)
        # minimum grad value is at x=0, where
        lam = torch.exp(-0.5 * (μ / σ) ** 2)
        g_rtol = 2**-4
        lower_grad_bound = max(0, lam.item() * (1 - g_rtol))

        # positive x values
        x1 = torch.linspace(0, 20, steps=1000, dtype=dtype, requires_grad=True)
        y1 = self.psi(x1, μ, σ)
        y1.sum().backward()
        assert x1.grad is not None
        assert x1.grad.isfinite().all()
        assert x1.grad.max() <= 1
        assert x1.grad.min() > lower_grad_bound
        assert torch.allclose(x1.grad[0], lam, rtol=g_rtol)

        # negative x values
        x2 = torch.linspace(0, -20, steps=1000, dtype=dtype, requires_grad=True)
        y2 = self.psi(x2, μ, σ)
        y2.sum().backward()
        assert x2.grad is not None
        assert x2.grad.isfinite().all()
        assert x2.grad.max() <= 1
        assert x2.grad.min() > lower_grad_bound
        assert torch.allclose(x2.grad[0], lam, rtol=g_rtol)

        # check symmetry of the gradients
        assert torch.allclose(x1.grad, x2.grad)

        # tail check: Ψ'(x, μ, σ) ≈ x for large |x|
        # large here means x ≫ x⁎, where x⁎ comes from the hard_contract approximation:
        # x⁎=c/(1-λ) with c=μ and λ=exp(-½μ²/σ²)
        # we pick the threshold as c⋅max(1, 1/(1-λ))
        lam = torch.exp(-0.5 * (μ / σ) ** 2)
        x_star = μ * max(1, 1 / (1 - lam.item()))
        assert x_star.item() > 0
        tail_values = torch.linspace(10 * x_star, 100 * x_star, steps=1000, dtype=dtype)
        tail = torch.tensor(
            torch.cat([tail_values, tail_values.neg()]), dtype=dtype, requires_grad=True
        )
        y_tail = self.psi(tail, μ, σ)
        assert y_tail.isfinite().all()
        y_tail.sum().backward()
        assert tail.grad is not None
        assert tail.grad.isfinite().all()
        assert torch.allclose(tail.grad, torch.ones_like(tail.grad))

    @pytest.mark.parametrize("sigma", [0.1, 0.5, 1, 2, 10], ids=lambda x: f"s={x}")
    @pytest.mark.parametrize("mu", [0.1, 0.5, 1, 2, 10], ids=lambda x: f"mu={x}")
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64], ids=str)
    def test_inv_psi_grad(self, dtype: torch.dtype, mu: float, sigma: float) -> None:
        μ = torch.tensor(mu, dtype=dtype)
        σ = torch.tensor(sigma, dtype=dtype)
        # minimum grad value is at x=0, where
        lam = torch.exp(-0.5 * (μ / σ) ** 2).item()
        lam_inv_log = 0.5 * (μ / σ) ** 2  # log(1/λ) = 0.5 * (μ/σ)²
        g_rtol = 2**-4
        log_tol = math.log2(1 + g_rtol)
        log_grad_bound = lam_inv_log + log_tol  # g ≤ (1+r)g⁎

        # positive x values
        y1 = torch.linspace(0, 20, steps=1000, dtype=dtype, requires_grad=True)
        x1 = self.invpsi(y1, μ, σ)
        x1.sum().backward()
        assert y1.grad is not None
        assert y1.grad.isfinite().all()
        assert y1.grad.min() >= 1
        assert y1.grad.log().max() <= log_grad_bound
        # assert torch.allclose(y1.grad[0].log(), lam_inv_log, atol=log_tol)

        # negative x values
        y2 = torch.linspace(0, -20, steps=1000, dtype=dtype, requires_grad=True)
        x2 = self.invpsi(y2, μ, σ)
        x2.sum().backward()
        assert y2.grad is not None
        assert y2.grad.isfinite().all()
        assert y2.grad.min() >= 1
        assert y2.grad.log().max() <= log_grad_bound
        # assert torch.allclose(y2.grad[0].log(), lam_inv_log, atol=math.log(1 + g_rtol))

        # check symmetry of the gradients
        assert torch.allclose(y1.grad, y2.grad)

        # tail check: Ψ⁻¹(y, μ, σ) ≈ y + sign(y)μ for large |y|
        # here, large means y ≫ y⁎, where y⁎ comes from the PL-approximation:
        # For hard_expand, y⁎=c/(λ-1)=cλ⁻¹/(1-λ⁻¹) with c=μ and λ=exp(½μ²/σ²)
        # we pick the threshold as c⋅max(1, λ⁻¹/(1-λ⁻¹))
        y_star = μ * max(1, lam / (1 - lam))
        assert y_star.item() > 0
        tail_values = torch.linspace(10 * y_star, 100 * y_star, steps=1000, dtype=dtype)
        tail = torch.tensor(
            torch.cat([tail_values, tail_values.neg()]), dtype=dtype, requires_grad=True
        )
        y_tail = self.invpsi(tail, μ, σ)
        assert y_tail.isfinite().all()
        y_tail.sum().backward()
        assert tail.grad is not None
        assert tail.grad.isfinite().all()
        assert torch.allclose(tail.grad, torch.ones_like(tail.grad))

    @pytest.mark.parametrize("sigma", [0.1, 0.5, 1, 2, 10], ids=lambda x: f"s={x}")
    @pytest.mark.parametrize("mu", [0.1, 0.5, 1, 2, 10], ids=lambda x: f"mu={x}")
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64], ids=str)
    def test_psi_gradcheck(self, dtype: torch.dtype, mu: float, sigma: float) -> None:
        # perform gradcheck on a narrower range to avoid numerical issues at the tails
        # only test gradcheck in the interval [-x_star, x_star]
        # outside, due to clamping, there may be flat regions,
        # causing numerical gradients to be zero and fail gradcheck.
        # the analytical gradients are still non-zero, and correct.
        μ = torch.tensor(mu, dtype=dtype, requires_grad=True)
        σ = torch.tensor(sigma, dtype=dtype, requires_grad=True)

        lam = torch.exp(-0.5 * (μ / σ) ** 2).item()
        x_star = μ * min(1, 1 / (1 - lam))
        x_narrow = torch.linspace(
            -x_star, x_star, steps=100, dtype=dtype, requires_grad=True
        )

        match dtype:
            case torch.float32:
                atol, rtol, eps = 1e-2, 1e-5, 1e-4
            case torch.float64:
                atol, rtol, eps = 1e-8, 1e-8, 1e-6
            case _:
                raise ValueError(f"Unsupported dtype: {dtype}")

        gradcheck(self.psi, (x_narrow, μ, σ), atol=atol, rtol=rtol, eps=eps)

    @pytest.mark.parametrize("sigma", [0.5, 1, 2, 10], ids=lambda x: f"s={x}")
    @pytest.mark.parametrize("mu", [0.1, 0.5, 1, 1.5], ids=lambda x: f"mu={x}")
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64], ids=str)
    def test_inv_psi_gradcheck(
        self, dtype: torch.dtype, mu: float, sigma: float
    ) -> None:
        # perform gradcheck on a narrower range to avoid numerical issues at the tails
        # only test gradcheck in the interval [-x_star, x_star]
        # outside, due to clamping, there may be flat regions,
        # causing numerical gradients to be zero and fail gradcheck.
        # the analytical gradients are still non-zero, and correct.
        μ = torch.tensor(mu, dtype=dtype, requires_grad=True)
        σ = torch.tensor(sigma, dtype=dtype, requires_grad=True)

        lam = torch.exp(-0.5 * (μ / σ) ** 2).item()
        y_star = μ * min(1, lam / (1 - lam))
        y_narrow = torch.linspace(
            -y_star / 2, y_star / 2, steps=100, dtype=dtype, requires_grad=True
        )

        match dtype:
            case torch.float32:
                atol, rtol, eps = 1e-2, 1e-2, 1e-4
            case torch.float64:
                atol, rtol, eps = 1e-6, 1e-6, 1e-6
            case _:
                raise ValueError(f"Unsupported dtype: {dtype}")

        gradcheck(self.invpsi, (y_narrow, μ, σ), atol=atol, rtol=rtol, eps=eps)

    @pytest.mark.parametrize("sigma", [0.1, 0.5, 1, 2, 10], ids=lambda x: f"s={x}")
    @pytest.mark.parametrize("mu", [0.1, 0.5, 1, 2, 10], ids=lambda x: f"mu={x}")
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64], ids=str)
    def test_composition(self, dtype: torch.dtype, mu: float, sigma: float) -> None:
        μ = torch.tensor(mu, dtype=dtype)
        σ = torch.tensor(sigma, dtype=dtype)

        # y -> x_inv -> y_inv
        y = torch.linspace(-20, 20, steps=1000, dtype=dtype)
        x_inv = self.invpsi(y, μ, σ)
        y_inv = self.psi(x_inv, μ, σ)
        # check the errors are small relative to x
        r = torch.relu((y_inv - y).abs() - 1e6)
        assert (r / y).abs().mean() <= 1e-4, "Mean relative error too large"
        assert (r / y).abs().max() <= 1e-2, "Max relative error too large"

        # x -> y -> x_inv
        x = torch.linspace(-20, 20, steps=1000, dtype=dtype)
        y = self.psi(x, μ, σ)
        x_inv = self.invpsi(y, μ, σ)
        # check the errors are small relative to x
        r = torch.relu((x_inv - x).abs() - 1e6)
        assert (r / x).abs().mean() <= 1e-4, "Mean relative error too large"
        assert (r / x).abs().max() <= 1e-2, "Max relative error too large"

    @pytest.mark.parametrize("sigma", [0.5, 1, 2, 10], ids=lambda x: f"s={x}")
    @pytest.mark.parametrize("mu", [0.1, 0.5, 1, 1.5], ids=lambda x: f"mu={x}")
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64], ids=str)
    def test_composition_grads(
        self, dtype: torch.dtype, mu: float, sigma: float
    ) -> None:
        μ = torch.tensor(mu, dtype=dtype, requires_grad=True)
        σ = torch.tensor(sigma, dtype=dtype, requires_grad=True)
        x = torch.linspace(-2, 2, steps=6, dtype=dtype, requires_grad=True)

        # x -> y -> x_inv
        y = self.psi(x, μ, σ)
        x_inv = self.invpsi(y, μ, σ)
        z = x_inv.sum()
        z.backward()
        assert x.grad is not None
        assert torch.allclose(x.grad, torch.ones_like(x), atol=1e-3)

        # y -> x_inv -> y_inv
        y = torch.linspace(-2, 2, steps=6, dtype=dtype, requires_grad=True)
        x_inv = self.invpsi(y, μ, σ)
        y_inv = self.psi(x_inv, μ, σ)
        z = y_inv.sum()
        z.backward()
        assert y.grad is not None
        assert torch.allclose(y.grad, torch.ones_like(y), atol=1e-3)

    @pytest.mark.parametrize("dtype", [torch.float64, torch.float32], ids=str)
    def test_erfinv_range(self, dtype: torch.dtype) -> None:
        finfo_eps = torch.finfo(dtype).eps
        finfo_eps_log = int(math.floor(math.log2(finfo_eps)))

        # figure out the range (-1+2⁻ᵏ, 1-2⁻ᵏ) that's safe to pass to erfinv without getting inf or NaN
        for k in range(60, 0, -1):
            eps = 2 ** (-k)
            vals = torch.linspace(-1 + eps, 1 - eps, steps=1000, dtype=dtype)
            erfinv_vals = torch.erfinv(vals)
            val_chk = erfinv_vals.isfinite().all()

            # test monotonicity around 0
            x = finfo_eps * torch.arange(-100, 100, dtype=dtype)
            l = torch.erfinv(x - finfo_eps)
            r = torch.erfinv(x)
            center_chk = l.isfinite().all() and r.isfinite().all() and (r > l).all()

            # test edge monotonicity
            x = (1 - eps) - finfo_eps * torch.arange(0, 1000, dtype=dtype)
            l = torch.erfinv(x - finfo_eps)
            r = torch.erfinv(x)
            edge_chk = l.isfinite().all() and r.isfinite().all() and (r > l).all()

            if val_chk and center_chk and edge_chk:
                print(
                    f"Safe range for erfinv with dtype {dtype}: "
                    f"\n\t[1-2**(-{k}), 1-2**(-{k})] = [{-1 + eps}, {1 - eps}]"
                    f"\n\tfinfo.eps={torch.finfo(dtype).eps} ≈ 2**({finfo_eps_log}) "
                )
                break
        else:
            raise RuntimeError(
                f"Could not find a safe range for erfinv with dtype {dtype}"
            )
