import math

import pytest
import torch
from torch.autograd import Function, gradcheck

SQRT_2 = math.sqrt(2)


def test_ot_activation_gradcheck():
    MU = 1.0
    SIGMA = 0.5
    x = torch.randn(10, dtype=torch.double, requires_grad=True)

    class F(Function):
        @staticmethod
        def forward(ctx, x):
            s = SIGMA * SQRT_2
            a = (x + MU) / s
            b = (x - MU) / s
            z = torch.erfinv(0.5 * torch.erf(a) + 0.5 * torch.erf(b))
            ctx.save_for_backward(a, b, z)
            return s * z

        @staticmethod
        def backward(ctx, grad_output):
            a, b, z = ctx.saved_tensors
            y_prime = 0.5 * torch.exp(z**2) * (torch.exp(-(a**2)) + torch.exp(-(b**2)))
            return grad_output * y_prime

    gradcheck(F.apply, (x,), eps=1e-6, atol=1e-4)


def test_gradcheck_with_parameters():
    class F(Function):
        @staticmethod
        def forward(ctx, x, mu, sigma):
            s = sigma * SQRT_2
            a = (x + mu) / s
            b = (x - mu) / s
            z = torch.erfinv(0.5 * torch.erf(a) + 0.5 * torch.erf(b))
            ctx.save_for_backward(a, b, z)
            return s * z

        @staticmethod
        def backward(ctx, outer_grad):
            a, b, z = ctx.saved_tensors
            dX = 0.5 * torch.exp(z**2) * (torch.exp(-(a**2)) + torch.exp(-(b**2)))
            dMu = 0.5 * torch.exp(z**2) * (torch.exp(-(a**2)) - torch.exp(-(b**2)))
            dSigma = SQRT_2 * (
                z
                - 0.5
                * torch.exp(z**2)
                * (a * torch.exp(-(a**2)) + b * torch.exp(-(b**2)))
            )
            return (outer_grad * dX), (outer_grad * dMu), (outer_grad * dSigma)

    mu = torch.tensor(1.0, dtype=torch.double, requires_grad=True)
    sigma = torch.tensor(0.5, dtype=torch.double, requires_grad=True)
    x = torch.linspace(-2.0, 2.0, steps=6, dtype=torch.double, requires_grad=True)
    gradcheck(F.apply, (x, mu, sigma), eps=1e-6, atol=1e-4)


def test_gradcheck_simplified():
    class Psi(Function):
        @staticmethod
        def forward(ctx, x, mu, sigma):
            s = sigma * SQRT_2
            a = (x + mu) / s
            b = (x - mu) / s
            y = torch.erfinv(0.5 * torch.erf(a) + 0.5 * torch.erf(b))
            ctx.save_for_backward(a, b, y)
            return s * y

        @staticmethod
        def backward(ctx, outer_grad):
            a, b, z = ctx.saved_tensors
            phi1 = torch.exp(z**2 - a**2)
            phi2 = torch.exp(z**2 - b**2)
            dX = 0.5 * (phi1 + phi2)
            dMu = 0.5 * (phi1 - phi2)
            dSigma = SQRT_2 * (z - 0.5 * (a * phi1 + b * phi2))
            return (outer_grad * dX), (outer_grad * dMu), (outer_grad * dSigma)

    mu = torch.tensor(1.0, dtype=torch.double, requires_grad=True)
    sigma = torch.tensor(0.5, dtype=torch.double, requires_grad=True)
    x = torch.linspace(-2.0, 2.0, steps=1000, dtype=torch.double, requires_grad=True)
    gradcheck(Psi.apply, (x, mu, sigma), eps=1e-6, atol=1e-4)


MAXITER = 10


class TestPsiInverse:
    class Psi(Function):
        @staticmethod
        def forward(ctx, x, mu, sigma):
            s = sigma * SQRT_2
            eps = torch.finfo(x.dtype).eps

            a = (x + mu) / s
            b = (x - mu) / s
            mix = 0.5 * (torch.erf(a) + torch.erf(b))
            mix = torch.clamp(mix, -1, 1)

            # compute y = √2σ * erfinv(mix), with tail handling
            mask = mix.abs() < 1 - 8 * eps
            y = s * torch.erfinv(mix)
            y = torch.where(mask, y, x - torch.sign(x) * mu)
            assert y.isfinite().all()

            # project to legal range
            y = torch.clamp(y, x - mu, x + mu)

            ctx.save_for_backward(a, b, y / s, mask)
            return y

        @staticmethod
        def backward(ctx, outer_grad):
            a, b, z, mask = ctx.saved_tensors
            finfo = torch.finfo(z.dtype)

            phi1 = torch.exp(z**2 - a**2)
            phi2 = torch.exp(z**2 - b**2)

            # compute the exact derivatives
            d_x_exact = 0.5 * (phi1 + phi2)
            d_mu_exact = 0.5 * (phi1 - phi2)
            d_sigma_exact = SQRT_2 * (z - 0.5 * (a * phi1 + b * phi2))

            # clamp gradient away from zero.
            d_x_exact = torch.clamp(d_x_exact, finfo.tiny, 1)
            d_mu_exact = torch.clamp(d_mu_exact, -1, 1)

            # compute the tail terms
            d_x_tail = torch.ones_like(d_x_exact)
            d_mu_tail = -torch.sign(z)
            d_sigma_tail = torch.zeros_like(d_sigma_exact)

            # combine via mask
            d_x = torch.where(mask, d_x_exact, d_x_tail)
            d_mu = torch.where(mask, d_mu_exact, d_mu_tail)
            d_sigma = torch.where(mask, d_sigma_exact, d_sigma_tail)

            return (
                (outer_grad * d_x),
                (outer_grad * d_mu),
                (outer_grad * d_sigma),
            )

    class InvPsi(Function):
        @staticmethod
        def forward(ctx, y, mu, sigma):
            r"""Solve y = Ψ(x, μ, σ) for x using Newton's method.

            Note: ∂Ψ/∂x = \exp(-½μ²/σ²) at x=0. This is the minimum slope of Ψ
            So:   ∂Ψ⁻¹/∂y = 1 / (∂Ψ/∂x) ≈ \exp(½μ²/σ²) at y=0.

            How to make good initial guess:
                1. approximate Ψ⁻¹(y, μ, σ) ≈ hard_bend(y, λ=\exp(μ²/σ²), c=μ)
                2. invert hard_bend to get initial guess for x.
            """
            s = sigma * SQRT_2
            finfo = torch.finfo(y.dtype)

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
            lam = torch.exp(-0.5 * (mu / sigma) ** 2)
            x = torch.where(
                (y / lam).abs() <= y.abs() - mu,
                y / lam,
                y + torch.sign(y) * mu,
            )

            for _ in range(MAXITER):
                # project onto legal range
                x = torch.clamp(x, y - mu, y + mu)

                a = (x + mu) / s
                b = (x - mu) / s
                mix = 0.5 * (torch.erf(a) + torch.erf(b))
                mix = torch.clamp(mix, -1, 1)

                # compute y = √2σ * erfinv(mix), with tail handling
                mask = mix.abs() < 1 - 8 * finfo.eps
                fx = s * torch.erfinv(mix)
                fx = torch.where(mask, fx, x - torch.sign(x) * mu)
                assert fx.isfinite().all()

                # project to legal range
                fx = torch.clamp(fx, x - mu, x + mu)

                # compute the exact derivatives
                z = fx / s
                phi1 = torch.exp(z**2 - a**2)
                phi2 = torch.exp(z**2 - b**2)
                d_x_exact = 0.5 * (phi1 + phi2)

                # clamp gradient away from zero.
                d_x_exact = torch.clamp(d_x_exact, finfo.tiny, 1)

                # compute the tail terms
                d_x_tail = torch.ones_like(d_x_exact)

                # combine via mask
                d_x = torch.where(mask, d_x_exact, d_x_tail)

                # perform the newton update
                x = x - (fx - y) / d_x

            # compute final derivatives for backward pass
            # project onto legal range
            x = torch.clamp(x, y - mu, y + mu)

            a = (x + mu) / s
            b = (x - mu) / s
            mix = 0.5 * (torch.erf(a) + torch.erf(b))
            mix = torch.clamp(mix, -1, 1)

            # compute y = √2σ * erfinv(mix), with tail handling
            mask = mix.abs() < 1 - 8 * finfo.eps
            fx = s * torch.erfinv(mix)
            fx = torch.where(mask, fx, x - torch.sign(x) * mu)
            assert fx.isfinite().all()

            # project to legal range
            fx = torch.clamp(fx, x - mu, x + mu)

            # compute the exact derivatives
            z = fx / s
            phi1 = torch.exp(z**2 - a**2)
            phi2 = torch.exp(z**2 - b**2)
            # compute the exact derivatives
            d_x_exact = 0.5 * (phi1 + phi2)
            d_mu_exact = 0.5 * (phi1 - phi2)
            d_sigma_exact = SQRT_2 * (z - 0.5 * (a * phi1 + b * phi2))

            # clamp gradient away from zero.
            d_x_exact = torch.clamp(d_x_exact, finfo.tiny, 1)
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
        def backward(ctx, outer_grad):
            """Use the derivatives of Ψ to compute the derivatives of x with respect to y, μ, and σ.

            .. math::  ∂Ψ(x(y, μ, σ)) = y
                ⟹ ∂x/∂y = (∂Ψ/∂x)⁻¹
                ⟹ ∂x/∂μ = - (∂Ψ/∂x)⁻¹ * (∂Ψ/∂μ)
                ⟹ ∂x/∂σ = - (∂Ψ/∂x)⁻¹ * (∂Ψ/∂σ)
            """
            dx, dmu, dsigma = ctx.saved_tensors
            dy = outer_grad * (1 / dx)
            dmu = outer_grad * (-dmu / dx)
            dsigma = outer_grad * (-dsigma / dx)
            return dy, dmu, dsigma

    @pytest.mark.parametrize("sigma", [0.01, 0.1, 1, 10], ids=lambda x: f"s={x}")
    @pytest.mark.parametrize("mu", [0.1, 0.5, 1, 2, 10], ids=lambda x: f"mu={x}")
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64], ids=str)
    def test_hard_bend_approximation(self, dtype, mu, sigma):
        μ = torch.tensor(mu, dtype=dtype)
        σ = torch.tensor(sigma, dtype=dtype)

        # Test the hard_bend approximation for a range of y values
        x = torch.linspace(-20.0, 20.0, steps=1000, dtype=dtype)
        y = self.Psi.apply(x, μ, σ)
        assert y.dtype == dtype
        assert y.isfinite().all(), "Psi should produce finite outputs for finite inputs"

        # compute the inverse approximation
        lam = torch.exp(-0.5 * (μ / σ) ** 2)
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
    def test_psi_forward(self, dtype, mu, sigma):
        μ = torch.tensor(mu, dtype=dtype)
        σ = torch.tensor(sigma, dtype=dtype)

        # zero check: Ψ(0, μ, σ) = 0
        zero = torch.tensor(0, dtype=dtype)
        assert torch.allclose(self.Psi.apply(zero, μ, σ), zero)

        # positive x values
        x1 = torch.linspace(0, 20, steps=1000, dtype=dtype)
        y1 = self.Psi.apply(x1, μ, σ)
        assert y1.dtype == dtype
        assert y1.isfinite().all()

        # negative x values
        x2 = torch.linspace(0, -20, steps=1000, dtype=dtype)
        y2 = self.Psi.apply(x2, μ, σ)
        assert y2.dtype == dtype
        assert y2.isfinite().all()

        # approximate symmetry check: Ψ(-x, μ, σ) ≈ -Ψ(x, μ, σ)
        assert torch.allclose(y1, -y2)

        # tail check: Ψ(x, μ, σ) ≈ x - sign(x) * μ for large |x|
        # large here means x ≫ x⁎, where x⁎ comes from the hard_contract approximation:
        # x⁎=c/(1-λ) with c=μ and λ=exp(-½μ²/σ²)
        # we pick the threshold as c⋅max(1, 1/(1-λ))
        lam = torch.exp(-0.5 * (μ / σ) ** 2)
        x_star = μ * max(1, 1 / (1 - lam))
        assert x_star.item() > 0
        x1 = torch.linspace(10 * x_star, 100 * x_star, steps=1000, dtype=dtype)
        x2 = torch.linspace(-10 * x_star, -100 * x_star, steps=1000, dtype=dtype)
        tail1 = x1 - torch.sign(x1) * μ
        tail2 = x2 - torch.sign(x2) * μ
        y1 = self.Psi.apply(x1, μ, σ)
        y2 = self.Psi.apply(x2, μ, σ)
        assert torch.allclose(y1, tail1)
        assert torch.allclose(y2, tail2)

    @pytest.mark.parametrize("sigma", [0.1, 0.5, 1, 2, 10], ids=lambda x: f"s={x}")
    @pytest.mark.parametrize("mu", [0.1, 0.5, 1, 2, 10], ids=lambda x: f"mu={x}")
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64], ids=str)
    def test_inv_psi_forward(self, dtype, mu, sigma):
        μ = torch.tensor(mu, dtype=dtype)
        σ = torch.tensor(sigma, dtype=dtype)

        # zero check: Ψ(0, μ, σ) = 0
        zero = torch.tensor(0, dtype=dtype)
        assert torch.allclose(self.InvPsi.apply(zero, μ, σ), zero)

        # positive x values
        y1 = torch.linspace(0, 20, steps=1000, dtype=dtype)
        x1 = self.InvPsi.apply(y1, μ, σ)
        assert x1.dtype == dtype
        assert x1.isfinite().all()

        # negative x values
        y2 = torch.linspace(0, -20, steps=1000, dtype=dtype)
        x2 = self.InvPsi.apply(y2, μ, σ)
        assert x2.dtype == dtype
        assert x2.isfinite().all()

        # approximate symmetry check: Ψ(-x, μ, σ) ≈ -Ψ(x, μ, σ)
        assert torch.allclose(x1, -x2)

        # tail check: Ψ⁻¹(y, μ, σ) ≈ y + sign(y)μ for large |y|
        # here, large means y ≫ y⁎, where y⁎ comes from the PL-approximation:
        # For hard_expand, y⁎=c/(λ-1)=cλ⁻¹/(1-λ⁻¹) with c=μ and λ=exp(½μ²/σ²)
        # we pick the threshold as c⋅max(1, λ⁻¹/(1-λ⁻¹))
        lam_inv = torch.exp(-0.5 * (μ / σ) ** 2)
        y_star = μ * max(1, lam_inv / (1 - lam_inv))
        assert y_star.item() > 0
        y1 = torch.linspace(10 * y_star, 100 * y_star, steps=1000, dtype=dtype)
        y2 = torch.linspace(-10 * y_star, -100 * y_star, steps=1000, dtype=dtype)
        tail1 = y1 + μ
        tail2 = y2 - μ
        x1 = self.InvPsi.apply(y1, μ, σ)
        x2 = self.InvPsi.apply(y2, μ, σ)
        assert x1.isfinite().all()
        assert x2.isfinite().all()
        assert torch.allclose(x1, tail1)
        assert torch.allclose(x2, tail2)

    @pytest.mark.parametrize("sigma", [0.1, 0.5, 1, 2, 10], ids=lambda x: f"s={x}")
    @pytest.mark.parametrize("mu", [0.1, 0.5, 1, 2, 10], ids=lambda x: f"mu={x}")
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64], ids=str)
    def test_psi_grad(self, dtype, mu, sigma):
        μ = torch.tensor(mu, dtype=dtype)
        σ = torch.tensor(sigma, dtype=dtype)
        # minimum grad value is at x=0, where
        lam = torch.exp(-0.5 * (μ / σ) ** 2)
        g_rtol = 2**-6
        g_atol = 2**-6
        lower_grad_bound = max(0, lam * (1 - g_rtol) - g_atol)

        # positive x values
        x1 = torch.linspace(0, 20, steps=1000, dtype=dtype, requires_grad=True)
        y1 = self.Psi.apply(x1, μ, σ)
        y1.sum().backward()
        assert x1.grad is not None, "Gradient should be computed for x"
        assert x1.grad.isfinite().all(), "Gradient should be finite for all inputs"
        assert (x1.grad <= 1).all()
        assert (x1.grad > lower_grad_bound).all()
        assert torch.allclose(x1.grad[0], lam)

        # negative x values
        x2 = torch.linspace(0, -20, steps=1000, dtype=dtype, requires_grad=True)
        y2 = self.Psi.apply(x2, μ, σ)
        y2.sum().backward()
        assert x2.grad is not None, "Gradient should be computed for x"
        assert x2.grad.isfinite().all(), "Gradient should be finite for all inputs"
        assert (x2.grad <= 1).all()
        assert (x2.grad > lower_grad_bound).all()
        assert torch.allclose(x2.grad[0], lam)

        # check symmetry of the gradients
        assert torch.allclose(x1.grad, x2.grad)

        # tail check: Ψ'(x, μ, σ) ≈ x for large |x|
        # large here means x ≫ x⁎, where x⁎ comes from the hard_contract approximation:
        # x⁎=c/(1-λ) with c=μ and λ=exp(-½μ²/σ²)
        # we pick the threshold as c⋅max(1, 1/(1-λ))
        lam = torch.exp(-0.5 * (μ / σ) ** 2)
        x_star = μ * max(1, 1 / (1 - lam))
        assert x_star.item() > 0
        tail_values = torch.linspace(10 * x_star, 100 * x_star, steps=1000, dtype=dtype)
        tail = torch.tensor(
            torch.cat([tail_values, tail_values.neg()]), dtype=dtype, requires_grad=True
        )
        y_tail = self.Psi.apply(tail, μ, σ)
        assert y_tail.isfinite().all()
        y_tail.sum().backward()
        assert tail.grad is not None
        assert tail.grad.isfinite().all()
        assert torch.allclose(tail.grad, torch.ones_like(tail.grad))

    @pytest.mark.parametrize("sigma", [0.1, 0.5, 1, 2, 10], ids=lambda x: f"s={x}")
    @pytest.mark.parametrize("mu", [0.1, 0.5, 1, 2, 10], ids=lambda x: f"mu={x}")
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64], ids=str)
    def test_inv_psi_grad(self, dtype, mu, sigma):
        μ = torch.tensor(mu, dtype=dtype)
        σ = torch.tensor(sigma, dtype=dtype)
        # minimum grad value is at x=0, where
        lam = torch.exp(-0.5 * (μ / σ) ** 2)
        lam_inv = torch.exp(0.5 * (μ / σ) ** 2)
        g_rtol = 2**-4
        upper_grad_bound = max(0, lam_inv * (1 + g_rtol))

        # positive x values
        y1 = torch.linspace(0, 20, steps=1000, dtype=dtype, requires_grad=True)
        x1 = self.InvPsi.apply(y1, μ, σ)
        x1.sum().backward()
        assert y1.grad is not None, "Gradient should be computed for x"
        assert y1.grad.isfinite().all(), "Gradient should be finite for all inputs"
        assert (y1.grad >= 1).all()
        assert (y1.grad <= upper_grad_bound).all()
        assert torch.allclose(y1.grad[0], lam_inv)

        # negative x values
        y2 = torch.linspace(0, -20, steps=1000, dtype=dtype, requires_grad=True)
        x2 = self.InvPsi.apply(y2, μ, σ)
        x2.sum().backward()
        assert y2.grad is not None, "Gradient should be computed for x"
        assert y2.grad.isfinite().all(), "Gradient should be finite for all inputs"
        assert (y1.grad >= 1).all()
        assert (y1.grad <= upper_grad_bound).all()
        assert torch.allclose(y2.grad[0], lam_inv)

        # check symmetry of the gradients
        assert torch.allclose(y1.grad, y2.grad)

        # tail check: Ψ'(x, μ, σ) ≈ x for large |x|
        # large here means x ≫ x⁎, where x⁎ comes from the hard_contract approximation:
        # x⁎=c/(1-λ) with c=μ and λ=exp(-½μ²/σ²)
        # we pick the threshold as c⋅max(1, 1/(1-λ))
        x_star = μ * max(1, 1 / (1 - lam))
        assert x_star.item() > 0
        tail_values = torch.linspace(10 * x_star, 100 * x_star, steps=1000, dtype=dtype)
        tail = torch.tensor(
            torch.cat([tail_values, tail_values.neg()]), dtype=dtype, requires_grad=True
        )
        y_tail = self.InvPsi.apply(tail, μ, σ)
        assert y_tail.isfinite().all()
        y_tail.sum().backward()
        assert tail.grad is not None
        assert tail.grad.isfinite().all()
        assert torch.allclose(tail.grad, torch.ones_like(tail.grad))

    @pytest.mark.parametrize("sigma", [0.1, 0.5, 1, 2, 10], ids=lambda x: f"s={x}")
    @pytest.mark.parametrize("mu", [0.1, 0.5, 1, 2, 10], ids=lambda x: f"mu={x}")
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64], ids=str)
    def test_psi_gradcheck(self, dtype, mu, sigma):
        # perform gradcheck on a narrower range to avoid numerical issues at the tails
        μ = torch.tensor(mu, dtype=dtype, requires_grad=True)
        σ = torch.tensor(sigma, dtype=dtype, requires_grad=True)
        x_narrow = torch.linspace(-2, 2, steps=100, dtype=dtype, requires_grad=True)
        gradcheck(self.Psi.apply, (x_narrow, μ, σ))

    @pytest.mark.parametrize("sigma", [0.1, 0.5, 1, 2, 10], ids=lambda x: f"s={x}")
    @pytest.mark.parametrize("mu", [0.1, 0.5, 1, 2, 10], ids=lambda x: f"mu={x}")
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64], ids=str)
    def test_inv_psi_gradcheck(self, dtype, mu, sigma):
        # perform gradcheck on a narrower range to avoid numerical issues at the tails
        μ = torch.tensor(mu, dtype=dtype, requires_grad=True)
        σ = torch.tensor(sigma, dtype=dtype, requires_grad=True)
        y_narrow = torch.linspace(-2, 2, steps=100, dtype=dtype, requires_grad=True)
        gradcheck(self.InvPsi.apply, (y_narrow, μ, σ))

    @pytest.mark.parametrize("sigma", [0.1, 0.5, 1, 2, 10], ids=lambda x: f"s={x}")
    @pytest.mark.parametrize("mu", [0.1, 0.5, 1, 2, 10], ids=lambda x: f"mu={x}")
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64], ids=str)
    def test_composition(self, dtype, mu, sigma):
        μ = torch.tensor(mu, dtype=dtype)
        σ = torch.tensor(sigma, dtype=dtype)

        # y -> x_inv -> y_inv
        y = torch.linspace(-20.0, 20.0, steps=1000, dtype=dtype)
        x_inv = self.InvPsi.apply(y, μ, σ)
        y_inv = self.Psi.apply(x_inv, μ, σ)
        # check the errors are small relative to x
        r = torch.relu((y_inv - y).abs() - 1e6)
        assert (r / y).abs().mean() <= 1e-4, "Mean relative error too large"
        assert (r / y).abs().max() <= 1e-2, "Max relative error too large"

        # x -> y -> x_inv
        x = torch.linspace(-20.0, 20.0, steps=1000, dtype=dtype)
        y = self.Psi.apply(x, μ, σ)
        x_inv = self.InvPsi.apply(y, μ, σ)
        # check the errors are small relative to x
        r = torch.relu((x_inv - x).abs() - 1e6)
        assert (r / x).abs().mean() <= 1e-4, "Mean relative error too large"
        assert (r / x).abs().max() <= 1e-2, "Max relative error too large"

    @pytest.mark.parametrize("sigma", [0.1, 0.5, 1, 2, 10], ids=lambda x: f"s={x}")
    @pytest.mark.parametrize("mu", [0.1, 0.5, 1, 2, 10], ids=lambda x: f"mu={x}")
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64], ids=str)
    def test_composition_grads(self, dtype, mu, sigma):
        μ = torch.tensor(mu, dtype=dtype, requires_grad=True)
        σ = torch.tensor(sigma, dtype=dtype, requires_grad=True)
        x = torch.linspace(-2.0, 2.0, steps=6, dtype=dtype, requires_grad=True)
        y = self.Psi.apply(x, μ, σ)
        x_inv = self.InvPsi.apply(y, μ, σ)
        z = x_inv.sum()
        z.backward()
        assert torch.allclose(x.grad, torch.ones_like(x), atol=1e-4)

    def test_inv_psi_edge(self):
        dtype = torch.float64
        μ = torch.tensor(0.5, dtype=torch.float64)
        σ = torch.tensor(0.1, dtype=torch.float64)

        # tail check: Ψ⁻¹(y, μ, σ) ≈ y + sign(y)μ for large |y|
        # here, large means y ≫ y⁎, where y⁎ comes from the PL-approximation:
        # For hard_expand, y⁎=c/(λ-1)=cλ⁻¹/(1-λ⁻¹) with c=μ and λ=exp(½μ²/σ²)
        # we pick the threshold as c⋅max(1, λ⁻¹/(1-λ⁻¹))
        lam_inv = torch.exp(-0.5 * (μ / σ) ** 2)
        y_star = μ * max(1, lam_inv / (1 - lam_inv))
        assert y_star.item() > 0
        y1 = torch.linspace(20 * y_star, 100 * y_star, steps=10, dtype=dtype)
        y2 = torch.linspace(-20 * y_star, -100 * y_star, steps=10, dtype=dtype)
        tail1 = y1 + μ
        tail2 = y2 - μ
        x1 = self.InvPsi.apply(y1, μ, σ)
        x2 = self.InvPsi.apply(y2, μ, σ)
        assert x1.isfinite().all()
        assert x2.isfinite().all()
        assert torch.allclose(x1, tail1)
        assert torch.allclose(x2, tail2)

    def test_psi_edge(self):
        dtype = torch.float64
        μ = torch.tensor(0.5, dtype=torch.float64)
        σ = torch.tensor(0.1, dtype=torch.float64)

        # tail check: Ψ(x, μ, σ) ≈ x - sign(x) * μ for large |x|
        # large here means x ≫ x⁎, where x⁎ comes from the hard_contract approximation:
        # x⁎=c/(1-λ) with c=μ and λ=exp(-½μ²/σ²)
        # we pick the threshold as c⋅max(1, 1/(1-λ))
        lam = torch.exp(-0.5 * (μ / σ) ** 2)
        x_star = μ * max(1, 1 / (1 - lam))
        assert x_star.item() > 0
        x1 = torch.linspace(10 * x_star, 100 * x_star, steps=1000, dtype=dtype)
        x2 = torch.linspace(-10 * x_star, -100 * x_star, steps=1000, dtype=dtype)
        tail1 = x1 - torch.sign(x1) * μ
        tail2 = x2 - torch.sign(x2) * μ
        y1 = self.Psi.apply(x1, μ, σ)
        y2 = self.Psi.apply(x2, μ, σ)
        assert torch.allclose(y1, tail1)
        assert torch.allclose(y2, tail2)
