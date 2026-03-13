r"""Implementation of the optimal transport based activation function."""
# mypy: disable-error-code="no-untyped-def"

__all__ = [
    "MAXITER",
    "_TwinToGaussian",
    "_GaussianToTwin",
    "_GaussianToBimodal",
    "_BimodalToGaussian",
    "_GaussianToMixture",
    "_MixtureToGaussian",
    # functional interfaces
    "gaussian_to_twin",
    "twin_to_gaussian",
    "gaussian_to_bimodal",
    "bimodal_to_gaussian",
    "gaussian_to_mixture",
    "mixture_to_gaussian",
]


import math
from typing import Final

import torch
from torch import Tensor
from torch.autograd import Function
from torch.special import log_ndtr

from linodenet_special.fallbacks.ndtri_exp import ndtri_exp

_SQRT_2: Final[float] = math.sqrt(2)
r"""CONST: √2, used for scaling the erfinv output."""
_LOG_HALF: Final[float] = math.log(0.5)
r"""CONST: log(0.5) is used in the tail handling of the erfinv computation."""
_LOG_2PI: Final[float] = math.log(math.tau)
MAXITER: int = 10
r"""CONFIG: maximum number of iterations for Newton's method in InvPsi."""


class _TwinToGaussian(Function):
    r"""Optimal Transport from mixture ½N(-μ, σ²) + ½N(μ, σ²) to N(0, 1)."""

    @staticmethod
    def forward(ctx, x: Tensor, /, mu: Tensor, sigma: Tensor) -> Tensor:
        s = sigma * _SQRT_2
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
    def backward(ctx, *outer: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        (g,) = outer
        a, b, z, mask = ctx.saved_tensors
        finfo = torch.finfo(z.dtype)
        TINY = finfo.tiny

        phi1 = torch.exp(z**2 - a**2)
        phi2 = torch.exp(z**2 - b**2)

        # compute the exact derivatives
        d_x_exact = 0.5 * (phi1 + phi2)
        d_mu_exact = 0.5 * (phi1 - phi2)
        d_sigma_exact = _SQRT_2 * (z - 0.5 * (a * phi1 + b * phi2))

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


class _GaussianToTwin(Function):
    r"""Optimal Transport from $N(0, 1)$ to symmetric mixture $½N(-μ, σ²) + ½N(μ, σ²)$."""

    @staticmethod
    def forward(ctx, y: Tensor, /, mu: Tensor, sigma: Tensor) -> Tensor:
        r"""Solve y = Ψ(x, μ, σ) for x using Newton's method.

        Note: ∂Ψ/∂x = \exp(-½μ²/σ²) at x=0. This is the minimum slope of Ψ
        So:   ∂Ψ⁻¹/∂y = 1 / (∂Ψ/∂x) ≈ \exp(½μ²/σ²) at y=0.

        How to make good initial guess:
            1. approximate Ψ⁻¹(y, μ, σ) ≈ hard_bend(y, λ=\exp(μ²/σ²), c=μ)
            2. invert hard_bend to get initial guess for x.
        """
        s = sigma * _SQRT_2
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
        lam = torch.exp(-0.5 * (mu / sigma) ** 2)
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
        d_sigma_exact = _SQRT_2 * (z - 0.5 * (a * phi1 + b * phi2))

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
    def backward(ctx, *outer: Tensor) -> tuple[Tensor, Tensor, Tensor]:
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


class _MixtureToGaussian(Function):
    r"""Optimal Transport from mixture $p = ∑ₖωₖN(μₖ,σₖ²)$ to $q = N(0,1)$.

    If $F_p$ and $F_q$ are the CDFs of $p$ and $q$, then the
    optimal transport map is given by

    .. math:: y = F_q⁻¹(F_p(x))

    Letting Φ be the CDF of $N(0,1)$, and letting $pₖ = N(μₖ,σₖ²)$ be the k-th component
    of $p$, then $F_p = ∑ₖωₖΦ((x-μₖ)/σₖ)$, and the optimal transport map is given by

    .. math:: y = Φ⁻¹( ∑ₖ ωₖ⋅Φ((x-μₖ)/σₖ) )

    To increase numerical stability, we compute the log of the CDFs and use the log-sum-exp trick:

    .. math:: y =
        \begin{cases}
            \NdtriExp\Bigl(\logsumexp(\log ωₖ + \log Φ(zₖ)) \Bigr) & \text{if } \log p < \log(½) \\
            -\NdtriExp\Bigl(\logsumexp_k(\log ωₖ + \log Φ(-zₖ))\Bigr) & \text{otherwise}
        \end{cases}

    where $zₖ = (x-μₖ)/σₖ$ and $\log p = \log ∑ₖ ωₖ Φ(zₖ)$. The second branch uses
    $\log(1-p) = \log ∑ₖ ωₖ Φ(-zₖ)$ to avoid loss of precision when $p$ is close to $1$.

    Regarding the derivative, we have, with $zₖ = (x-μₖ)/σₖ$

    .. math:: \dv{y}{x}  = ∑ₖ\frac{ωₖ}{σₖ} ℯ^{½ (y² - zₖ²)}
    .. math:: \dv{y}{ωₖ} = \sqrt{2π} ℯ^{½y²} Φ(zₖ)
    .. math:: \dv{y}{μₖ} = -\frac{ωₖ}{σₖ} ℯ^{½ (y² - zₖ²)}
    .. math:: \dv{y}{σₖ} = -\frac{ωₖ zₖ}{σₖ} ℯ^{½ (y² - zₖ²)}

    Proof:

        Via chain rule. The outer derivative is

        .. math:: \dv{Φ⁻¹(p)}{p} = \frac{1}{Φ'(Φ⁻¹(p))}

        and since $Φ'(x) = \frac{1}{\sqrt{2π}} ℯ^{-½x²}$, we have

        .. math::  \dv{Φ⁻¹}{p} = \sqrt{2π} ℯ^{½Φ⁻¹(p)²} =  \sqrt{2π} ℯ^{½y²}

        the inner derivative is, with some simplification

        .. math:: \dv{p}{x} &= \dv{x} ∑ₖ ωₖ⋅Φ((x-μₖ)/σₖ)   \\
                            &= \frac{1}{\sqrt{2π}}∑ₖ\frac{ωₖ}{σₖ}\exp{-½zₖ²}
        .. math:: \dv{p}{ωₖ} = Φ(zₖ)
        .. math:: \dv{p}{μₖ} = -\frac{1}{\sqrt{2π}}\frac{ωₖ}{σₖ}\exp{-½zₖ²}
        .. math:: \dv{p}{σₖ} = -\frac{zₖ}{\sqrt{2π}}\frac{ωₖ}{σₖ}\exp{-½zₖ²}

        So the total derivative is

        .. math:: \dv{y}{x} &= \dv{y}{p}\dv{p}{x} \\
                            &= \sqrt{2π} ℯ^{½y²} \frac{1}{\sqrt{2π}}∑ₖ\frac{ωₖ}{σₖ}\exp{-½zₖ²}
    """

    @staticmethod
    def forward(
        ctx, y: Tensor, /, weights: Tensor, means: Tensor, sigmas: Tensor
    ) -> Tensor:
        z = (y.unsqueeze(-1) - means) / sigmas
        assert z.shape[-1] == weights.shape[0] == means.shape[0] == sigmas.shape[0]

        log_w = torch.log(weights)
        log_p = torch.logsumexp(log_w + log_ndtr(z), dim=-1)
        log_q = torch.logsumexp(log_w + log_ndtr(-z), dim=-1)

        u = torch.where(log_p < _LOG_HALF, ndtri_exp(log_p), -ndtri_exp(log_q))
        assert u.isfinite().all()

        ctx.save_for_backward(z, u, log_w, torch.log(sigmas))
        return u

    @staticmethod
    def backward(ctx, *outer: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        (g,) = outer
        z, y, log_w, log_sigmas = ctx.saved_tensors

        y2 = y.square()

        # exp(½(y² - zₖ²)) = φ(zₖ) / φ(y)
        log_ratio = 0.5 * (y2.unsqueeze(-1) - z.square())
        # (ωₖ / σₖ) exp(½(y² - zₖ²)) appears in ∂y/∂x, ∂y/∂μₖ, and ∂y/∂σₖ.
        scaled_ratio = torch.exp(log_ratio + log_w - log_sigmas)

        # ∂y/∂x = ∑ₖ (ωₖ / σₖ) exp(½(y² - zₖ²))
        d_values = scaled_ratio.sum(dim=-1)
        # ∂y/∂μₖ = -(ωₖ / σₖ) exp(½(y² - zₖ²))
        d_means = -scaled_ratio
        # ∂y/∂σₖ = -(ωₖ zₖ / σₖ) exp(½(y² - zₖ²))
        d_sigmas = -z * scaled_ratio

        log_pdf_u = -0.5 * (_LOG_2PI + y2)
        d_weights = -torch.exp(log_ndtr(-z) - log_pdf_u.unsqueeze(-1))

        grad_values = g * d_values
        grad_weights = torch.einsum("..., ...k -> k", g, d_weights)
        grad_means = torch.einsum("..., ...k -> k", g, d_means)
        grad_sigmas = torch.einsum("..., ...k -> k", g, d_sigmas)

        # Project weight gradient onto the simplex tangent space.
        # ∆ⁿ = {x∈ℝⁿ⁺¹ : ∑ₖxₖ = 0, xₖ≥0}
        # 𝓣ₓ∆ⁿ = {v∈ℝⁿ⁺¹ : ∑ₖvₖ = 0} is the tangent space of the simplex at x.
        # proj(g) = g - ⟨𝟏∣g⟩ / ⟨𝟏∣𝟏⟩ * 𝟏 = g - mean(g) * 𝟏
        grad_weights = grad_weights - grad_weights.mean(dim=-1, keepdim=True)

        return grad_values, grad_weights, grad_means, grad_sigmas


class _GaussianToMixture(Function):
    r"""Optimal Transport from $N(0,1)$ to mixture $∑ₖωₖN(μₖ, σₖ²)$."""

    @staticmethod
    def forward(
        ctx, y: Tensor, /, weights: Tensor, means: Tensor, sigmas: Tensor
    ) -> Tensor:
        raise NotImplementedError

    @staticmethod
    def backward(ctx, *outer: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        raise NotImplementedError


class _BimodalToGaussian(Function):
    r"""Optimal Transport from mixture $ω₁N(μ₁,σ₁²) + ω₂N(μ₂,σ₂²)$ to $N(0,1)$."""

    @staticmethod
    def forward(
        ctx, y: Tensor, weights: Tensor, means: Tensor, sigmas: Tensor, /
    ) -> Tensor:
        raise NotImplementedError

    @staticmethod
    def backward(ctx, *outer: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        raise NotImplementedError


class _GaussianToBimodal(Function):
    r"""Optimal Transport from $N(0,1)$ to mixture $ω₁N(μ₁,σ₁²) + ω₂N(μ₂,σ₂²)$."""

    @staticmethod
    def forward(
        ctx, y: Tensor, /, weights: Tensor, means: Tensor, sigmas: Tensor
    ) -> Tensor:
        raise NotImplementedError

    @staticmethod
    def backward(ctx, *outer: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        raise NotImplementedError


def gaussian_to_twin(y: Tensor, /, mu: Tensor, sigma: Tensor) -> Tensor:
    r"""Optimal Transport from $N(0, 1)$ to symmetric mixture $½N(-μ, σ²) + ½N(μ, σ²)$."""
    return _GaussianToTwin.apply(y, mu, sigma)  # pyright: ignore[reportReturnType]


def twin_to_gaussian(x: Tensor, /, mu: Tensor, sigma: Tensor) -> Tensor:
    r"""Optimal Transport from mixture ½N(-μ, σ²) + ½N(μ, σ²) to N(0, 1)."""
    return _TwinToGaussian.apply(x, mu, sigma)  # pyright: ignore[reportReturnType]


def gaussian_to_bimodal(
    y: Tensor, /, weights: Tensor, means: Tensor, sigmas: Tensor
) -> Tensor:
    r"""Optimal Transport from $N(0,1)$ to mixture $ω₁N(μ₁,σ₁²) + ω₂N(μ₂,σ₂²)$."""
    return _GaussianToBimodal.apply(y, weights, means, sigmas)  # pyright: ignore[reportReturnType]


def bimodal_to_gaussian(
    x: Tensor, /, weights: Tensor, means: Tensor, sigmas: Tensor
) -> Tensor:
    r"""Optimal Transport from mixture $ω₁N(μ₁,σ₁²) + ω₂N(μ₂,σ₂²)$ to $N(0,1)$."""
    return _BimodalToGaussian.apply(x, weights, means, sigmas)  # pyright: ignore[reportReturnType]


def gaussian_to_mixture(
    y: Tensor, /, weights: Tensor, means: Tensor, sigmas: Tensor
) -> Tensor:
    r"""Optimal Transport from $N(0,1)$ to mixture $∑ₖωₖN(μₖ, σₖ²)$."""
    return _GaussianToMixture.apply(y, weights, means, sigmas)  # pyright: ignore[reportReturnType]


def mixture_to_gaussian(
    x: Tensor, /, weights: Tensor, means: Tensor, sigmas: Tensor
) -> Tensor:
    r"""Optimal Transport from mixture $∑ₖωₖN(μₖ,σₖ²)$ to $N(0,1)$."""
    return _MixtureToGaussian.apply(x, weights, means, sigmas)  # pyright: ignore[reportReturnType]
