r"""Implementation of the optimal transport based activation function."""
# mypy: disable-error-code="no-untyped-def"

__all__ = [
    # functional interfaces
    "gaussian_to_bimodal",
    "bimodal_to_gaussian",
    "bimodal_to_gaussian_value_and_jac",
    "gaussian_to_mixture",
    "mixture_to_gaussian",
]

import math
from typing import Final

import torch
from torch import Tensor
from torch.autograd import Function
from torch.special import log_ndtr

from .hard_bend import hard_bend
from .ndtri_exp import ndtri_exp


def _bimodal_to_gaussian_value(x: Tensor, mu: Tensor, sigma: Tensor, /) -> Tensor:
    r"""Evaluate the bimodal-to-Gaussian transport and cache the normalized coordinates."""
    LOG_HALF: Final[float] = -0.6931471805599453  # log(½)

    m = mu.abs()
    z_plus = (x + m) / sigma
    z_minus = (x - m) / sigma

    log_p = torch.logaddexp(LOG_HALF + log_ndtr(z_plus), LOG_HALF + log_ndtr(z_minus))
    log_q = torch.logaddexp(LOG_HALF + log_ndtr(-z_plus), LOG_HALF + log_ndtr(-z_minus))
    y = torch.where(log_p < LOG_HALF, ndtri_exp(log_p), -ndtri_exp(log_q))

    # apply analytical bounds
    return torch.clamp(y, z_minus, z_plus)


def _bimodal_to_gaussian_value_and_jac(
    x: Tensor, mu: Tensor, sigma: Tensor, /
) -> tuple[Tensor, Tensor]:
    r"""Evaluate the bimodal transport and its $x$-derivative in one pass."""
    LOG_HALF: Final[float] = -0.6931471805599453  # log(½)

    m = mu.abs()
    z_plus = (x + m) / sigma
    z_minus = (x - m) / sigma

    log_p = torch.logaddexp(LOG_HALF + log_ndtr(z_plus), LOG_HALF + log_ndtr(z_minus))
    log_q = torch.logaddexp(LOG_HALF + log_ndtr(-z_plus), LOG_HALF + log_ndtr(-z_minus))
    fx = torch.where(log_p < LOG_HALF, ndtri_exp(log_p), -ndtri_exp(log_q))
    fx = torch.clamp(fx, z_minus, z_plus)

    y2 = fx.square()
    log_sigma = sigma.log()
    log_phi_plus = 0.5 * (y2 - z_plus.square()) - log_sigma + LOG_HALF
    log_phi_minus = 0.5 * (y2 - z_minus.square()) - log_sigma + LOG_HALF
    lower_bound = torch.exp(-0.5 * (m / sigma) ** 2) / sigma
    upper_bound = 1 / sigma
    d_fx = torch.logaddexp(log_phi_plus, log_phi_minus).exp()
    d_fx = torch.clamp(d_fx, lower_bound, upper_bound)

    return fx, d_fx


def _bimodal_to_gaussian_derivatives(
    x: Tensor, mu: Tensor, sigma: Tensor, y: Tensor, /
) -> tuple[Tensor, Tensor, Tensor]:
    r"""Compute stable partial derivatives for the bimodal-to-Gaussian transport.

    Returns:
        ∂y/∂x:  ½σ⁻¹(E₊ + E₋})
        ∂y/∂μ:  ½σ⁻¹(E₊ - E₋})
        ∂y/∂σ: -½σ⁻¹(z₊E₊ + z₋E₋)
    """
    LOG_HALF: Final[float] = -0.6931471805599453  # log(½)

    m = mu.abs()
    z_plus = (x + m) / sigma
    z_minus = (x - m) / sigma
    mu_sign = torch.sign(mu)
    y2 = y.square()
    log_sigma = sigma.log()
    log_phi_plus = 0.5 * (y2 - z_plus.square()) - log_sigma + LOG_HALF
    log_phi_minus = 0.5 * (y2 - z_minus.square()) - log_sigma + LOG_HALF

    d_x_exact = torch.logaddexp(log_phi_plus, log_phi_minus).exp()
    hi = torch.maximum(log_phi_plus, log_phi_minus)
    lo = torch.minimum(log_phi_plus, log_phi_minus)
    d_m_exact = torch.sign(log_phi_plus - log_phi_minus) * torch.exp(
        hi + torch.log1p(-torch.exp(lo - hi))
    )
    d_sigma_exact = -(0.5 * (z_plus + z_minus) * d_x_exact + (m / sigma) * d_m_exact)

    lower_bound = torch.exp(-0.5 * (m / sigma) ** 2) / sigma
    upper_bound = 1 / sigma
    d_x = torch.clamp(d_x_exact, lower_bound, upper_bound)
    d_mu = mu_sign * torch.clamp(d_m_exact, -upper_bound, upper_bound)
    return d_x, d_mu, d_sigma_exact


def _bimodal_to_gaussian_derivatives2(
    x: Tensor, mu: Tensor, sigma: Tensor, y: Tensor, /
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    r"""Compute first and second derivatives for the bimodal-to-Gaussian transport.

    Returns:
        ∂y/∂x:  ½σ⁻¹(E₊ + E₋})
        ∂y/∂μ:  ½σ⁻¹(E₊ - E₋})
        ∂y/∂σ: -½σ⁻¹(z₊E₊ + z₋E₋)
        ∂g/∂x: -½σ⁻²(z₊E₊ + z₋E₋)   + yg(∂y/∂x)
        ∂g/∂μ: -½σ⁻²(z₊E₊ - z₋E₋)   + yg(∂y/∂μ)
        ∂g/∂σ: +½σ⁻²(z₊²E₊ + z₋²E₋) + yg(∂y/∂σ) -g/σ
    """
    LOG_HALF: Final[float] = -0.6931471805599453  # log(½)

    m = mu.abs()
    z_plus = (x + m) / sigma
    z_minus = (x - m) / sigma
    mu_sign = torch.sign(mu)
    y2 = y.square()
    log_sigma = sigma.log()
    log_phi_plus = 0.5 * (y2 - z_plus.square()) - log_sigma + LOG_HALF
    log_phi_minus = 0.5 * (y2 - z_minus.square()) - log_sigma + LOG_HALF

    d_x_exact = torch.logaddexp(log_phi_plus, log_phi_minus).exp()
    hi = torch.maximum(log_phi_plus, log_phi_minus)
    lo = torch.minimum(log_phi_plus, log_phi_minus)
    d_m_exact = torch.sign(log_phi_plus - log_phi_minus) * torch.exp(
        hi + torch.log1p(-torch.exp(lo - hi))
    )
    d_sigma_exact = -(0.5 * (z_plus + z_minus) * d_x_exact + (m / sigma) * d_m_exact)

    lower_bound = torch.exp(-0.5 * (m / sigma) ** 2) / sigma
    upper_bound = 1 / sigma
    d_x = torch.clamp(d_x_exact, lower_bound, upper_bound)
    d_mu = mu_sign * torch.clamp(d_m_exact, -upper_bound, upper_bound)

    phi_plus = log_phi_plus.exp()
    phi_minus = log_phi_minus.exp()
    z_term_sum = (z_plus * phi_plus + z_minus * phi_minus) / sigma
    z_term_diff = (z_plus * phi_plus - z_minus * phi_minus) / sigma
    z2_term_sum = (z_plus.square() * phi_plus + z_minus.square() * phi_minus) / sigma

    d2_x = y * d_x.square() - z_term_sum
    d2_mu = mu_sign * (y * d_x * d_m_exact - z_term_diff)
    d2_sigma = -d_x / sigma + y * d_x * d_sigma_exact + z2_term_sum
    return d_x, d_mu, d_sigma_exact, d2_x, d2_mu, d2_sigma


def _gaussian_to_bimodal_guess(x: Tensor, mu: Tensor, sigma: Tensor, /) -> Tensor:
    r"""Approximate $Ψ⁻¹(x, μ, σ)$ by the matching `hard_bend` inverse.

    Here $λ = Ψ'(0, μ, σ) = σ⁻¹ℯ^{-½(μ/σ)²}$ is the slope at the origin.

    Using

    .. math::  y = hard\_bend(x, 1/λ, μ, σ) \iff x = hard\_bend(y, λ, μ, 1/σ),

    we obtain a cheap initial guess for the safeguarded Newton solve.
    """
    λ = torch.exp(-0.5 * (mu / sigma) ** 2) / sigma
    return hard_bend(x, λ, mu, 1 / sigma)


def _mixture_to_gaussian_value(
    x: Tensor, weights: Tensor, mus: Tensor, sigmas: Tensor, /
) -> tuple[Tensor, Tensor]:
    r"""Evaluate the mixture-to-Gaussian transport and cache normalized coordinates."""
    LOG_HALF: Final[float] = -0.6931471805599453  # log(½)

    z = (x.unsqueeze(-1) - mus) / sigmas
    log_w = torch.log(weights)
    log_p = torch.logsumexp(log_w + log_ndtr(z), dim=-1)
    log_q = torch.logsumexp(log_w + log_ndtr(-z), dim=-1)

    y = torch.where(log_p < LOG_HALF, ndtri_exp(log_p), -ndtri_exp(log_q))
    y = torch.clamp(y, z.min(dim=-1).values, z.max(dim=-1).values)
    return y, z


def _mixture_to_gaussian_value_and_jac(
    x: Tensor, weights: Tensor, mus: Tensor, sigmas: Tensor, /
) -> tuple[Tensor, Tensor]:
    r"""Evaluate the mixture transport and its $x$-derivative in one pass."""
    LOG_HALF: Final[float] = -0.6931471805599453  # log(½)

    z = (x.unsqueeze(-1) - mus) / sigmas
    log_w = torch.log(weights)
    log_p = torch.logsumexp(log_w + log_ndtr(z), dim=-1)
    log_q = torch.logsumexp(log_w + log_ndtr(-z), dim=-1)

    fx = torch.where(log_p < LOG_HALF, ndtri_exp(log_p), -ndtri_exp(log_q))
    fx = torch.clamp(fx, z.min(dim=-1).values, z.max(dim=-1).values)

    log_ratio = 0.5 * (fx.square().unsqueeze(-1) - z.square())
    d_fx = torch.exp(log_ratio + log_w - torch.log(sigmas)).sum(dim=-1)
    return fx, d_fx


def _mixture_to_gaussian_derivatives(
    z: Tensor, weights: Tensor, sigmas: Tensor, y: Tensor, /
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    r"""Compute stable partial derivatives for the mixture-to-Gaussian transport."""
    LOG_2PI: Final[float] = 1.8378770664093453  # log(2π)

    y2 = y.square()
    # exp(½(y² - zₖ²)) = φ(zₖ) / φ(y)
    log_ratio = 0.5 * (y2.unsqueeze(-1) - z.square())
    log_w = torch.log(weights)
    log_sigmas = torch.log(sigmas)
    # (ωₖ / σₖ) exp(½(y² - zₖ²)) appears in ∂y/∂x, ∂y/∂μₖ, and ∂y/∂σₖ.
    scaled_ratio = torch.exp(log_ratio + log_w - log_sigmas)

    # ∂y/∂x = ∑ₖ (ωₖ / σₖ) exp(½(y² - zₖ²))
    d_x = scaled_ratio.sum(dim=-1)
    # ∂y/∂μₖ = -(ωₖ / σₖ) exp(½(y² - zₖ²))
    d_mus = -scaled_ratio
    # ∂y/∂σₖ = -(ωₖ zₖ / σₖ) exp(½(y² - zₖ²))
    d_sigmas = -z * scaled_ratio

    # ∂y/∂ωₖ = √(2π) ℯ^{½y²}⋅(Φ(zₖ) - (1/n)∑ⱼΦ(zⱼ))
    # Project weight gradient onto the simplex tangent space.
    # ∆ⁿ = {x∈ℝⁿ⁺¹ : ∑ₖxₖ = 0, xₖ≥0}
    # 𝓣ₓ∆ⁿ = {v∈ℝⁿ⁺¹ : ∑ₖvₖ = 0} is the tangent space of the simplex at x.
    # proj(g) = g - ⟨𝟏∣g⟩ / ⟨𝟏∣𝟏⟩ * 𝟏 = g - mean(g) * 𝟏
    log_pdf_u = (0.5 * (LOG_2PI + y2)).unsqueeze(-1)
    log_phi = log_ndtr(z)  # log Φ(zₖ)
    log_phi_tangent = (  # log(Φ(zₖ) / (1/n)∑ⱼΦ(zⱼ))
        log_phi
        - log_phi.logsumexp(dim=-1, keepdim=True)
        - math.log(log_phi.shape[-1])  # emulates log_mean_exp
    )
    d_weights = torch.logaddexp(log_pdf_u, log_phi_tangent).exp()

    return d_x, d_weights, d_mus, d_sigmas


class _BimodalToGaussianImpl(Function):
    r"""Optimal Transport from mixture $p = ½N(-μ, σ²) + ½N(μ, σ²)$ to $q = N(0, 1)$.

    If $F_p$ and $F_q$ are the CDFs of $p$ and $q$, then the optimal transport map is given by

    .. math:: y = F_q⁻¹(Fₚ(x))

    Letting Φ be the CDF of $N(0,1)$, then we have

    .. math:: y = Φ⁻¹\Bigl( ½Φ((x+μ)/σ) + ½Φ((x-μ)/σ) \Bigr)
                = √2 \erf⁻¹\Bigl(½\erf((x+μ)/√2σ) + ½\erf((x-μ)/√2σ) \Bigr)

    Unlike the general mixture case, the two components share the mean $±μ$ and scale $σ$.

    Using the shorthands

    .. math::
        z₊ &= \frac{x+μ}{σ}     &   z₋ &= \frac{x-μ}{σ} \\
        E₊ &= ℯ^{½(y²-z₊²)}     &   E₋ &= ℯ^{½(y²-z₋²)}

    The first order derivatives can be written as:

    .. math::
        ∂y/∂x &=  ½σ⁻¹(E₊ + E₋})    \\
        ∂y/∂μ &=  ½σ⁻¹(E₊ - E₋})    \\
        ∂y/∂σ &= -½σ⁻¹(z₊E₊ + z₋E₋)

    And the derivatives of $g(x) = ∂y/∂x$ are

    .. math::
        ∂g/∂x &= -½σ⁻²(z₊E₊ + z₋E₋)   + yg(∂y/∂x)      \\
        ∂g/∂μ &= -½σ⁻²(z₊E₊ - z₋E₋)   + yg(∂y/∂μ)      \\
        ∂g/∂σ &= +½σ⁻²(z₊²E₊ + z₋²E₋) + yg(∂y/∂σ) -g/σ

    Proof:

        Let $u = ½Φ(z₊) + ½Φ(z₋)$, then, by the chain rule,

        .. math:: \dv{y}{u} = \frac{1}{Φ'(Φ⁻¹(u))} = \sqrt{2π} ℯ^{½y²}

        Using $Φ'(z) = \frac{1}{\sqrt{2π}}ℯ^{-½z²}$ and the coupling of the parameters,

        .. math::
            \dv{u}{x} &= ½(√2πσ)⁻¹(ℯ^{-½z₊²} + ℯ^{-½z₋²}) \\
            \dv{u}{μ} &= ½(√2πσ)⁻¹(ℯ^{-½z₊²} - ℯ^{-½z₋²}) \\
            \dv{u}{σ} &= -½(√2πσ)⁻¹(z₊ℯ^{-½z₊²} + z₋ℯ^{-½z₋²})

        Combining these terms yields the formulas above.
    """

    @staticmethod
    @torch.no_grad()
    def forward(ctx, x: Tensor, mu: Tensor, sigma: Tensor, /) -> Tensor:
        y = _bimodal_to_gaussian_value(x, mu, sigma)
        ctx.save_for_backward(x, mu, sigma, y)
        return y

    @staticmethod
    def backward(ctx, *outer: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        (g,) = outer
        x, mu, sigma, y = ctx.saved_tensors
        d_x, d_mu, d_sigma = _bimodal_to_gaussian_derivatives(x, mu, sigma, y)
        return (g * d_x), (g * d_mu), (g * d_sigma)


class _BimodalToGaussianValueAndJacImpl(Function):
    r"""Return the bimodal-to-Gaussian transport and its $x$-derivative."""

    @staticmethod
    @torch.no_grad()
    def forward(ctx, x: Tensor, mu: Tensor, sigma: Tensor, /) -> tuple[Tensor, Tensor]:
        y, d_x = _bimodal_to_gaussian_value_and_jac(x, mu, sigma)
        ctx.save_for_backward(x, mu, sigma, y, d_x)
        return y, d_x

    @staticmethod
    def backward(ctx, *outer: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        grad_y, grad_dy = outer
        x, mu, sigma, y, d_x = ctx.saved_tensors
        m = mu.abs()
        z_plus = (x + m) / sigma
        z_minus = (x - m) / sigma
        d_x, d_mu, d_sigma = _bimodal_to_gaussian_derivatives(x, mu, sigma, y)

        grad_x = grad_y * d_x
        grad_mu = grad_y * d_mu
        grad_sigma = grad_y * d_sigma

        if not grad_dy.any():
            return grad_x, grad_mu, grad_sigma

        LOG_HALF: Final[float] = -0.6931471805599453  # log(½)

        mu_sign = torch.sign(mu)
        y2 = y.square()
        log_sigma = sigma.log()
        log_phi_plus = 0.5 * (y2 - z_plus.square()) - log_sigma + LOG_HALF
        log_phi_minus = 0.5 * (y2 - z_minus.square()) - log_sigma + LOG_HALF
        phi_plus = log_phi_plus.exp()
        phi_minus = log_phi_minus.exp()
        d_m_exact = phi_plus - phi_minus
        z_term_sum = (z_plus * phi_plus + z_minus * phi_minus) / sigma
        z_term_diff = (z_plus * phi_plus - z_minus * phi_minus) / sigma
        z2_term_sum = (
            z_plus.square() * phi_plus + z_minus.square() * phi_minus
        ) / sigma

        d2_x = y * d_x.square() - z_term_sum
        d2_mu = mu_sign * (y * d_x * d_m_exact - z_term_diff)
        d2_sigma = -d_x / sigma + y * d_x * d_sigma + z2_term_sum

        grad_x = grad_x + grad_dy * d2_x
        grad_mu = grad_mu + grad_dy * d2_mu
        grad_sigma = grad_sigma + grad_dy * d2_sigma
        return grad_x, grad_mu, grad_sigma


class _GaussianToBimodalImpl(Function):
    r"""Optimal Transport from $N(0, 1)$ to symmetric mixture $½N(-μ, σ²) + ½N(μ, σ²)$."""

    @staticmethod
    @torch.no_grad()
    def forward(ctx, y: Tensor, mu: Tensor, sigma: Tensor, /) -> Tensor:
        r"""Solve $y = T(x, μ, σ)$ for $x$ using Newton's method.

        Here $T$ is the transport from the symmetric bimodal mixture to $N(0,1)$.
        Since $T'(0) = σ⁻¹ℯ^{-½|μ/σ|²}$, the inverse slope at the origin is

        .. math:: (T⁻¹)'(0) = σℯ^{½|μ/σ|²}.

        The transport only depends on $|μ|$. The tails satisfy
        $T(x, μ, σ) ≈ σ⁻¹(x-\sign(x)|μ|)$, so
        $T⁻¹(y, μ, σ) ≈ σy + \sign(y)|μ|$.
        """
        MAXITER: Final[int] = 10

        m = mu.abs()
        lower = sigma * y - m
        upper = sigma * y + m
        x = _gaussian_to_bimodal_guess(y, mu, sigma)

        for _ in range(MAXITER):
            x = torch.clamp(x, lower, upper)
            fx, d_fx = _bimodal_to_gaussian_value_and_jac(x, mu, sigma)
            r = fx - y
            lower = torch.where(r < 0, x, lower)
            upper = torch.where(r > 0, x, upper)
            x_newton = x - r / d_fx
            x_bisect = 0.5 * (lower + upper)
            x = torch.where(
                (x_newton >= lower) & (x_newton <= upper),
                x_newton,
                x_bisect,
            )

        x = torch.clamp(x, lower, upper)
        fx = _bimodal_to_gaussian_value(x, mu, sigma)

        ctx.save_for_backward(x, mu, sigma, fx)
        return x

    @staticmethod
    def backward(ctx, *outer: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        r"""Use the derivatives of $T$ to differentiate the inverse map.

        .. math::  ∂T(x(y, μ, σ), μ, σ) = y

        Hence

        .. math::
            ∂x/∂y &= (∂T/∂x)⁻¹ \\
            ∂x/∂μ &= -(∂T/∂x)⁻¹ ∂T/∂μ, \\
            ∂x/∂σ &= -(∂T/∂x)⁻¹ ∂T/∂σ.
        """
        (g,) = outer
        x, mu, sigma, fx = ctx.saved_tensors
        d_x, d_mu, d_sigma = _bimodal_to_gaussian_derivatives(x, mu, sigma, fx)
        dx_inv = d_x.reciprocal()

        d_y = dx_inv
        d_mu = -d_mu * dx_inv
        d_sigma = -d_sigma * dx_inv

        # clamp to legal range
        lower_bound = sigma
        upper_bound = sigma * torch.exp(0.5 * (mu / sigma) ** 2)
        d_y = d_y.clamp(lower_bound, upper_bound)
        d_mu = d_mu.clamp(-upper_bound, upper_bound)

        return (g * d_y), (g * d_mu), (g * d_sigma)


class _MixtureToGaussian(Function):
    r"""Optimal transport from $∑ₖωₖN(μₖ,σₖ²)$ to $N(0,1)$.

    If $Fₚ$ is the CDF of the mixture and $Φ$ is the standard normal CDF, then

    .. math:: y = Φ⁻¹(Fₚ(x)) = Φ⁻¹\Bigl(∑ₖ ωₖ Φ((x-μₖ)/σₖ)\Bigr)

    Numerically, we evaluate the mixture CDF in log space and switch between the
    lower-tail and upper-tail representations to avoid cancellation near $0$ and $1$.

    Using the shorthands

    .. math:: zₖ &= (x-μₖ)/σₖ  &  Eₖ &= ℯ^{½(y²-zₖ²)}

    then the first order derivatives are

    .. math::
        ∂y/∂x  &= ∑ₖ (ωₖ/σₖ) Eₖ    \\
        ∂y/∂ωₖ &= √(2π)ℯ^{½y²}(Φ(zₖ) - (1/n)∑ⱼΦ(zⱼ))    \\
        ∂y/∂μₖ &= -(ωₖ/σₖ) Eₖ     \\
        ∂y/∂σₖ &= -(ωₖ zₖ/σₖ) Eₖ

    Note that in the case of ∂y/∂ωₖ, we include the projection on the tangent space of ∆ⁿ⁻¹.

    And the derivatives of $g(x) = ∂y/∂x$ are

    .. math::
        ∂g/∂x  &= y⋅g⋅(∂y/∂x)  - ∑ₖ(ωₖ/σₖ²)zₖEₖ         \\
        ∂g/∂ωₖ &= y⋅g⋅(∂y/∂ωₖ) + Eₖ/σₖ - (1/n)∑ⱼEⱼ/σⱼ   \\
        ∂g/∂μₖ &= y⋅g⋅(∂y/∂μₖ) + (ωₖ/σₖ²)zₖEₖ           \\
        ∂g/∂σₖ &= y⋅g⋅(∂y/∂σₖ) + (ωₖ/σₖ²)(zₖ²-1)Eₖ
    """

    @staticmethod
    @torch.no_grad()
    def forward(
        ctx, y: Tensor, weights: Tensor, mus: Tensor, sigmas: Tensor, /
    ) -> Tensor:
        assert weights.shape[0] == mus.shape[0] == sigmas.shape[0]
        u, z = _mixture_to_gaussian_value(y, weights, mus, sigmas)
        ctx.save_for_backward(z, u, weights, sigmas)
        return u

    @staticmethod
    def backward(ctx, *outer: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        r"""Differentiate the explicit mixture-to-Gaussian transport map."""
        (g,) = outer
        z, y, weights, sigmas = ctx.saved_tensors
        d_values, d_weights, d_mus, d_sigmas = _mixture_to_gaussian_derivatives(
            z, weights, sigmas, y
        )

        grad_values = g * d_values
        grad_weights = torch.einsum("..., ...k -> k", g, d_weights)
        grad_mus = torch.einsum("..., ...k -> k", g, d_mus)
        grad_sigmas = torch.einsum("..., ...k -> k", g, d_sigmas)

        return grad_values, grad_weights, grad_mus, grad_sigmas


class _GaussianToMixture(Function):
    r"""Optimal Transport from $N(0,1)$ to mixture $∑ₖωₖN(μₖ, σₖ²)$.

    This transform cannot be expressed in "closed form", so we compute the
    inverse through newton-iteration / bisection.
    """

    @staticmethod
    @torch.no_grad()
    def forward(
        ctx, y: Tensor, weights: Tensor, mus: Tensor, sigmas: Tensor, /
    ) -> Tensor:
        r"""Solve $T(x, ω, μ, σ)=y$ by safeguarded Newton iteration."""
        MAXITER: Final[int] = 10

        assert weights.shape[0] == mus.shape[0] == sigmas.shape[0]

        # Each component alone would invert y to xₖ = μₖ + σₖy. The mixture inverse
        # must lie between the smallest and largest of these affine tail candidates,
        # so we use their pointwise min/max as a safe bracket and their weighted mean
        # as a cheap initial guess for the safeguarded Newton iteration.
        lines = mus + sigmas * y.unsqueeze(-1)
        lower = lines.min(dim=-1).values
        upper = lines.max(dim=-1).values
        x = torch.einsum("k, ...k -> ...", weights, lines)

        for _ in range(MAXITER):
            x = torch.clamp(x, lower, upper)
            fy, d_fy = _mixture_to_gaussian_value_and_jac(x, weights, mus, sigmas)
            r = fy - y
            # Since T is monotone, the sign of the residual tells us which side of
            # the bracket still contains the inverse solution.
            lower = torch.where(r < 0, x, lower)
            upper = torch.where(r > 0, x, upper)
            x_newton = x - r / d_fy
            x_bisect = 0.5 * (lower + upper)
            x = torch.where(
                (x_newton >= lower) & (x_newton <= upper),
                x_newton,
                x_bisect,
            )

        x = torch.clamp(x, lower, upper)
        fy, z = _mixture_to_gaussian_value(x, weights, mus, sigmas)

        ctx.save_for_backward(z, fy, weights, mus, sigmas)
        return x

    @staticmethod
    def backward(ctx, *outer: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        r"""Use the derivatives of $T$ to compute the derivatives of $T⁻¹$.

        Writing $T(x, ω, μ, σ)=y$ and $x=x(y, ω, μ, σ)$, implicit differentiation gives

        .. math::  ∂T(x(y, ω, μ, σ), ω, μ, σ) = y

        Hence

        .. math::
            ∂x/∂y &= (∂T/∂x)⁻¹ \\
            ∂x/∂θ &= -(∂T/∂x)⁻¹ ∂T/∂θ,
            \qquad θ∈\{ω, μ, σ\}.
        """
        (g,) = outer
        z, y, weights, _, sigmas = ctx.saved_tensors
        d_x, d_weights, d_mus, d_sigmas = _mixture_to_gaussian_derivatives(
            z, weights, sigmas, y
        )
        grad_y = g * d_x.reciprocal()
        grad_weights = torch.einsum("..., ...k -> k", grad_y, -d_weights)
        grad_mus = torch.einsum("..., ...k -> k", grad_y, -d_mus)
        grad_sigmas = torch.einsum("..., ...k -> k", grad_y, -d_sigmas)

        return grad_y, grad_weights, grad_mus, grad_sigmas


def gaussian_to_bimodal(
    y: Tensor, /, mu: Tensor | float = 2.0, sigma: Tensor | float = 1.0
) -> Tensor:
    r"""Map $N(0,1)$ to the symmetric mixture $½N(-μ,σ²) + ½N(μ,σ²)$.

    This is the inverse of

    .. math:: y = Φ⁻¹\Bigl(½Φ((x+μ)/σ) + ½Φ((x-μ)/σ)\Bigr)

    The inverse map is not available in closed form and is computed with a
    safeguarded Newton iteration. Evaluation of the underlying transport uses
    numerically stable lower-tail and upper-tail formulas based on `log_ndtr`
    and `ndtri_exp`.

    so the returned value is the unique $x$ whose bimodal CDF equals $Φ(y)$.
    """
    mu = torch.as_tensor(mu, dtype=y.dtype, device=y.device)
    sigma = torch.as_tensor(sigma, dtype=y.dtype, device=y.device)
    return _GaussianToBimodalImpl.apply(y, mu, sigma)


def bimodal_to_gaussian(
    x: Tensor, /, mu: Tensor | float = 2.0, sigma: Tensor | float = 1.0
) -> Tensor:
    r"""Map the symmetric mixture $½N(-μ,σ²) + ½N(μ,σ²)$ to $N(0,1)$.

    .. math:: y = Φ⁻¹\Bigl(½Φ((x+μ)/σ) + ½Φ((x-μ)/σ)\Bigr)

    The transport is evaluated with numerically stable lower-tail and upper-tail
    formulas based on `log_ndtr` and `ndtri_exp`.
    """
    mu = torch.as_tensor(mu, dtype=x.dtype, device=x.device)
    sigma = torch.as_tensor(sigma, dtype=x.dtype, device=x.device)
    return _BimodalToGaussianImpl.apply(x, mu, sigma)


def bimodal_to_gaussian_value_and_jac(
    x: Tensor, /, mu: Tensor | float = 2.0, sigma: Tensor | float = 1.0
) -> tuple[Tensor, Tensor]:
    r"""Map the symmetric mixture to $N(0,1)$ and return $(f(x), ∂f/∂x)$."""
    mu = torch.as_tensor(mu, dtype=x.dtype, device=x.device)
    sigma = torch.as_tensor(sigma, dtype=x.dtype, device=x.device)
    return _BimodalToGaussianValueAndJacImpl.apply(x, mu, sigma)


def gaussian_to_mixture(
    y: Tensor, /, weights: Tensor, mus: Tensor, sigmas: Tensor
) -> Tensor:
    r"""Map $N(0,1)$ to the mixture $∑ₖ ωₖ N(μₖ,σₖ²)$.

    This is the inverse of

    .. math::  y = Φ⁻¹\Bigl(∑ₖ ωₖΦ((x-μₖ)/σₖ)\Bigr)

    The inverse map is not available in closed form and is computed with a
    safeguarded Newton iteration. Evaluation of the underlying transport uses
    numerically stable lower-tail and upper-tail formulas based on `log_ndtr`
    and `ndtri_exp`.

    so the returned value is the unique $x$ whose mixture CDF equals $Φ(y)$.
    """
    return _GaussianToMixture.apply(y, weights, mus, sigmas)


def mixture_to_gaussian(
    x: Tensor, /, weights: Tensor, mus: Tensor, sigmas: Tensor
) -> Tensor:
    r"""Map the mixture $∑ₖ ωₖ N(μₖ,σₖ²)$ to $N(0,1)$.

    .. math::  y = Φ⁻¹\Bigl(∑ₖ ωₖΦ((x-μₖ)/σₖ)\Bigr)

    The transport is evaluated with numerically stable lower-tail and upper-tail
    formulas based on `log_ndtr` and `ndtri_exp`.
    """
    return _MixtureToGaussian.apply(x, weights, mus, sigmas)
