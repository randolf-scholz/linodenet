r"""Implementation of the optimal transport based activation function."""
# mypy: disable-error-code="no-untyped-def"

__all__ = [
    # functional interfaces
    "gaussian_to_bimodal",
    "gaussian_to_bimodal_value_and_grad",
    "bimodal_to_gaussian",
    "bimodal_to_gaussian_value_and_grad",
    "gaussian_to_mixture",
    "gaussian_to_mixture_value_and_grad",
    "mixture_to_gaussian",
    "mixture_to_gaussian_value_and_grad",
]

from typing import Final

import torch
from torch import Tensor
from torch.linalg import vecdot
from torch.special import log_ndtr

from linodenet_special.interfaces import DEFAULT_NEWTON_MAXITER

from .hard_bend import hard_bend
from .ndtri_exp import ndtri_exp


def _bimodal_value_and_stats(
    x: Tensor, mu: Tensor, sigma: Tensor, /
) -> tuple[Tensor, Tensor, Tensor]:
    r"""Return the transport value and shared bimodal intermediates."""
    LOG_HALF: Final[float] = -0.6931471805599453  # log(½)

    m = mu.abs()
    z_plus = (x + m) / sigma
    z_minus = (x - m) / sigma
    log_p = torch.logaddexp(LOG_HALF + log_ndtr(z_plus), LOG_HALF + log_ndtr(z_minus))
    log_q = torch.logaddexp(LOG_HALF + log_ndtr(-z_plus), LOG_HALF + log_ndtr(-z_minus))
    y = torch.where(log_p < LOG_HALF, ndtri_exp(log_p), -ndtri_exp(log_q))
    return y.clamp(z_minus, z_plus), z_plus, z_minus


def _bimodal_to_gaussian_value_and_grad(
    x: Tensor, mu: Tensor, sigma: Tensor, /
) -> tuple[Tensor, Tensor]:
    r"""Evaluate the bimodal transport and its $x$-derivative in one pass."""
    LOG_HALF: Final[float] = -0.6931471805599453  # log(½)

    fx, z_plus, z_minus = _bimodal_value_and_stats(x, mu, sigma)
    log_sigma = sigma.log()
    y2 = fx.square()
    log_phi_plus = 0.5 * (y2 - z_plus.square()) - log_sigma + LOG_HALF
    log_phi_minus = 0.5 * (y2 - z_minus.square()) - log_sigma + LOG_HALF
    lower_bound = torch.exp(-0.5 * (mu / sigma) ** 2) / sigma
    upper_bound = 1 / sigma
    d_fx = torch.logaddexp(log_phi_plus, log_phi_minus).exp()
    d_fx = d_fx.clamp(lower_bound, upper_bound)

    return fx, d_fx


def _bimodal_to_gaussian_derivatives(
    x: Tensor, mu: Tensor, sigma: Tensor, y: Tensor, /
) -> tuple[Tensor, Tensor, Tensor]:
    r"""Compute stable partial derivatives for the bimodal-to-Gaussian transport.

    Returns:
        ∂y/∂x:  ½σ⁻¹(E₊ + E₋)
        ∂y/∂μ:  ½σ⁻¹(E₊ - E₋)
        ∂y/∂σ: -½σ⁻¹(z₊E₊ + z₋E₋)
    """
    LOG_HALF: Final[float] = -0.6931471805599453  # log(½)

    m = mu.abs()
    z_plus = (x + m) / sigma
    z_minus = (x - m) / sigma
    log_sigma = sigma.log()
    mu_sign = torch.sign(mu)
    y2 = y.square()
    # Evaluate the two mode contributions in log space to avoid tail underflow.
    log_phi_plus = 0.5 * (y2 - z_plus.square()) - log_sigma + LOG_HALF
    log_phi_minus = 0.5 * (y2 - z_minus.square()) - log_sigma + LOG_HALF

    log_norm = torch.logaddexp(log_phi_plus, log_phi_minus)
    d_x = log_norm.exp()
    w_plus = torch.exp(log_phi_plus - log_norm)
    w_minus = torch.exp(log_phi_minus - log_norm)
    d_mu_abs = d_x * (w_plus - w_minus)
    d_sigma_exact = -(0.5 * (z_plus + z_minus) * d_x + (m / sigma) * d_mu_abs)

    # The analytic slope lives in [exp(-½(m/σ)²)/σ, 1/σ]; clamp only to absorb drift.
    lower_bound = torch.exp(-0.5 * (m / sigma) ** 2) / sigma
    upper_bound = 1 / sigma
    d_x = d_x.clamp(lower_bound, upper_bound)
    d_mu = mu_sign * d_mu_abs.clamp(-upper_bound, upper_bound)
    return d_x, d_mu, d_sigma_exact


def _bimodal_to_gaussian_derivatives2(
    x: Tensor, mu: Tensor, sigma: Tensor, y: Tensor, /
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    r"""Compute first and second derivatives for the bimodal-to-Gaussian transport.

    Returns:
        ∂y/∂x:  ½σ⁻¹(E₊ + E₋)
        ∂y/∂μ:  ½σ⁻¹(E₊ - E₋)
        ∂y/∂σ: -½σ⁻¹(z₊E₊ + z₋E₋)
        ∂g/∂x: -½σ⁻²(z₊E₊ + z₋E₋)   + yg(∂y/∂x)
        ∂g/∂μ: -½σ⁻²(z₊E₊ - z₋E₋)   + yg(∂y/∂μ)
        ∂g/∂σ: +½σ⁻²(z₊²E₊ + z₋²E₋) + yg(∂y/∂σ) -g/σ
    """
    LOG_HALF: Final[float] = -0.6931471805599453  # log(½)

    m = mu.abs()
    z_plus = (x + m) / sigma
    z_minus = (x - m) / sigma
    log_sigma = sigma.log()
    mu_sign = torch.sign(mu)
    y2 = y.square()
    # Evaluate the two mode contributions in log space to avoid tail underflow.
    log_phi_plus = 0.5 * (y2 - z_plus.square()) - log_sigma + LOG_HALF
    log_phi_minus = 0.5 * (y2 - z_minus.square()) - log_sigma + LOG_HALF

    log_norm = torch.logaddexp(log_phi_plus, log_phi_minus)
    d_x = log_norm.exp()
    w_plus = torch.exp(log_phi_plus - log_norm)
    w_minus = torch.exp(log_phi_minus - log_norm)
    d_mu_abs = d_x * (w_plus - w_minus)
    d_sigma_exact = -(0.5 * (z_plus + z_minus) * d_x + (m / sigma) * d_mu_abs)

    # The analytic slope lives in [exp(-½(m/σ)²)/σ, 1/σ]; clamp only to absorb drift.
    lower_bound = torch.exp(-0.5 * (m / sigma) ** 2) / sigma
    upper_bound = 1 / sigma
    d_x = d_x.clamp(lower_bound, upper_bound)
    d_mu = mu_sign * d_mu_abs.clamp(-upper_bound, upper_bound)

    # Reuse d_x as the common scale so only the normalized weights need exponentials.
    z_avg = z_plus * w_plus + z_minus * w_minus
    z_diff = z_plus * w_plus - z_minus * w_minus
    z2_avg = z_plus.square() * w_plus + z_minus.square() * w_minus
    z_term_sum = d_x * z_avg / sigma
    z_term_diff = d_x * z_diff / sigma
    z2_term_sum = d_x * z2_avg / sigma

    # Here g = ∂y/∂x, so these are the second derivatives that feed the Jacobian output.
    d2_x = y * d_x.square() - z_term_sum
    d2_mu = mu_sign * (y * d_x * d_mu_abs - z_term_diff)
    d2_sigma = y * d_x * d_sigma_exact - d_x / sigma + z2_term_sum
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


def _gaussian_to_bimodal_value_and_grad(
    y: Tensor, mu: Tensor, sigma: Tensor, maxiter: int, /
) -> tuple[Tensor, Tensor]:
    r"""Solve the inverse bimodal transport by safeguarded Newton iteration."""
    m = mu.abs()
    lower = sigma * y - m
    upper = sigma * y + m
    x = _gaussian_to_bimodal_guess(y, mu, sigma)
    x = x.clamp(lower, upper)  # unnecessary.
    fx, d_fx = _bimodal_to_gaussian_value_and_grad(x, mu, sigma)
    r = fx - y

    for _ in range(maxiter):  # consider adding tol / using torch.while_loop.
        lower = torch.where(r < 0, x, lower)
        upper = torch.where(r > 0, x, upper)
        x_newton = x - r / d_fx
        x_bisect = 0.5 * (lower + upper)
        x = torch.where(
            (x_newton >= lower) & (x_newton <= upper),
            x_newton,
            x_bisect,
        ).clamp(lower, upper)
        fx, d_fx = _bimodal_to_gaussian_value_and_grad(x, mu, sigma)
        r = fx - y

    return x, d_fx.reciprocal()


def _mixture_value_and_stats(
    x: Tensor, weights: Tensor, mus: Tensor, sigmas: Tensor, /
) -> tuple[Tensor, Tensor, Tensor]:
    r"""Return the transport value and shared mixture intermediates."""
    assert weights.shape[0] == mus.shape[0] == sigmas.shape[0]
    LOG_HALF: Final[float] = -0.6931471805599453  # log(½)

    z = (x.unsqueeze(-1) - mus) / sigmas
    log_w = torch.log(weights)
    log_p = torch.logsumexp(log_w + log_ndtr(z), dim=-1)
    log_q = torch.logsumexp(log_w + log_ndtr(-z), dim=-1)
    y = torch.where(log_p < LOG_HALF, ndtri_exp(log_p), -ndtri_exp(log_q))
    return y.clamp(z.amin(dim=-1), z.amax(dim=-1)), z, log_w


def _mixture_to_gaussian_value_and_grad(
    x: Tensor, weights: Tensor, mus: Tensor, sigmas: Tensor, /
) -> tuple[Tensor, Tensor]:
    r"""Evaluate the mixture transport and its $x$-derivative in one pass."""
    fx, z, log_w = _mixture_value_and_stats(x, weights, mus, sigmas)
    log_sigmas = torch.log(sigmas)
    log_ratio = 0.5 * (fx.square().unsqueeze(-1) - z.square())
    d_fx = torch.exp(log_ratio + log_w - log_sigmas).sum(dim=-1)
    return fx, d_fx


def _mixture_to_gaussian_derivatives(
    x: Tensor, weights: Tensor, mus: Tensor, sigmas: Tensor, y: Tensor, /
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    r"""Compute stable partial derivatives for the mixture-to-Gaussian transport.

    Returns:
        ∂y/∂x  = ∑ₖ(ωₖ/σₖ)Eₖ
        ∂y/∂ωₖ = √(2π)ℯ^{½y²}(Φ(zₖ) - (1/n)∑ⱼΦ(zⱼ))
        ∂y/∂μₖ = -(ωₖ/σₖ)Eₖ
        ∂y/∂σₖ = -(ωₖ/σₖ)zₖEₖ
    """
    LOG_2PI: Final[float] = 1.8378770664093453  # log(2π)

    z = (x.unsqueeze(-1) - mus) / sigmas
    log_w = torch.log(weights)
    log_sigmas = torch.log(sigmas)
    y2 = y.square()
    # exp(½(y² - zₖ²)) = φ(zₖ) / φ(y)
    log_ratio = 0.5 * (y2.unsqueeze(-1) - z.square())
    # (ωₖ / σₖ) exp(½(y² - zₖ²)) appears in ∂y/∂x, ∂y/∂μₖ, and ∂y/∂σₖ.
    scaled_ratio = torch.exp(log_ratio + log_w - log_sigmas)

    # ∂y/∂x = ∑ₖ (ωₖ / σₖ) exp(½(y² - zₖ²))
    d_x = scaled_ratio.sum(dim=-1)
    # ∂y/∂μₖ = -(ωₖ / σₖ) exp(½(y² - zₖ²))
    d_mus = -scaled_ratio
    # ∂y/∂σₖ = -(ωₖ zₖ / σₖ) exp(½(y² - zₖ²))
    d_sigmas = -z * scaled_ratio

    # ∂y/∂ωₖ = √(2π) ℯ^{½y²}⋅(Φ(zₖ) - (1/n)∑ⱼΦ(zⱼ)).
    # Factor out max(log Φ(zₖ)) to keep the centered CDF difference in a stable range.
    log_pdf_u = (-0.5 * (LOG_2PI + y2)).unsqueeze(-1)
    log_phi = log_ndtr(z)
    log_phi_max = log_phi.amax(dim=-1, keepdim=True)
    scaled_phi = torch.exp(log_phi - log_phi_max)
    centered_scaled_phi = scaled_phi - scaled_phi.mean(dim=-1, keepdim=True)
    d_weights = torch.exp(log_phi_max - log_pdf_u) * centered_scaled_phi

    return d_x, d_weights, d_mus, d_sigmas


def _mixture_to_gaussian_derivatives2(
    x: Tensor, weights: Tensor, mus: Tensor, sigmas: Tensor, y: Tensor, /
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    r"""Compute first and second derivatives for the mixture-to-Gaussian transport.

    Returns:
        ∂y/∂x  = ∑ₖ(ωₖ/σₖ)Eₖ
        ∂y/∂ωₖ = √(2π)ℯ^{½y²}(Φ(zₖ) - (1/n)∑ⱼΦ(zⱼ))
        ∂y/∂μₖ = -(ωₖ/σₖ)Eₖ
        ∂y/∂σₖ = -(ωₖ/σₖ)zₖEₖ
        ∂g/∂x  = y⋅g⋅(∂y/∂x)  - ∑ₖ(ωₖ/σₖ²)zₖEₖ
        ∂g/∂ωₖ = y⋅g⋅(∂y/∂ωₖ) + Eₖ/σₖ - (1/n)∑ⱼEⱼ/σⱼ
        ∂g/∂μₖ = y⋅g⋅(∂y/∂μₖ) + (ωₖ/σₖ²)zₖEₖ
        ∂g/∂σₖ = y⋅g⋅(∂y/∂σₖ) + (ωₖ/σₖ²)(zₖ²-1)Eₖ
    """
    LOG_2PI: Final[float] = 1.8378770664093453  # log(2π)

    _, z, log_w = _mixture_value_and_stats(x, weights, mus, sigmas)
    log_sigmas = torch.log(sigmas)
    y2 = y.square()
    # exp(½(y² - zₖ²)) = φ(zₖ) / φ(y)
    log_ratio = 0.5 * (y2.unsqueeze(-1) - z.square())
    scaled_ratio = torch.exp(log_ratio + log_w - log_sigmas)

    d_x = scaled_ratio.sum(dim=-1)
    d_mus = -scaled_ratio
    d_sigmas = -z * scaled_ratio

    log_pdf_u = (-0.5 * (LOG_2PI + y2)).unsqueeze(-1)
    log_phi = log_ndtr(z)
    log_phi_max = log_phi.amax(dim=-1, keepdim=True)
    scaled_phi = torch.exp(log_phi - log_phi_max)
    centered_scaled_phi = scaled_phi - scaled_phi.mean(dim=-1, keepdim=True)
    d_weights = torch.exp(log_phi_max - log_pdf_u) * centered_scaled_phi

    e_over_sigma = torch.exp(log_ratio - log_sigmas)
    d2_x = y * d_x.square() + (d_sigmas / sigmas).sum(dim=-1)
    d2_weights = y.unsqueeze(-1) * d_x.unsqueeze(-1) * d_weights
    d2_weights = d2_weights + e_over_sigma - e_over_sigma.mean(dim=-1, keepdim=True)
    d2_mus = y.unsqueeze(-1) * d_x.unsqueeze(-1) * d_mus - d_sigmas / sigmas
    d2_sigmas = (
        y.unsqueeze(-1) * d_x.unsqueeze(-1) * d_sigmas
        + (z.square() - 1) * scaled_ratio / sigmas
    )
    return d_x, d_weights, d_mus, d_sigmas, d2_x, d2_weights, d2_mus, d2_sigmas


def _gaussian_to_mixture_value_and_grad(
    y: Tensor, weights: Tensor, mus: Tensor, sigmas: Tensor, maxiter: int, /
) -> tuple[Tensor, Tensor]:
    r"""Solve the inverse mixture transport by safeguarded Newton iteration."""
    assert weights.shape[0] == mus.shape[0] == sigmas.shape[0]
    # Each component alone would invert y to xₖ = μₖ + σₖy. The mixture inverse
    # must lie between the smallest and largest of these affine tail candidates,
    # so we use their pointwise min/max as a safe bracket and their weighted mean
    # as a cheap initial guess for the safeguarded Newton iteration.
    lines = mus + sigmas * y.unsqueeze(-1)
    lower = lines.amin(dim=-1)
    upper = lines.amax(dim=-1)
    x = vecdot(weights, lines, dim=-1)
    x = x.clamp(lower, upper)
    y_star, d_fx = _mixture_to_gaussian_value_and_grad(x, weights, mus, sigmas)
    r = y_star - y

    for _ in range(maxiter):
        lower = torch.where(r < 0, x, lower)
        upper = torch.where(r > 0, x, upper)
        x_newton = x - r / d_fx
        x_bisect = 0.5 * (lower + upper)
        x = torch.where(
            (x_newton >= lower) & (x_newton <= upper),
            x_newton,
            x_bisect,
        ).clamp(lower, upper)
        y_star, d_fx = _mixture_to_gaussian_value_and_grad(x, weights, mus, sigmas)
        r = y_star - y

    return x, d_fx.reciprocal()


class _BimodalToGaussian(torch.autograd.Function):
    r"""Optimal Transport from mixture $p = ½N(-μ, σ²) + ½N(μ, σ²)$ to $q = N(0, 1)$.

    If $F_p$ and $F_q$ are the CDFs of $p$ and $q$, then the optimal transport map is given by

    .. math:: y = F_q⁻¹(Fₚ(x))

    Letting Φ be the CDF of $N(0,1)$, then we have

    .. math:: y = Φ⁻¹\Bigl( ½Φ((x+μ)/σ) + ½Φ((x-μ)/σ) \Bigr)
                = √2 \erf⁻¹\Bigl(½\erf((x+μ)/√2σ) + ½\erf((x-μ)/√2σ) \Bigr)

    Asymptotic Expansion with $g(x)=\frac{x-\sign(x)|μ|}{σ}$:

    .. math:: y ~ g(x) + \log(2)/g(x) as x → ±∞

    Unlike the general mixture case, the two components share the mean $±μ$ and scale $σ$.

    Using the shorthands

    .. math::
        z₊ &= \frac{x+μ}{σ}     &   z₋ &= \frac{x-μ}{σ} \\
        E₊ &= ℯ^{½(y²-z₊²)}     &   E₋ &= ℯ^{½(y²-z₋²)}

    The first order derivatives can be written as:

    .. math::
        ∂y/∂x &=  ½σ⁻¹(E₊ + E₋)    \\
        ∂y/∂μ &=  ½σ⁻¹(E₊ - E₋)    \\
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
    def forward(x: Tensor, mu: Tensor, sigma: Tensor, /) -> Tensor:
        return _bimodal_value_and_stats(x, mu, sigma)[0]

    @staticmethod
    def setup_context(ctx, inputs, output) -> None:
        ctx.save_for_backward(*inputs, output)  # x, μ, σ, y

    @staticmethod
    def backward(ctx, *outer: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        (g,) = outer
        x, mu, sigma, y = ctx.saved_tensors
        d_x, d_mu, d_sigma = _bimodal_to_gaussian_derivatives(x, mu, sigma, y)
        return (g * d_x), (g * d_mu), (g * d_sigma)


class _BimodalToGaussianValueAndGrad(torch.autograd.Function):
    r"""Return the bimodal-to-Gaussian transport and its $x$-derivative."""

    @staticmethod
    @torch.no_grad()
    def forward(x: Tensor, mu: Tensor, sigma: Tensor, /) -> tuple[Tensor, Tensor]:
        return _bimodal_to_gaussian_value_and_grad(x, mu, sigma)

    @staticmethod
    def setup_context(ctx, inputs, output) -> None:
        ctx.save_for_backward(*inputs, output[0])  # x, μ, σ, y

    @staticmethod
    def backward(ctx, *outer: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        grad_y, grad_dy = outer
        x, mu, sigma, y = ctx.saved_tensors
        d_x, d_mu, d_sigma, d2_x, d2_mu, d2_sigma = _bimodal_to_gaussian_derivatives2(
            x, mu, sigma, y
        )
        return (
            grad_y * d_x + grad_dy * d2_x,
            grad_y * d_mu + grad_dy * d2_mu,
            grad_y * d_sigma + grad_dy * d2_sigma,
        )


class _GaussianToBimodal(torch.autograd.Function):
    r"""Optimal Transport from $N(0, 1)$ to symmetric mixture $½N(-μ, σ²) + ½N(μ, σ²)$."""

    @staticmethod
    @torch.no_grad()
    def forward(y: Tensor, mu: Tensor, sigma: Tensor, maxiter: int, /) -> Tensor:
        r"""Solve $y = T(x, μ, σ)$ for $x$ using Newton's method.

        Here $T$ is the transport from the symmetric bimodal mixture to $N(0,1)$.
        Since $T'(0) = σ⁻¹ℯ^{-½|μ/σ|²}$, the inverse slope at the origin is

        .. math:: (T⁻¹)'(0) = σℯ^{½|μ/σ|²}

        The transport only depends on $|μ|$. The tails satisfy
        $T(x, μ, σ) ≈ σ⁻¹(x-\sign(x)|μ|)$, so
        $T⁻¹(y, μ, σ) ≈ σy + \sign(y)|μ|$.
        """
        return _gaussian_to_bimodal_value_and_grad(y, mu, sigma, maxiter)[0]

    @staticmethod
    def setup_context(ctx, inputs, output) -> None:
        ctx.save_for_backward(*inputs[:-1], output)  # y, μ, σ, x

    @staticmethod
    def backward(ctx, *outer: Tensor) -> tuple[Tensor, Tensor, Tensor, None]:
        r"""Use the derivatives of $T$ to differentiate the inverse map.

        .. math::  ∂T(x(y, μ, σ), μ, σ) = y

        Hence

        .. math::
            ∂x/∂y &= (∂T/∂x)⁻¹ \\
            ∂x/∂μ &= -(∂T/∂x)⁻¹ ∂T/∂μ \\
            ∂x/∂σ &= -(∂T/∂x)⁻¹ ∂T/∂σ

        For the inverse Jacobian $j = ∂x/∂y = (∂T/∂x)⁻¹$, differentiating once
        more gives

        .. math::
            ∂j/∂y &= -(∂²T/∂x²)(∂T/∂x)⁻³ \\
            ∂j/∂μ &= (∂²T/∂x²)(∂T/∂μ)(∂T/∂x)⁻³ - (∂²T/∂x∂μ)(∂T/∂x)⁻² \\
            ∂j/∂σ &= (∂²T/∂x²)(∂T/∂σ)(∂T/∂x)⁻³ - (∂²T/∂x∂σ)(∂T/∂x)⁻².
        """
        (g,) = outer
        y, mu, sigma, x = ctx.saved_tensors
        d_x, d_mu, d_sigma = _bimodal_to_gaussian_derivatives(x, mu, sigma, y)
        dx_inv = d_x.reciprocal()

        d_y = dx_inv
        d_mu = -d_mu * dx_inv
        d_sigma = -d_sigma * dx_inv

        # clamp to legal range
        lower_bound = sigma
        upper_bound = sigma * torch.exp(0.5 * (mu / sigma) ** 2)
        d_y = d_y.clamp(lower_bound, upper_bound)
        # ∂x/∂μ = -(∂T/∂μ)/(∂T/∂x) is dimensionless and equals ±1 in the tails.
        d_mu = d_mu.clamp(-1, 1)

        return (g * d_y), (g * d_mu), (g * d_sigma), None


class _GaussianToBimodalValueAndGrad(torch.autograd.Function):
    r"""Return the Gaussian-to-bimodal transport and its $y$-derivative."""

    @staticmethod
    @torch.no_grad()
    def forward(
        y: Tensor, mu: Tensor, sigma: Tensor, maxiter, /
    ) -> tuple[Tensor, Tensor]:
        return _gaussian_to_bimodal_value_and_grad(y, mu, sigma, maxiter)

    @staticmethod
    def setup_context(ctx, inputs, output) -> None:
        ctx.save_for_backward(*inputs[:-1], output[0])  # y, μ, σ, x

    @staticmethod
    def backward(ctx, *outer: Tensor) -> tuple[Tensor, Tensor, Tensor, None]:
        r"""Use the derivatives of $T$ to differentiate the inverse map.

        .. math::  ∂T(x(y, μ, σ), μ, σ) = y

        .. math::
            ∂j/∂y &= -(∂²T/∂x²)(∂T/∂x)⁻³ \\
            ∂j/∂μ &= (∂²T/∂x²)(∂T/∂μ)(∂T/∂x)⁻³ - (∂²T/∂x∂μ)(∂T/∂x)⁻² \\
            ∂j/∂σ &= (∂²T/∂x²)(∂T/∂σ)(∂T/∂x)⁻³ - (∂²T/∂x∂σ)(∂T/∂x)⁻².
        """
        grad_x, grad_dx = outer
        y, mu, sigma, x = ctx.saved_tensors
        d_x, d_mu, d_sigma, d2_x, d2_mu, d2_sigma = _bimodal_to_gaussian_derivatives2(
            x, mu, sigma, y
        )
        dx_inv = d_x.reciprocal()

        d_y = dx_inv
        d_mu_inv = -d_mu * dx_inv
        d_sigma_inv = -d_sigma * dx_inv

        lower_bound = sigma
        upper_bound = sigma * torch.exp(0.5 * (mu / sigma) ** 2)
        d_y = d_y.clamp(lower_bound, upper_bound)
        # ∂x/∂μ = -(∂T/∂μ)/(∂T/∂x) is dimensionless and equals ±1 in the tails.
        d_mu_inv = d_mu_inv.clamp(-1, 1)

        j_y = -d2_x * dx_inv.pow(3)
        j_mu = (d2_x * d_mu - d2_mu * d_x) * dx_inv.pow(3)
        j_sigma = (d2_x * d_sigma - d2_sigma * d_x) * dx_inv.pow(3)

        return (
            grad_x * d_y + grad_dx * j_y,
            grad_x * d_mu_inv + grad_dx * j_mu,
            grad_x * d_sigma_inv + grad_dx * j_sigma,
            None,
        )


class _MixtureToGaussian(torch.autograd.Function):
    r"""Optimal transport from $∑ₖωₖN(μₖ,σₖ²)$ to $N(0,1)$.

    If $Fₚ$ is the CDF of the mixture and $Φ$ is the standard normal CDF, then

    .. math:: y = Φ⁻¹(Fₚ(x)) = Φ⁻¹\Bigl(∑ₖ ωₖ Φ((x-μₖ)/σₖ)\Bigr)

    Numerically, we evaluate the mixture CDF in log space and switch between the
    lower-tail and upper-tail representations to avoid cancellation near $0$ and $1$.

    Using the shorthands

    .. math:: zₖ &= (x-μₖ)/σₖ  &  Eₖ &= ℯ^{½(y²-zₖ²)}

    then the first order derivatives are

    .. math::
        ∂y/∂x  &= ∑ₖ(ωₖ/σₖ)Eₖ    \\
        ∂y/∂ωₖ &= √(2π)ℯ^{½y²}(Φ(zₖ) - (1/n)∑ⱼΦ(zⱼ))    \\
        ∂y/∂μₖ &= -(ωₖ/σₖ)Eₖ     \\
        ∂y/∂σₖ &= -(ωₖ/σₖ)zₖEₖ

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
    def forward(x: Tensor, weights: Tensor, mus: Tensor, sigmas: Tensor, /) -> Tensor:
        return _mixture_value_and_stats(x, weights, mus, sigmas)[0]

    @staticmethod
    def setup_context(ctx, inputs, output) -> None:
        ctx.save_for_backward(*inputs, output)  # x, ω, μ, σ, y

    @staticmethod
    def backward(ctx, *outer: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        (g,) = outer
        d_values, d_weights, d_mus, d_sigmas = _mixture_to_gaussian_derivatives(
            *ctx.saved_tensors
        )

        return (
            g * d_values,
            g.unsqueeze(-1) * d_weights,
            g.unsqueeze(-1) * d_mus,
            g.unsqueeze(-1) * d_sigmas,
        )


class _MixtureToGaussianValueAndGrad(torch.autograd.Function):
    r"""Return the mixture-to-Gaussian transport and its $x$-derivative."""

    @staticmethod
    @torch.no_grad()
    def forward(
        x: Tensor, weights: Tensor, mus: Tensor, sigmas: Tensor, /
    ) -> tuple[Tensor, Tensor]:
        return _mixture_to_gaussian_value_and_grad(x, weights, mus, sigmas)

    @staticmethod
    def setup_context(ctx, inputs, output) -> None:
        ctx.save_for_backward(*inputs, output[0])  # x, ω, μ, σ, y

    @staticmethod
    def backward(ctx, *outer: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        grad_y, grad_dy = outer
        d_x, d_weights, d_mus, d_sigmas, d2_x, d2_weights, d2_mus, d2_sigmas = (
            _mixture_to_gaussian_derivatives2(*ctx.saved_tensors)
        )
        return (
            grad_y * d_x + grad_dy * d2_x,
            grad_y.unsqueeze(-1) * d_weights + grad_dy.unsqueeze(-1) * d2_weights,
            grad_y.unsqueeze(-1) * d_mus + grad_dy.unsqueeze(-1) * d2_mus,
            grad_y.unsqueeze(-1) * d_sigmas + grad_dy.unsqueeze(-1) * d2_sigmas,
        )


class _GaussianToMixture(torch.autograd.Function):
    r"""Optimal Transport from $N(0,1)$ to mixture $∑ₖωₖN(μₖ, σₖ²)$.

    This inverse map is not available in closed form, so we compute it with a
    safeguarded Newton iteration.
    """

    @staticmethod
    @torch.no_grad()
    def forward(
        y: Tensor, weights: Tensor, mus: Tensor, sigmas: Tensor, maxiter: int, /
    ) -> Tensor:
        r"""Solve $T(x, ω, μ, σ)=y$ by safeguarded Newton iteration."""
        return _gaussian_to_mixture_value_and_grad(y, weights, mus, sigmas, maxiter)[0]

    @staticmethod
    def setup_context(ctx, inputs, output) -> None:
        ctx.save_for_backward(*inputs[:-1], output)  # y, ω, μ, σ, x

    @staticmethod
    def backward(ctx, *outer: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, None]:
        r"""Use the derivatives of $T$ to compute the derivatives of $T⁻¹$.

        Writing $T(x, ω, μ, σ)=y$ and $x=x(y, ω, μ, σ)$, implicit differentiation gives

        .. math::  ∂T(x(y, ω, μ, σ), ω, μ, σ) = y

        Hence

        .. math::
            ∂x/∂y &= (∂T/∂x)⁻¹ \\
            ∂x/∂θ &= -(∂T/∂x)⁻¹ ∂T/∂θ
            \qquad θ∈\{ω, μ, σ\}
        """
        (g,) = outer
        y, weights, mus, sigmas, x = ctx.saved_tensors
        d_x, d_weights, d_mus, d_sigmas = _mixture_to_gaussian_derivatives(
            x, weights, mus, sigmas, y
        )
        grad_y = g * d_x.reciprocal()
        return (
            grad_y,
            -grad_y.unsqueeze(-1) * d_weights,
            -grad_y.unsqueeze(-1) * d_mus,
            -grad_y.unsqueeze(-1) * d_sigmas,
            None,
        )


class _GaussianToMixtureValueAndGrad(torch.autograd.Function):
    r"""Return the Gaussian-to-mixture transport and its $y$-derivative.

    Writing $T(x, ω, μ, σ)=y$ and $x=x(y, ω, μ, σ)$, implicit differentiation gives

    .. math::  ∂T(x(y, ω, μ, σ), ω, μ, σ) = y

    Hence

    .. math::
        ∂x/∂y &= (∂T/∂x)⁻¹ \\
        ∂x/∂θ &= -(∂T/∂x)⁻¹ ∂T/∂θ
        \qquad θ∈\{ω, μ, σ\}

    For the inverse Jacobian $j = ∂x/∂y = (∂T/∂x)⁻¹$, differentiating once more gives

    .. math::
        ∂j/∂y &= -(∂²T/∂x²)(∂T/∂x)⁻³ \\
        ∂j/∂θ &= (∂²T/∂x²)(∂T/∂θ)(∂T/∂x)⁻³ - (∂²T/∂x∂θ)(∂T/∂x)⁻²
        \qquad θ∈\{ω, μ, σ\}.
    """

    @staticmethod
    @torch.no_grad()
    def forward(
        y: Tensor, weights: Tensor, mus: Tensor, sigmas: Tensor, maxiter: int, /
    ) -> tuple[Tensor, Tensor]:
        return _gaussian_to_mixture_value_and_grad(y, weights, mus, sigmas, maxiter)

    @staticmethod
    def setup_context(ctx, inputs, output) -> None:
        ctx.save_for_backward(*inputs[:-1], output[0])  # y, ω, μ, σ, x

    @staticmethod
    def backward(ctx, *outer: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, None]:
        grad_x, grad_dx = outer
        y, weights, mus, sigmas, x = ctx.saved_tensors
        d_x, d_weights, d_mus, d_sigmas, d2_x, d2_weights, d2_mus, d2_sigmas = (
            _mixture_to_gaussian_derivatives2(x, weights, mus, sigmas, y)
        )
        dx_inv = d_x.reciprocal()

        d_y = dx_inv
        d_weights_inv = -d_weights * dx_inv.unsqueeze(-1)
        d_mus_inv = -d_mus * dx_inv.unsqueeze(-1)
        d_sigmas_inv = -d_sigmas * dx_inv.unsqueeze(-1)

        j_y = -d2_x * dx_inv.pow(3)
        j_weights = (
            d2_x.unsqueeze(-1) * d_weights - d2_weights * d_x.unsqueeze(-1)
        ) * (dx_inv.pow(3).unsqueeze(-1))
        j_mus = (d2_x.unsqueeze(-1) * d_mus - d2_mus * d_x.unsqueeze(-1)) * (
            dx_inv.pow(3).unsqueeze(-1)
        )
        j_sigmas = (
            d2_x.unsqueeze(-1) * d_sigmas - d2_sigmas * d_x.unsqueeze(-1)
        ) * dx_inv.pow(3).unsqueeze(-1)

        return (
            grad_x * d_y + grad_dx * j_y,
            grad_x.unsqueeze(-1) * d_weights_inv + grad_dx.unsqueeze(-1) * j_weights,
            grad_x.unsqueeze(-1) * d_mus_inv + grad_dx.unsqueeze(-1) * j_mus,
            grad_x.unsqueeze(-1) * d_sigmas_inv + grad_dx.unsqueeze(-1) * j_sigmas,
            None,
        )


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
    return _BimodalToGaussian.apply(x, mu, sigma)


def bimodal_to_gaussian_value_and_grad(
    x: Tensor, /, mu: Tensor | float = 2.0, sigma: Tensor | float = 1.0
) -> tuple[Tensor, Tensor]:
    r"""Map the symmetric mixture to $N(0,1)$ and return $(f(x), ∂f/∂x)$."""
    mu = torch.as_tensor(mu, dtype=x.dtype, device=x.device)
    sigma = torch.as_tensor(sigma, dtype=x.dtype, device=x.device)
    return _BimodalToGaussianValueAndGrad.apply(x, mu, sigma)


def mixture_to_gaussian(
    x: Tensor, /, weights: Tensor, mus: Tensor, sigmas: Tensor
) -> Tensor:
    r"""Map the mixture $∑ₖ ωₖ N(μₖ,σₖ²)$ to $N(0,1)$.

    .. math::  y = Φ⁻¹\Bigl(∑ₖ ωₖΦ((x-μₖ)/σₖ)\Bigr)

    The transport is evaluated with numerically stable lower-tail and upper-tail
    formulas based on `log_ndtr` and `ndtri_exp`.
    """
    return _MixtureToGaussian.apply(x, weights, mus, sigmas)


def mixture_to_gaussian_value_and_grad(
    x: Tensor, /, weights: Tensor, mus: Tensor, sigmas: Tensor
) -> tuple[Tensor, Tensor]:
    r"""Map the mixture to $N(0,1)$ and return $(f(x), ∂f/∂x)$."""
    return _MixtureToGaussianValueAndGrad.apply(x, weights, mus, sigmas)


def gaussian_to_bimodal(
    y: Tensor,
    /,
    mu: Tensor | float = 2.0,
    sigma: Tensor | float = 1.0,
    *,
    maxiter: int | None = None,
) -> Tensor:
    r"""Map $N(0,1)$ to the symmetric mixture $½N(-μ,σ²) + ½N(μ,σ²)$.

    This is the inverse of

    .. math:: y = Φ⁻¹\Bigl(½Φ((x+μ)/σ) + ½Φ((x-μ)/σ)\Bigr)

    The inverse map is not available in closed form and is computed with a
    safeguarded Newton iteration. Evaluation of the underlying transport uses
    numerically stable lower-tail and upper-tail formulas based on `log_ndtr`
    and `ndtri_exp`.

    The returned value is the unique $x$ whose bimodal CDF equals $Φ(y)$.
    """
    mu = torch.as_tensor(mu, dtype=y.dtype, device=y.device)
    sigma = torch.as_tensor(sigma, dtype=y.dtype, device=y.device)
    maxiter = DEFAULT_NEWTON_MAXITER[y.dtype] if maxiter is None else maxiter
    return _GaussianToBimodal.apply(y, mu, sigma, maxiter)


def gaussian_to_bimodal_value_and_grad(
    y: Tensor,
    /,
    mu: Tensor | float = 2.0,
    sigma: Tensor | float = 1.0,
    *,
    maxiter: int | None = None,
) -> tuple[Tensor, Tensor]:
    r"""Map $N(0,1)$ to the symmetric mixture and return $(f(y), ∂f/∂y)$."""
    mu = torch.as_tensor(mu, dtype=y.dtype, device=y.device)
    sigma = torch.as_tensor(sigma, dtype=y.dtype, device=y.device)
    maxiter = DEFAULT_NEWTON_MAXITER[y.dtype] if maxiter is None else maxiter
    return _GaussianToBimodalValueAndGrad.apply(y, mu, sigma, maxiter)


def gaussian_to_mixture(
    y: Tensor,
    /,
    weights: Tensor,
    mus: Tensor,
    sigmas: Tensor,
    *,
    maxiter: int | None = None,
) -> Tensor:
    r"""Map $N(0,1)$ to the mixture $∑ₖ ωₖ N(μₖ,σₖ²)$.

    This is the inverse of

    .. math::  y = Φ⁻¹\Bigl(∑ₖ ωₖΦ((x-μₖ)/σₖ)\Bigr)

    The inverse map is not available in closed form and is computed with a
    safeguarded Newton iteration. Evaluation of the underlying transport uses
    numerically stable lower-tail and upper-tail formulas based on `log_ndtr`
    and `ndtri_exp`.

    The returned value is the unique $x$ whose mixture CDF equals $Φ(y)$.
    """
    maxiter = DEFAULT_NEWTON_MAXITER[y.dtype] if maxiter is None else maxiter
    return _GaussianToMixture.apply(y, weights, mus, sigmas, maxiter)


def gaussian_to_mixture_value_and_grad(
    y: Tensor,
    /,
    weights: Tensor,
    mus: Tensor,
    sigmas: Tensor,
    *,
    maxiter: int | None = None,
) -> tuple[Tensor, Tensor]:
    r"""Map $N(0,1)$ to the mixture and return $(f(y), ∂f/∂y)$."""
    maxiter = DEFAULT_NEWTON_MAXITER[y.dtype] if maxiter is None else maxiter
    return _GaussianToMixtureValueAndGrad.apply(y, weights, mus, sigmas, maxiter)
