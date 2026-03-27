r"""Implementation of the optimal transport based activation function."""
# mypy: disable-error-code="no-untyped-def"

__all__ = [
    # functional interfaces
    "gaussian_to_bimodal",
    "bimodal_to_gaussian",
    "gaussian_to_mixture",
    "mixture_to_gaussian",
]

from typing import Final

import torch
from torch import Tensor
from torch.autograd import Function
from torch.special import log_ndtr

from .hard_bend import hard_bend
from .ndtri_exp import ndtri_exp


def _bimodal_to_gaussian_forward(
    x: Tensor, mu: Tensor, sigma: Tensor
) -> tuple[Tensor, Tensor, Tensor]:
    r"""Evaluate the bimodal-to-Gaussian transport and cache the normalized coordinates."""
    LOG_HALF: Final[float] = -0.6931471805599453  # log(½)

    m = mu.abs()
    z_plus = (x + m) / sigma
    z_minus = (x - m) / sigma

    log_p = torch.logaddexp(LOG_HALF + log_ndtr(z_plus), LOG_HALF + log_ndtr(z_minus))
    log_q = torch.logaddexp(LOG_HALF + log_ndtr(-z_plus), LOG_HALF + log_ndtr(-z_minus))
    y = torch.where(log_p < LOG_HALF, ndtri_exp(log_p), -ndtri_exp(log_q))
    y = torch.clamp(y, z_minus, z_plus)
    assert y.isfinite().all()
    return y, z_minus, z_plus


def _bimodal_to_gaussian_x_derivative(
    y: Tensor, z_minus: Tensor, z_plus: Tensor, mu: Tensor, sigma: Tensor
) -> Tensor:
    r"""Compute stable partial derivatives for the bimodal-to-Gaussian transport."""
    LOG_HALF: Final[float] = -0.6931471805599453  # log(½)
    m = mu.abs()
    y2 = y.square()
    log_sigma = sigma.log()
    log_phi_plus = 0.5 * (y2 - z_plus.square()) - log_sigma + LOG_HALF
    log_phi_minus = 0.5 * (y2 - z_minus.square()) - log_sigma + LOG_HALF
    d_x_exact = torch.logaddexp(log_phi_plus, log_phi_minus).exp()
    lower_bound = torch.exp(-0.5 * (m / sigma) ** 2) / sigma
    upper_bound = 1 / sigma
    d_x = torch.clamp(d_x_exact, lower_bound, upper_bound)
    return d_x


def _bimodal_to_gaussian_derivatives(
    y: Tensor, z_minus: Tensor, z_plus: Tensor, mu: Tensor, sigma: Tensor
) -> tuple[Tensor, Tensor, Tensor]:
    r"""Compute stable partial derivatives for the bimodal-to-Gaussian transport."""
    LOG_HALF: Final[float] = -0.6931471805599453  # log(½)

    m = mu.abs()
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


def _gaussian_to_bimodal_guess(x, mu, sigma):
    r"""Approximate $Ψ⁻¹(x, μ, σ)$ by the matching `hard_bend` inverse.

    Here $λ = Ψ'(0, μ, σ) = σ⁻¹ℯ^{-½(μ/σ)²}$ is the slope at the origin.

    Using

    .. math::  y = hard\_bend(x, 1/λ, μ, σ) \iff x = hard\_bend(y, λ, μ, 1/σ),

    we obtain a cheap initial guess for the safeguarded Newton solve.
    """
    λ = torch.exp(-0.5 * (mu / sigma) ** 2) / sigma
    return hard_bend(x, λ, mu, 1 / sigma)


def _mixture_to_gaussian_forward(
    x: Tensor, weights: Tensor, mus: Tensor, sigmas: Tensor
) -> tuple[Tensor, Tensor]:
    r"""Evaluate the mixture-to-Gaussian transport and cache normalized coordinates."""
    LOG_HALF: Final[float] = -0.6931471805599453  # log(½)

    z = (x.unsqueeze(-1) - mus) / sigmas
    log_w = torch.log(weights)
    log_p = torch.logsumexp(log_w + log_ndtr(z), dim=-1)
    log_q = torch.logsumexp(log_w + log_ndtr(-z), dim=-1)

    y = torch.where(log_p < LOG_HALF, ndtri_exp(log_p), -ndtri_exp(log_q))
    y = torch.clamp(y, z.min(dim=-1).values, z.max(dim=-1).values)
    assert y.isfinite().all()
    return y, z


def _mixture_to_gaussian_x_derivative(
    y: Tensor, z: Tensor, weights: Tensor, sigmas: Tensor
) -> Tensor:
    r"""Compute $∂T/∂x$ for the mixture-to-Gaussian transport."""
    y2 = y.square()
    log_ratio = 0.5 * (y2.unsqueeze(-1) - z.square())
    scaled_ratio = torch.exp(log_ratio + torch.log(weights) - torch.log(sigmas))
    return scaled_ratio.sum(dim=-1)


def _mixture_to_gaussian_derivatives(
    y: Tensor, z: Tensor, weights: Tensor, sigmas: Tensor
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

    log_pdf_u = -0.5 * (LOG_2PI + y2)
    d_weights = -torch.exp(log_ndtr(-z) - log_pdf_u.unsqueeze(-1))

    return d_x, d_weights, d_mus, d_sigmas


class _BimodalToGaussianImpl(Function):
    r"""Optimal Transport from mixture $p = ½N(-μ, σ²) + ½N(μ, σ²)$ to $q = N(0, 1)$.

    If $F_p$ and $F_q$ are the CDFs of $p$ and $q$, then the optimal transport map is given by

    .. math:: y = F_q⁻¹(F_p(x))

    Letting Φ be the CDF of $N(0,1)$, then we have

    .. math:: y = Φ⁻¹\Bigl( ½Φ((x+μ)/σ) + ½Φ((x-μ)/σ) \Bigr)
                = √2 \erf⁻¹\Bigl( ½\erf((x+μ)/√2σ) + ½\erf((x-μ)/√2σ) \Bigr)

    Unlike the general mixture case, the two components share the mean $±μ$ and scale $σ$.
    Writing $z₊ = \frac{x+μ}{σ}$ and $z₋ = \frac{x-μ}{σ}$, then the derivatives are

    .. math::
        \dv{y}{x} &= ½σ⁻¹(ℯ^{½(y²-z₊²)} + ℯ^{½(y²-z₋²)}) \\
        \dv{y}{μ} &= ½σ⁻¹(ℯ^{½(y²-z₊²)} - ℯ^{½(y²-z₋²)}) \\
        \dv{y}{σ} &= -½σ⁻¹(z₊ℯ^{½(y²-z₊²)} + z₋ℯ^{½(y²-z₋²)})

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
        y, z_minus, z_plus = _bimodal_to_gaussian_forward(x, mu, sigma)
        ctx.save_for_backward(y, z_minus, z_plus, mu, sigma)
        return y

    @staticmethod
    def backward(ctx, *outer: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        (g,) = outer
        y, z_minus, z_plus, mu, sigma = ctx.saved_tensors
        d_x, d_mu, d_sigma = _bimodal_to_gaussian_derivatives(
            y, z_minus, z_plus, mu, sigma
        )
        return (g * d_x), (g * d_mu), (g * d_sigma)


class _GaussianToBimodalImpl(Function):
    r"""Optimal Transport from $N(0, 1)$ to symmetric mixture $½N(-μ, σ²) + ½N(μ, σ²)$."""

    @staticmethod
    @torch.no_grad()
    def forward(ctx, y: Tensor, mu: Tensor, sigma: Tensor, /) -> Tensor:
        r"""Solve $y = T(x, μ, σ)$ for $x$ using Newton's method.

        Here $T$ is the transport from the symmetric bimodal mixture to $N(0,1)$.
        Since $T'(0) = σ⁻¹ℯ^{-½|μ|²/σ²}$, the inverse slope at the origin is

        .. math:: (T⁻¹)'(0) = σℯ^{½|μ|²/σ²}.

        The transport only depends on $|μ|$. The tails satisfy
        $T(x, μ, σ) ≈ (x-\operatorname{sign}(x)|μ|)/σ$, so
        $T⁻¹(y, μ, σ) ≈ σy + \operatorname{sign}(y)|μ|$.
        """
        MAXITER: Final[int] = 10

        m = mu.abs()
        lower = sigma * y - m
        upper = sigma * y + m
        x = _gaussian_to_bimodal_guess(y, mu, sigma)

        for _ in range(MAXITER):
            x = torch.clamp(x, lower, upper)
            fx, z_minus, z_plus = _bimodal_to_gaussian_forward(x, mu, sigma)
            d_fx = _bimodal_to_gaussian_x_derivative(fx, z_minus, z_plus, mu, sigma)
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
        fx, z_minus, z_plus = _bimodal_to_gaussian_forward(x, mu, sigma)

        ctx.save_for_backward(fx, z_minus, z_plus, mu, sigma)
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
        fx, z_minus, z_plus, mu, sigma = ctx.saved_tensors
        d_x, d_mu, d_sigma = _bimodal_to_gaussian_derivatives(
            fx, z_minus, z_plus, mu, sigma
        )
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

    If $F_p$ is the CDF of the mixture and $Φ$ is the standard normal CDF, then

    .. math::
        y = Φ⁻¹(F_p(x))
          = Φ⁻¹\left(∑ₖ ωₖ Φ\left((x-μₖ)/σₖ\right)\right).

    Numerically, we evaluate the mixture CDF in log space and switch between the
    lower-tail and upper-tail representations to avoid cancellation near $0$ and $1$.

    With $zₖ = (x-μₖ)/σₖ$, the derivatives are

    .. math::
        ∂y/∂x &= ∑ₖ (ωₖ/σₖ) ℯ^{½(y²-zₖ²)}, \\
        ∂y/∂ωₖ &= \sqrt{2π} ℯ^{½y²} Φ(zₖ), \\
        ∂y/∂μₖ &= -(ωₖ/σₖ) ℯ^{½(y²-zₖ²)}, \\
        ∂y/∂σₖ &= -(ωₖ zₖ/σₖ) ℯ^{½(y²-zₖ²)}.
    """

    @staticmethod
    @torch.no_grad()
    def forward(
        ctx, y: Tensor, weights: Tensor, mus: Tensor, sigmas: Tensor, /
    ) -> Tensor:
        assert weights.shape[0] == mus.shape[0] == sigmas.shape[0]
        u, z = _mixture_to_gaussian_forward(y, weights, mus, sigmas)
        ctx.save_for_backward(z, u, weights, sigmas)
        return u

    @staticmethod
    def backward(ctx, *outer: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        r"""Differentiate the explicit mixture-to-Gaussian transport map."""
        (g,) = outer
        z, y, weights, sigmas = ctx.saved_tensors
        d_values, d_weights, d_mus, d_sigmas = _mixture_to_gaussian_derivatives(
            y, z, weights, sigmas
        )

        grad_values = g * d_values
        grad_weights = torch.einsum("..., ...k -> k", g, d_weights)
        grad_mus = torch.einsum("..., ...k -> k", g, d_mus)
        grad_sigmas = torch.einsum("..., ...k -> k", g, d_sigmas)

        # Project weight gradient onto the simplex tangent space.
        # ∆ⁿ = {x∈ℝⁿ⁺¹ : ∑ₖxₖ = 0, xₖ≥0}
        # 𝓣ₓ∆ⁿ = {v∈ℝⁿ⁺¹ : ∑ₖvₖ = 0} is the tangent space of the simplex at x.
        # proj(g) = g - ⟨𝟏∣g⟩ / ⟨𝟏∣𝟏⟩ * 𝟏 = g - mean(g) * 𝟏
        grad_weights = grad_weights - grad_weights.mean(dim=-1, keepdim=True)

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
            fy, z = _mixture_to_gaussian_forward(x, weights, mus, sigmas)
            d_fy = _mixture_to_gaussian_x_derivative(fy, z, weights, sigmas)
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
        fy, z = _mixture_to_gaussian_forward(x, weights, mus, sigmas)

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
            y, z, weights, sigmas
        )
        grad_y = g * d_x.reciprocal()
        grad_weights = torch.einsum("..., ...k -> k", grad_y, -d_weights)
        grad_mus = torch.einsum("..., ...k -> k", grad_y, -d_mus)
        grad_sigmas = torch.einsum("..., ...k -> k", grad_y, -d_sigmas)

        # Project weight gradient onto the simplex tangent space.
        # ∆ⁿ = {x∈ℝⁿ⁺¹ : ∑ₖxₖ = 0, xₖ≥0}
        # 𝓣ₓ∆ⁿ = {v∈ℝⁿ⁺¹ : ∑ₖvₖ = 0} is the tangent space of the simplex at x.
        # proj(g) = g - ⟨𝟏∣g⟩ / ⟨𝟏∣𝟏⟩ * 𝟏 = g - mean(g) * 𝟏
        grad_weights = grad_weights - grad_weights.mean(dim=-1, keepdim=True)

        return grad_y, grad_weights, grad_mus, grad_sigmas


def gaussian_to_bimodal(
    y: Tensor, /, mu: Tensor | float = 2.0, sigma: Tensor | float = 1.0
) -> Tensor:
    r"""Map $N(0,1)$ to the symmetric mixture $½N(-μ,σ²) + ½N(μ,σ²)$.

    This is the inverse of

    .. math:: y = Φ⁻¹\Bigl(½Φ((x+μ)/σ) + ½Φ((x-μ)/σ)\Bigr)

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
    """
    mu = torch.as_tensor(mu, dtype=x.dtype, device=x.device)
    sigma = torch.as_tensor(sigma, dtype=x.dtype, device=x.device)
    return _BimodalToGaussianImpl.apply(x, mu, sigma)


def gaussian_to_mixture(
    y: Tensor, /, weights: Tensor, mus: Tensor, sigmas: Tensor
) -> Tensor:
    r"""Map $N(0,1)$ to the mixture $∑ₖ ωₖ N(μₖ,σₖ²)$.

    This is the inverse of

    .. math::  y = Φ⁻¹\Bigl(∑ₖ ωₖΦ((x-μₖ)/σₖ)\Bigr)

    so the returned value is the unique $x$ whose mixture CDF equals $Φ(y)$.
    """
    return _GaussianToMixture.apply(y, weights, mus, sigmas)


def mixture_to_gaussian(
    x: Tensor, /, weights: Tensor, mus: Tensor, sigmas: Tensor
) -> Tensor:
    r"""Map the mixture $∑ₖ ωₖ N(μₖ,σₖ²)$ to $N(0,1)$.

    .. math::  y = Φ⁻¹\Bigl(∑ₖ ωₖΦ((x-μₖ)/σₖ)\Bigr)
    """
    return _MixtureToGaussian.apply(x, weights, mus, sigmas)
