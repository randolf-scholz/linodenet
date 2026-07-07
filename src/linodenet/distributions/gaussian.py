r"""Gaussian Distribution.

Note:
    When marginalizing a parameterized distribution, it is often not enough that
    the distribution itself is analytically marginalizable. This is because,
    many distributions require specific constraints on the parameters such as
    for example that the covariance matrix is positive definite.

    However, we often use unconstrained weight tensor $w$, and only obtain the actual
    parameters $θ$ by applying a transformation $θ = f(w)$ called the parametrization.

    Now, marginalizing the distribution requires updating the parameters $θ$,
    but actually, we need to update the weights $w$ and then apply the parametrization.

    In essence, this means we need to be able to find a mapping $w → w'$ such that the
    following diagram commutes::

            parametrizations
        w ──────────────► θ
        │                 │
        │                 │m
        │φ                │a
        │                 │r
        │                 │
        ▼   parametrizations   ▼
        w'──────────────► θ'

    I.e. we need to find a mapping $φ$ on the unconstrained parameters such that
    $f(φ(w)) = m(f(w))$ where $m$ is the marginalization operation on the
    parameters.
"""

__all__ = [
    "argmin_proximal_kl",
    "fisher",
    "inverse_fisher",
    "kl",
    "log_prob",
    "multivariate_gaussian_log_likelihood",
    "MultivariateNormal",
    "MultiHeadGaussian",
    "CovarianceType",
]

import math
from collections.abc import Callable
from enum import StrEnum
from typing import Final, Optional, Self, assert_never

import torch
import torch.nn.functional as F
from torch import Tensor, distributions as dist, nn
from torch.linalg import cholesky, solve_triangular, vecdot

from .base import DistributionBase

type GaussianParams = tuple[Tensor, Tensor]


class CovarianceType(StrEnum):
    r"""Gaussian parametrizations."""

    COVARIANCE = "covariance"  # Σ ⪰ 0
    PRECISION = "precision"  # Λ ⪰ 0, Σ=Λ⁻¹
    CHOLESKY = "cholesky"  # L lower triangular, diag(L) > 0, Σ=LLᵀ
    LOG_CHOLESKY = "log-cholesky"  # L lower triangular, diag(L) holds logvals, Σ=LLᵀ


def log_prob(
    x: Tensor,  # (*S, ..., D)
    theta: GaussianParams,  # (..., D), (..., D, D)
    /,
    *,
    parametrization: str = "covariance",
) -> Tensor:  # (*S, ...)
    r"""Compute the Gaussian log-density at `x` in the chosen parametrization."""
    match CovarianceType(parametrization):
        case CovarianceType.COVARIANCE:
            mean, covariance = theta
            residual = x - mean
            chol = cholesky(covariance)
            whitened = solve_triangular(
                chol,
                residual.unsqueeze(-1),
                upper=False,
            ).squeeze(-1)
            logdet = 2 * chol.diagonal(dim1=-2, dim2=-1).log().sum(dim=-1)
            mahalanobis = vecdot(whitened, whitened, dim=-1)
            dim = residual.shape[-1]
            return -0.5 * (dim * math.log(2 * math.pi) + logdet + mahalanobis)
        case CovarianceType.PRECISION:
            mean, precision = theta
            residual = x - mean
            chol = cholesky(precision)
            projected = (chol.mT @ residual.unsqueeze(-1)).squeeze(-1)
            logdet = 2 * chol.diagonal(dim1=-2, dim2=-1).log().sum(dim=-1)
            mahalanobis = vecdot(projected, projected, dim=-1)
            dim = residual.shape[-1]
            return 0.5 * (logdet - dim * math.log(2 * math.pi) - mahalanobis)
        case CovarianceType.CHOLESKY:
            mean, chol = theta
            residual = x - mean
            whitened = solve_triangular(
                chol,
                residual.unsqueeze(-1),
                upper=False,
            ).squeeze(-1)
            logdet = 2 * chol.diagonal(dim1=-2, dim2=-1).log().sum(dim=-1)
            mahalanobis = vecdot(whitened, whitened, dim=-1)
            dim = residual.shape[-1]
            return -0.5 * (dim * math.log(2 * math.pi) + logdet + mahalanobis)
        case CovarianceType.LOG_CHOLESKY:
            mean, log_chol = theta
            residual = x - mean
            chol = log_chol.tril(diagonal=-1) + torch.diag_embed(
                log_chol.diagonal(dim1=-2, dim2=-1).exp()
            )
            whitened = solve_triangular(
                chol,
                residual.unsqueeze(-1),
                upper=False,
            ).squeeze(-1)
            logdet = 2 * log_chol.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
            mahalanobis = vecdot(whitened, whitened, dim=-1)
            dim = residual.shape[-1]
            return -0.5 * (dim * math.log(2 * math.pi) + logdet + mahalanobis)
        case other:
            assert_never(other)  # pyrefly: ignore[bad-argument-type]


def argmin_proximal_kl(
    fun: Callable[[GaussianParams], Tensor],
    theta: GaussianParams,
    /,
    *,
    gamma: float | Tensor = 1.0,
    parametrization: str = "covariance",
) -> GaussianParams:
    r"""Return the Gaussian KL-proximal minimizer in the chosen parametrization.

    This returns the solution of

    .. math:: \argmin_θ f(θ⁎) + ⟨∇f(θ⁎), θ - θ⁎⟩ + γ\kl(p(x∣θ) ∣ p(x∣θ⁎))

    where $θ$ is interpreted according to `parametrization` and $p(x∣θ) = 𝓝(μ, Σ)$.

    The implementation computes $∇f(θ⁎)$ with `torch.func.grad`
    and evaluates the exact closed-form minimizer in covariance,
    precision, Cholesky, or log-Cholesky coordinates.

    Args:
        fun: Scalar objective function to linearize at `theta`.
        theta: Linearization point $θ⁎$ in the selected parametrization.
        gamma: KL regularization strength.
        parametrization: One of `"covariance"`, `"precision"`, `"cholesky"` or `"log-cholesky"`.
    """
    mean, matrix = theta
    gamma = torch.as_tensor(gamma, dtype=matrix.dtype, device=matrix.device)
    scale = gamma.reciprocal()

    # (g, G) = ∇_θ f(θ⁎) where g is the mean gradient
    # and G is the covariance/precision/Cholesky gradient.
    g, G = torch.func.grad(fun)(theta)

    match CovarianceType(parametrization):
        case CovarianceType.COVARIANCE:
            # μ' = μ - γ⁻¹Σg,  Σ'⁻¹ = Σ⁻¹ + 2γ⁻¹ sym(G).
            covariance = matrix
            grad_covariance = 0.5 * (G + G.mT)
            mean_posterior = mean - torch.einsum(
                "...ij, ...j -> ...i", covariance, g * scale
            )

            chol_prior = cholesky(covariance)
            precision_prior = torch.cholesky_inverse(chol_prior)
            candidate_precision = 0.5 * (
                precision_prior
                + 2 * grad_covariance * scale
                + (precision_prior + 2 * grad_covariance * scale).mT
            )

            try:
                chol_precision = cholesky(candidate_precision)
            except RuntimeError as error:
                raise ValueError(
                    "The covariance-parametrized proximal Gaussian update does "
                    "not admit a finite minimizer. Try increasing gamma or "
                    "regularizing the covariance gradient."
                ) from error

            covariance_posterior = torch.cholesky_inverse(chol_precision)
            return mean_posterior, covariance_posterior

        case CovarianceType.PRECISION:
            # μ' = μ - γ⁻¹Σg,  Λ' = L U diag(2/(1 + √(1 + 8b/γ))) UᵀLᵀ
            # where sym(LᵀGL) = U diag(b) Uᵀ and Λ = LLᵀ.
            precision = matrix
            grad_precision = 0.5 * (G + G.mT)
            chol_prior = cholesky(precision)
            mean_posterior = mean - torch.cholesky_solve(
                (g * scale).unsqueeze(-1), chol_prior
            ).squeeze(-1)

            whitened_gradient = chol_prior.mT @ grad_precision @ chol_prior
            whitened_gradient = 0.5 * (whitened_gradient + whitened_gradient.mT)
            eigenvalues, V = torch.linalg.eigh(whitened_gradient)
            tolerance = (
                16
                * torch.finfo(eigenvalues.dtype).eps
                * eigenvalues.abs().amax(dim=-1, keepdim=True).clamp_min(1)
            )
            if torch.any(eigenvalues < -tolerance).item():
                raise ValueError(
                    "The precision-parametrized proximal Gaussian update does "
                    "not admit a finite minimizer. Regularize the precision "
                    "gradient or use the Cholesky parametrization."
                )

            eigenvalues = eigenvalues.clamp_min(0)
            spectral_scale = 2 / (1 + torch.sqrt(1 + 8 * eigenvalues * scale))
            whitened_precision = V @ torch.diag_embed(spectral_scale) @ V.mT
            precision_posterior = chol_prior @ whitened_precision @ chol_prior.mT
            precision_posterior = 0.5 * (precision_posterior + precision_posterior.mT)
            return mean_posterior, precision_posterior

        case CovarianceType.CHOLESKY:
            # μ' = μ - γ⁻¹Σg,  L' = LM with
            # M = tril(-LᵀG/γ, -1) + diag((-a + √(a² + 4γ²))/(2γ)),
            # a = diag(LᵀG).
            chol = matrix
            grad_cholesky = torch.tril(G)
            cov = chol @ chol.mT
            mean_posterior = mean - torch.einsum("...ij, ...j -> ...i", cov, g * scale)

            whitened_gradient = chol.mT @ grad_cholesky
            diagonal_gradient = whitened_gradient.diagonal(dim1=-2, dim2=-1)
            diagonal_update = (
                0.5
                * scale
                * (
                    -diagonal_gradient
                    + torch.sqrt(diagonal_gradient.square() + 4 * gamma.square())
                )
            )
            whitened_cholesky = torch.tril(
                -whitened_gradient * scale, diagonal=-1
            ) + torch.diag_embed(diagonal_update)
            cholesky_posterior = torch.tril(chol @ whitened_cholesky)
            return mean_posterior, cholesky_posterior

        case CovarianceType.LOG_CHOLESKY:
            # μ' = μ - γ⁻¹Σg,  X' = logchol(LW) with
            # Wᵢⱼ = -(LᵀGₗ)ᵢⱼ/γ for i>j and
            # wᵢᵢ = (-aᵢ + √(aᵢ² + 4γ(γ-gᵢ)))/(2γ),
            # a = diag(LᵀGₗ), g = diag(G), Gₗ = tril(G, -1).
            log_chol = matrix
            chol = log_chol.tril(diagonal=-1) + torch.diag_embed(
                log_chol.diagonal(dim1=-2, dim2=-1).exp()
            )
            cov = chol @ chol.mT
            mean_posterior = mean - torch.einsum("...ij,...j->...i", cov, g * scale)

            grad_log_cholesky = torch.tril(G)
            grad_off_diagonal = torch.tril(grad_log_cholesky, diagonal=-1)
            diagonal_gradient = grad_log_cholesky.diagonal(dim1=-2, dim2=-1)
            linear_term = chol.mT @ grad_off_diagonal
            diagonal_linear = linear_term.diagonal(dim1=-2, dim2=-1)

            eps = torch.finfo(chol.dtype).eps
            tolerance = (
                16
                * eps
                * (
                    gamma.abs() + diagonal_gradient.abs() + diagonal_linear.abs()
                ).clamp_min(1)
            )
            if torch.any(diagonal_gradient > gamma + tolerance).item():
                raise ValueError(
                    "The log-Cholesky-parametrized proximal Gaussian update does "
                    "not admit a finite minimizer. Try increasing gamma or "
                    "regularizing the diagonal log-Cholesky gradient."
                )

            radicand = diagonal_linear.square() + 4 * gamma * (
                gamma - diagonal_gradient
            )
            diagonal_update = (
                0.5 * scale * (-diagonal_linear + torch.sqrt(radicand.clamp_min(0)))
            )
            if torch.any(diagonal_update <= tolerance).item():
                raise ValueError(
                    "The log-Cholesky-parametrized proximal Gaussian update does "
                    "not admit a finite minimizer. Try increasing gamma or "
                    "regularizing the diagonal log-Cholesky gradient."
                )

            whitened_cholesky = torch.tril(
                -linear_term * scale, diagonal=-1
            ) + torch.diag_embed(diagonal_update)
            cholesky_posterior = torch.tril(chol @ whitened_cholesky)
            log_cholesky_posterior = cholesky_posterior.tril(
                diagonal=-1
            ) + torch.diag_embed(cholesky_posterior.diagonal(dim1=-2, dim2=-1).log())
            return mean_posterior, log_cholesky_posterior

        case _:
            raise ValueError(
                "Expected parametrization to be one of "
                "{'covariance', 'precision', 'cholesky', 'log_cholesky'}, "
                f"got {parametrization!r}."
            )


def fisher(
    theta: GaussianParams,
    tangent: GaussianParams,
    /,
    *,
    parametrization: str = "covariance",
) -> GaussianParams:
    r"""Apply the Gaussian Fisher/KL metric in the chosen parametrization.

    .. math::
            F_{(μ, Σ)}(δμ, δΣ) &= (Σ⁻¹δμ, ½Σ⁻¹\sym(δΣ)Σ⁻¹)
        \\  F_{(μ, Λ)}(δμ, δΛ) &= (Λδμ, ½Σ\sym(δΛ)Σ), \qquad Σ = Λ⁻¹
        \\  F_{(μ, L)}(δμ, δL) &= (Σ⁻¹δμ, L⁻ᵀ(L⁻¹δL + \diag(L⁻¹δL))), \qquad Σ = LLᵀ
        \\  F_{(μ, X)}(δμ, δX) &= (Σ⁻¹δμ, Jₓᵀ F_{(μ, L)}(JₓδX)),
            \qquad L = \tril(X, -1) + \diag(e^{\diag X})

    Args:
        theta: Gaussian parameters in the selected parametrization.
        tangent: Tangent/cotangent-like direction to which the metric is applied.
        parametrization: One of `"covariance"`, `"precision"`, `"cholesky"` or `"log-cholesky"`.
    """
    match CovarianceType(parametrization):
        case CovarianceType.COVARIANCE:
            # F(δμ, δΣ) = (Σ⁻¹δμ, ½Σ⁻¹ sym(δΣ) Σ⁻¹).
            _, covariance = theta
            tangent_mean, tangent_covariance = tangent
            precision = torch.cholesky_inverse(cholesky(covariance))
            return (
                torch.einsum("...ij,...j->...i", precision, tangent_mean),
                0.25
                * precision
                @ (tangent_covariance + tangent_covariance.mT)
                @ precision,
            )

        case CovarianceType.PRECISION:
            # F(δμ, δΛ) = (Λδμ, ½Σ sym(δΛ) Σ), where Σ = Λ⁻¹.
            _, precision = theta
            tangent_mean, tangent_precision = tangent
            covariance = torch.cholesky_inverse(cholesky(precision))
            return (
                torch.einsum("...ij,...j->...i", precision, tangent_mean),
                0.25
                * covariance
                @ (tangent_precision + tangent_precision.mT)
                @ covariance,
            )

        case CovarianceType.CHOLESKY:
            # F(δμ, δL) = (Σ⁻¹δμ, L⁻ᵀ(L⁻¹δL + diag(L⁻¹δL))) for lower-triangular δL.
            _, chol = theta
            tangent_mean, tangent_cholesky = tangent
            precision = torch.cholesky_inverse(chol)
            whitened = solve_triangular(chol, tangent_cholesky, upper=False)
            diag = torch.diag_embed(whitened.diagonal(dim1=-2, dim2=-1))
            return (
                torch.einsum("...ij,...j->...i", precision, tangent_mean),
                torch.cholesky_solve(tangent_cholesky + chol @ diag, chol),
            )

        case CovarianceType.LOG_CHOLESKY:
            # F(δμ, δX) = (Σ⁻¹δμ, JₓᵀF_L(JₓδX)),
            # JₓδX = tril(δX, -1) + diag(Lᵢᵢ δxᵢᵢ).
            mean, log_chol = theta
            tangent_mean, tangent_log_cholesky = tangent
            chol = log_chol.tril(diagonal=-1) + torch.diag_embed(
                log_chol.diagonal(dim1=-2, dim2=-1).exp()
            )
            diagonal = chol.diagonal(dim1=-2, dim2=-1)
            tangent_cholesky = tangent_log_cholesky.tril(
                diagonal=-1
            ) + torch.diag_embed(
                diagonal * tangent_log_cholesky.diagonal(dim1=-2, dim2=-1)
            )
            cotangent_mean, cotangent_cholesky = fisher(
                (mean, chol),
                (tangent_mean, tangent_cholesky),
                parametrization="cholesky",
            )
            return (
                cotangent_mean,
                cotangent_cholesky.tril(diagonal=-1)
                + torch.diag_embed(
                    diagonal * cotangent_cholesky.diagonal(dim1=-2, dim2=-1)
                ),
            )

        case _:
            raise ValueError(
                "Expected parametrization to be one of "
                "{'covariance', 'precision', 'cholesky', 'log-cholesky'}, "
                f"got {parametrization!r}."
            )


def inverse_fisher(
    theta: GaussianParams,
    cotangent: GaussianParams,
    /,
    *,
    parametrization: str = "covariance",
) -> GaussianParams:
    r"""Apply the inverse Gaussian Fisher/KL metric in the chosen parametrization.

    .. math::
            F_{(μ, Σ)}⁻¹(g, G) &= (Σg, 2Σ\sym(G)Σ)
        \\  F_{(μ, Λ)}⁻¹(g, G) &= (Σg, 2Λ\sym(G)Λ), \qquad Σ = Λ⁻¹
        \\  F_{(μ, L)}⁻¹(g, G) &= (Σg, L(\tril(LᵀG) - ½\diag(LᵀG))), \qquad Σ = LLᵀ
        \\  F_{(μ, X)}⁻¹(g, G) &= (Σg, Jₓ⁻¹F_{(μ, L)}⁻¹(Jₓ^{-ᵀ}G)),
                \qquad L = \tril(X, -1) + \diag(e^{\diag X})

    Args:
        theta: Gaussian parameters in the selected parametrization.
        cotangent: Cotangent-like direction to which the inverse metric is applied.
        parametrization: One of `"covariance"`, `"precision"`, `"cholesky"`, or `"log-cholesky"`.
    """
    match CovarianceType(parametrization):
        case CovarianceType.COVARIANCE:
            # F⁻¹(g, G) = (Σg, 2Σ sym(G) Σ).
            _, covariance = theta
            cotangent_mean, cotangent_covariance = cotangent
            return (
                torch.einsum("...ij,...j->...i", covariance, cotangent_mean),
                covariance
                @ (cotangent_covariance + cotangent_covariance.mT)
                @ covariance,
            )

        case CovarianceType.PRECISION:
            # F⁻¹(g, G) = (Σg, 2Λ sym(G) Λ), where Σ = Λ⁻¹.
            _, precision = theta
            cotangent_mean, cotangent_precision = cotangent
            return (
                torch.einsum(
                    "...ij,...j->...i",
                    torch.cholesky_inverse(cholesky(precision)),
                    cotangent_mean,
                ),
                precision @ (cotangent_precision + cotangent_precision.mT) @ precision,
            )

        case CovarianceType.CHOLESKY:
            # F⁻¹(g, G) = (Σg, L(tril(LᵀG) - ½diag(LᵀG))).
            _, chol = theta
            cotangent_mean, cotangent_cholesky = cotangent
            mixed = chol.mT @ cotangent_cholesky
            diag = torch.diag_embed(mixed.diagonal(dim1=-2, dim2=-1))
            whitened = torch.tril(mixed) - 0.5 * diag
            return (
                torch.einsum("...ij,...j->...i", chol @ chol.mT, cotangent_mean),
                chol @ whitened,
            )

        case CovarianceType.LOG_CHOLESKY:
            # F⁻¹(g, G) = (Σg, Jₓ⁻¹F_L⁻¹(Jₓ^{-ᵀ}G)),
            # Jₓ⁻¹ΔL = tril(ΔL, -1) + diag(ΔLᵢᵢ / Lᵢᵢ).
            mean, log_chol = theta
            cotangent_mean, cotangent_log_cholesky = cotangent
            chol = log_chol.tril(diagonal=-1) + torch.diag_embed(
                log_chol.diagonal(dim1=-2, dim2=-1).exp()
            )
            diagonal = chol.diagonal(dim1=-2, dim2=-1)
            cotangent_cholesky = cotangent_log_cholesky.tril(
                diagonal=-1
            ) + torch.diag_embed(
                cotangent_log_cholesky.diagonal(dim1=-2, dim2=-1) / diagonal
            )
            tangent_mean, tangent_cholesky = inverse_fisher(
                (mean, chol),
                (cotangent_mean, cotangent_cholesky),
                parametrization="cholesky",
            )
            return (
                tangent_mean,
                tangent_cholesky.tril(diagonal=-1)
                + torch.diag_embed(
                    tangent_cholesky.diagonal(dim1=-2, dim2=-1) / diagonal
                ),
            )

        case _:
            raise ValueError(
                "Expected parametrization to be one of "
                "{'covariance', 'precision', 'cholesky', 'log-cholesky'}, "
                f"got {parametrization!r}."
            )


def kl(
    p: tuple[Tensor, Tensor],
    q: tuple[Tensor, Tensor],
    /,
    *,
    parametrization: str = "covariance",
) -> Tensor:
    r"""Return the KL divergence between two Normal Distributions.

    Args:
        p: Gaussian parameters in the selected parametrization.
        q: Gaussian parameters in the selected parametrization.
        parametrization: One of `"covariance"`, `"precision"`, `"cholesky"`, or `"log-cholesky"`.

    Returns:
        The KL divergence `KL(p, q)`.
    """
    match CovarianceType(parametrization):
        case CovarianceType.COVARIANCE:
            # KL = ½(tr(Σᵥ⁻¹Σᵤ) - d + (μᵥ-μᵤ)ᵀΣᵥ⁻¹(μᵥ-μᵤ) + log det Σᵥ - log det Σᵤ).
            mean_p, covariance_p = p
            mean_q, covariance_q = q
            chol_p = cholesky(covariance_p)
            chol_q = cholesky(covariance_q)
            delta = mean_q - mean_p
            # (μᵥ - μᵤ)ᵀΣᵥ⁻¹(μᵥ - μᵤ) = ‖Lᵥ⁻¹(μᵥ - μᵤ)‖².
            whitened = solve_triangular(
                chol_q,
                delta.unsqueeze(-1),
                upper=False,
            ).squeeze(-1)
            mahalanobis = vecdot(whitened, whitened, dim=-1)
            # tr(Σᵥ⁻¹Σᵤ) = ⟨Σᵥ⁻¹, Σᵤ⟩ = ⟨Lᵥ⁻ᵀLᵥ⁻¹, LᵤLᵤᵀ⟩
            #            = ⟨Lᵥ⁻¹Lᵤ, Lᵥ⁻¹Lᵤ⟩ = ‖Lᵥ⁻¹Lᵤ‖².
            trace_term = solve_triangular(chol_q, chol_p, upper=False)
            trace_term = trace_term.square().sum(dim=(-2, -1))
            # log det Σ = 2 ∑ᵢ log Lᵢᵢ for Σ = LLᵀ.
            logdet_p = 2 * chol_p.diagonal(dim1=-2, dim2=-1).log().sum(dim=-1)
            logdet_q = 2 * chol_q.diagonal(dim1=-2, dim2=-1).log().sum(dim=-1)
            dim = mean_p.shape[-1]
            return 0.5 * (trace_term + mahalanobis - dim + logdet_q - logdet_p)

        case CovarianceType.PRECISION:
            # KL = ½(tr(ΛᵥΛᵤ⁻¹) - d + (μᵥ-μᵤ)ᵀΛᵥ(μᵥ-μᵤ) + log det Λᵤ - log det Λᵥ).
            mean_p, precision_p = p
            mean_q, precision_q = q
            chol_p = cholesky(precision_p)
            chol_q = cholesky(precision_q)
            delta = mean_q - mean_p
            # (μᵥ - μᵤ)ᵀΛᵥ(μᵥ - μᵤ) = ‖Lᵥᵀ(μᵥ - μᵤ)‖² for Λᵥ = LᵥLᵥᵀ.
            projected = (chol_q.mT @ delta.unsqueeze(-1)).squeeze(-1)
            mahalanobis = vecdot(projected, projected, dim=-1)
            # tr(ΛᵥΛᵤ⁻¹) = ⟨Λᵥ, Σᵤ⟩ = ⟨LᵥLᵥᵀ, Lᵤ⁻ᵀLᵤ⁻¹⟩
            #            = ⟨Lᵤ⁻¹Lᵥ, Lᵤ⁻¹Lᵥ⟩ = ‖Lᵤ⁻¹Lᵥ‖².
            trace_term = solve_triangular(chol_p, chol_q, upper=False)
            trace_term = trace_term.square().sum(dim=(-2, -1))
            # log det Λ = 2 ∑ᵢ log Lᵢᵢ for Λ = LLᵀ.
            logdet_p = 2 * chol_p.diagonal(dim1=-2, dim2=-1).log().sum(dim=-1)
            logdet_q = 2 * chol_q.diagonal(dim1=-2, dim2=-1).log().sum(dim=-1)
            dim = mean_p.shape[-1]
            return 0.5 * (trace_term + mahalanobis - dim + logdet_p - logdet_q)

        case CovarianceType.CHOLESKY:
            # KL = ½(‖Lᵥ⁻¹Lᵤ‖² + ‖Lᵥ⁻¹(μᵥ-μᵤ)‖² - d + log det Σᵥ - log det Σᵤ).
            mean_p, chol_p = p
            mean_q, chol_q = q
            delta = mean_q - mean_p
            # (μᵥ - μᵤ)ᵀΣᵥ⁻¹(μᵥ - μᵤ) = ‖Lᵥ⁻¹(μᵥ - μᵤ)‖².
            whitened = solve_triangular(
                chol_q,
                delta.unsqueeze(-1),
                upper=False,
            ).squeeze(-1)
            mahalanobis = vecdot(whitened, whitened, dim=-1)
            # tr(Σᵥ⁻¹Σᵤ) = ⟨Σᵥ⁻¹, Σᵤ⟩ = ⟨Lᵥ⁻ᵀLᵥ⁻¹, LᵤLᵤᵀ⟩
            #            = ⟨Lᵥ⁻¹Lᵤ, Lᵥ⁻¹Lᵤ⟩ = ‖Lᵥ⁻¹Lᵤ‖².
            trace_term = solve_triangular(chol_q, chol_p, upper=False)
            trace_term = trace_term.square().sum(dim=(-2, -1))
            # log det Σ = 2 ∑ᵢ log Lᵢᵢ for Σ = LLᵀ.
            logdet_p = 2 * chol_p.diagonal(dim1=-2, dim2=-1).log().sum(dim=-1)
            logdet_q = 2 * chol_q.diagonal(dim1=-2, dim2=-1).log().sum(dim=-1)
            dim = mean_p.shape[-1]
            return 0.5 * (trace_term + mahalanobis - dim + logdet_q - logdet_p)

        case CovarianceType.LOG_CHOLESKY:
            # KL = KL((μ, L(X)) \| (ν, L(Y))) with
            # L(X) = tril(X, -1) + diag(exp(diag(X))).
            mean_p, log_chol_p = p
            mean_q, log_chol_q = q
            chol_p = log_chol_p.tril(diagonal=-1) + torch.diag_embed(
                log_chol_p.diagonal(dim1=-2, dim2=-1).exp()
            )
            chol_q = log_chol_q.tril(diagonal=-1) + torch.diag_embed(
                log_chol_q.diagonal(dim1=-2, dim2=-1).exp()
            )
            delta = mean_q - mean_p
            # (μᵥ - μᵤ)ᵀΣᵥ⁻¹(μᵥ - μᵤ) = ‖Lᵥ⁻¹(μᵥ - μᵤ)‖².
            whitened = solve_triangular(
                chol_q,
                delta.unsqueeze(-1),
                upper=False,
            ).squeeze(-1)
            mahalanobis = vecdot(whitened, whitened, dim=-1)
            # tr(Σᵥ⁻¹Σᵤ) = ⟨Σᵥ⁻¹, Σᵤ⟩ = ⟨Lᵥ⁻ᵀLᵥ⁻¹, LᵤLᵤᵀ⟩
            #            = ⟨Lᵥ⁻¹Lᵤ, Lᵥ⁻¹Lᵤ⟩ = ‖Lᵥ⁻¹Lᵤ‖².
            trace_term = solve_triangular(chol_q, chol_p, upper=False)
            trace_term = trace_term.square().sum(dim=(-2, -1))
            # log det Σ = 2 ∑ᵢ log Lᵢᵢ for Σ = LLᵀ.
            logdet_p = 2 * log_chol_p.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
            logdet_q = 2 * log_chol_q.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
            dim = mean_p.shape[-1]
            return 0.5 * (trace_term + mahalanobis - dim + logdet_q - logdet_p)

        case _:
            raise ValueError(
                "Expected parametrization to be one of "
                "{'covariance', 'precision', 'cholesky', 'log-cholesky'}, "
                f"got {parametrization!r}."
            )


def multivariate_gaussian_log_likelihood(
    value: Tensor,
    /,
    *,
    mean: Tensor,
    covariance_matrix: Tensor,
) -> Tensor:
    r"""Return the log-likelihood of a multivariate Gaussian.

    Args:
        value: Evaluation point $x$ with shape `(..., d)`.
        mean: Mean $μ$ with shape `(..., d)`.
        covariance_matrix: Covariance $Σ$ with shape `(..., d, d)`.

    Returns:
        The log-density

        .. math:: \log 𝓝(x; μ, Σ) = -½(d\log(2π) + \log\det Σ + (x-μ)ᵀΣ⁻¹(x-μ))
    """
    # Factor Σ = LLᵀ so the Mahalanobis term becomes ‖L⁻¹(x-μ)‖².
    residual = value - mean
    L = cholesky(covariance_matrix)
    whitened = solve_triangular(
        L,
        residual.unsqueeze(-1),
        upper=False,
    ).squeeze(-1)
    dim = residual.shape[-1]
    logdet = 2 * L.diagonal(dim1=-2, dim2=-1).log().sum(dim=-1)
    mahalanobis = vecdot(whitened, whitened, dim=-1)
    return -0.5 * (dim * math.log(2 * math.pi) + logdet + mahalanobis)


class MultivariateNormal(dist.MultivariateNormal):
    r"""Augmented Multivariate Normal distribution.

    We add some utilities to the base class.
    """

    def __add__(self, bias: float | Tensor, /) -> Self:
        r"""Add a tensor to the mean."""
        return self.__class__(
            self.mean + bias,
            self.covariance_matrix,
        )

    def __mul__(self, scale: float | Tensor, /) -> Self:
        r"""Multiply by a tensor."""
        return self.__class__(
            scale * self.mean,
            scale**2 * self.covariance_matrix,
        )

    def __matmul__(self, scale: Tensor, /) -> Self:
        r"""Multiply by a tensor."""
        return self.__class__(
            scale @ self.mean,
            scale @ self.covariance_matrix @ scale.T,
        )


class MultiHeadGaussian(DistributionBase):
    r"""Implements a multi-head Gaussian distribution."""

    normalization_constant: Tensor
    r"""CONST: Normalization constant of a Gaussian distribution."""
    num_heads: Final[int]
    r"""CONST: Shape of heads"""
    num_features: Final[int]
    r"""CONST: Number of features in input."""

    # parameters/buffers
    means: Tensor
    r"""PARAM: Means of the gaussians."""
    scale_tril: Tensor  # shape: (n_gaussians, n_inputs, n_inputs)
    r"""PARAM: Parameters determining the covariances."""

    # non-permanent buffers
    eye: Tensor
    r"""BUFFER: Identity matrix."""
    covs: Tensor
    r"""BUFFER: Covariances of the gaussians."""
    cholesky_factor: Tensor  # shape: (n_gaussians, n_inputs, n_inputs)
    r"""BUFFER: Cholesky factor of the covariance matrix."""
    samples: Tensor
    r"""BUFFER: Stored samples when sampling."""
    latents: Tensor
    r"""BUFFER: Stored latents when evaluating log_probs."""
    log_probs: Tensor
    r"""BUFFER: Stored log_probs when evaluating log_probs."""

    def __init__(
        self,
        n_heads: int,
        n_feats: int,
        *,
        means: Optional[Tensor] = None,
        covs: Optional[Tensor] = None,
    ) -> None:
        super().__init__(batch_shape=(n_heads,), event_shape=(n_feats,))
        # CONSTANTS
        self.num_heads = int(n_heads)
        self.num_features = int(n_feats)
        normalization_constant = (
            0.5 * self.num_features * math.log(2 * math.pi)
        )  # -log (2π)^{-k/2}
        self.register_buffer(
            "normalization_constant", torch.tensor(normalization_constant)
        )
        self.register_buffer("eye", torch.eye(n_feats, dtype=torch.bool))

        # BUFFERS
        self.register_buffer("covs", torch.empty(0), persistent=False)
        self.register_buffer("cholesky_factor", torch.empty(0), persistent=False)
        self.register_buffer("samples", torch.empty(0), persistent=False)
        self.register_buffer("latents", torch.empty(0), persistent=False)
        self.register_buffer("log_probs", torch.empty(0), persistent=False)

        # initialize the means
        self.means = nn.Parameter(
            torch.as_tensor(means)
            if means is not None
            else self.sample_default_means(n_heads, n_feats)
        )
        # initialize the covariances
        self.scale_tril = nn.Parameter(  # not a parameter!
            torch.as_tensor(covs)
            if covs is not None
            else self.sample_default_covs(n_heads, n_feats)
        )

    @staticmethod
    def sample_default_means(n_heads: int, n_feats: int) -> Tensor:
        r"""Sample default means $μᵢ∼𝓝(0,1)$."""
        return torch.randn(n_heads, n_feats)

    @staticmethod
    def sample_default_covs(n_heads: int, n_feats: int) -> Tensor:
        r"""Sample default covariances."""
        return torch.eye(n_feats) + torch.randn(n_heads, n_feats, n_feats) / n_feats

    def get_cholesky(self) -> Tensor:
        r"""Compute cholesky factor of covariance matrix."""
        lower = self.scale_tril.tril()
        diag = lower.diagonal(dim1=-2, dim2=-1)
        # need to make the diagonal positive
        new_diag = F.softplus(diag) + 1e-6  # (M, D)
        # (D, D), (M, D, 1), (M, D, D) -> (M, D, D)
        self.cholesky_factor = torch.where(self.eye, new_diag.unsqueeze(-1), lower)
        return self.cholesky_factor

    def get_covariance(self) -> Tensor:
        r"""Compute covariance matrix from cholesky factor."""
        L = self.get_cholesky()  # M x D x D
        self.covs = torch.einsum("mij,mkj->mik", L, L)
        return self.covs

    def forward(self, x: Tensor, /) -> Tensor:
        r"""Transform $x -> y = Lx + μ$.

        Args:
            x (..., H, D): input tensor

        Returns:
            y (..., H, D): transformed tensor
        """
        L = self.get_cholesky()
        y = self.means + torch.einsum("...mj, mij -> ...mi", x, L)
        return y

    def inverse(self, y: Tensor, /) -> tuple[Tensor, Tensor]:
        r"""Transform $y -> x = L⁻¹(y-μ)$.

        Args:
            y (B, H, D): input tensor

        Returns:
            x (B, H, D): transformed tensor
            ldj (H): log determinant of the Jacobian
        """
        L = self.get_cholesky()

        # compute z = L⁻¹(x-μ)
        y = y - self.means
        y = y.movedim(0, -1)  # (B, H, D) -> (H, D, B)
        # (H, D, D), (H, D, B) -> (H, D, B)
        u = solve_triangular(L, y, upper=False)
        u = u.movedim(-1, 0)  # (H, D, B) -> (B, H, D)

        # compute log |det L⁻¹| = - log |det L| = log ∏ᵢ Lᵢᵢ
        ldj = -L.diagonal(dim1=-2, dim2=-1).log().sum(-1)
        return u, ldj

    def sample(self, size: int | tuple[int, ...] = (), /) -> Tensor:
        r"""Sample from the model.

        Args:
            size (int | tuple[int, ...]): size of the sample

        Returns:
            u (..., H, D): sample
        """
        shape = (size,) if isinstance(size, int) else size
        shape = (*shape, self.num_heads, self.num_features)
        z = torch.randn(*shape, device=self.normalization_constant.device)
        u = self.forward(z)
        self.samples = u  # store buffer for post-hoc analysis
        return u

    def log_prob(self, u: Tensor, /) -> Tensor:
        r"""Compute the log probability of the input.

        Args:
            u (..., H, D): input tensor

        Returns:
            log_prob (..., H): log likelihood
        """
        self.latents = u  # store buffer for post-hoc analysis

        # parse through the gaussians
        z, ldj = self.inverse(u)

        # compute the base log probability
        # ½*log(2π) + ½\log(σ²) + ½‖x-μ‖²/σ² = ½*log(2π) +  ½‖x‖²
        log_prob = self.normalization_constant + 0.5 * vecdot(z, z, dim=-1)  # (..., H)
        log_prob = log_prob - ldj  # (..., H)
        self.log_probs = log_prob  # store buffer for post-hoc analysis
        return log_prob

    def marginalize(self, indices: Tensor, /) -> MultiHeadGaussian:
        r"""Marginalize the distribution over the given indices."""
        # (M, D) -> (M, D), (M, D, D) -> (M, D, D)
        idx = indices.tolist()
        assert len(set(idx)) == len(indices), "Indices must be unique"

        # initialize the marginal model
        marg_model = MultiHeadGaussian(n_feats=len(idx), n_heads=self.num_heads)

        # set the marginal models parameters

        # validate the marginalization
        assert marg_model.means.shape == (self.num_heads, len(idx))
        assert marg_model.covs.shape == (self.num_heads, len(idx), len(idx))
        marg_means = self.means[..., idx]
        marg_covs = self.scale_tril[..., idx, :][..., idx]
        assert torch.allclose(marg_model.means, marg_means)
        assert torch.allclose(marg_model.scale_tril, marg_covs)

        return marg_model
