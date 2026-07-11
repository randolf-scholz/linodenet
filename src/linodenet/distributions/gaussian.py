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
    "argmin_reverse_kl",
    "argmin_proximal_kl",
    "solve_proximal_kl",
    "fisher",
    "inverse_fisher",
    "kl",
    "log_prob",
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
from torch import Tensor, cholesky_inverse, cholesky_solve, distributions as dist, nn
from torch.linalg import cholesky, solve_triangular, vecdot

from .base import DistributionBase

type GaussianParams = tuple[Tensor, Tensor]


_LOG2PI: Final = math.log(2 * math.pi)


class CovarianceType(StrEnum):
    r"""Gaussian parametrizations."""

    COVARIANCE = "covariance"  # Σ ⪰ 0
    PRECISION = "precision"  # Λ ⪰ 0, Σ=Λ⁻¹
    CHOLESKY = "cholesky"  # L lower triangular, diag(L) > 0, Σ=LLᵀ
    LOG_CHOLESKY = "log-cholesky"  # L lower triangular, diag(L) holds logvals, Σ=LLᵀ

    # possible further parametrizatrions:
    # - exp(S), S symmetric  (matrix exp)
    # - diag(σ) + UUᵀ  (low rank perturbation)

    def to_covariance(self, theta: GaussianParams, /) -> GaussianParams:
        r"""Return Gaussian parameters in covariance parametrization."""
        mean, matrix = theta

        match self:
            case CovarianceType.COVARIANCE:
                return mean, matrix

            case CovarianceType.PRECISION:
                return mean, cholesky_inverse(cholesky(matrix))

            case CovarianceType.CHOLESKY:
                return mean, matrix @ matrix.mT

            case CovarianceType.LOG_CHOLESKY:
                chol = matrix.tril(diagonal=-1) + torch.diag_embed(
                    matrix.diagonal(dim1=-2, dim2=-1).exp()
                )
                return mean, chol @ chol.mT

            case other:
                assert_never(other)

    def from_covariance(self, theta: GaussianParams, /) -> GaussianParams:
        r"""Return covariance-parametrized Gaussian parameters in this form."""
        mean, covariance = theta

        match self:
            case CovarianceType.COVARIANCE:
                return mean, covariance

            case CovarianceType.PRECISION:
                return mean, cholesky_inverse(cholesky(covariance))

            case CovarianceType.CHOLESKY:
                return mean, cholesky(covariance)

            case CovarianceType.LOG_CHOLESKY:
                chol = cholesky(covariance)
                return mean, chol.tril(diagonal=-1) + torch.diag_embed(
                    chol.diagonal(dim1=-2, dim2=-1).log()
                )

            case other:
                assert_never(other)


def log_prob(
    x: Tensor,  # (*S, ..., D)
    theta: GaussianParams,  # (..., D), (..., D, D)
    /,
    *,
    parametrization: str = "covariance",
) -> Tensor:  # (*S, ...)
    r"""Compute the Gaussian log-density at `x` in the chosen parametrization."""
    mean, matrix = theta
    residual = x - mean
    dim = residual.shape[-1]

    match CovarianceType(parametrization):
        case CovarianceType.COVARIANCE:
            L = cholesky(matrix)
            z = solve_triangular(L, residual.unsqueeze(-1), upper=False).squeeze(-1)
            logdet = 2 * L.diagonal(dim1=-2, dim2=-1).log().sum(dim=-1)
            mahalanobis = vecdot(z, z, dim=-1)
            return -0.5 * (dim * _LOG2PI + logdet + mahalanobis)

        case CovarianceType.PRECISION:
            L = cholesky(matrix)
            projected = (L.mT @ residual.unsqueeze(-1)).squeeze(-1)
            logdet = 2 * L.diagonal(dim1=-2, dim2=-1).log().sum(dim=-1)
            mahalanobis = vecdot(projected, projected, dim=-1)
            return 0.5 * (logdet - dim * _LOG2PI - mahalanobis)

        case CovarianceType.CHOLESKY:
            L = matrix
            z = solve_triangular(L, residual.unsqueeze(-1), upper=False).squeeze(-1)
            logdet = 2 * L.diagonal(dim1=-2, dim2=-1).log().sum(dim=-1)
            mahalanobis = vecdot(z, z, dim=-1)
            return -0.5 * (dim * _LOG2PI + logdet + mahalanobis)

        case CovarianceType.LOG_CHOLESKY:
            log_chol = matrix
            L = log_chol.tril(diagonal=-1) + torch.diag_embed(
                log_chol.diagonal(dim1=-2, dim2=-1).exp()
            )
            z = solve_triangular(L, residual.unsqueeze(-1), upper=False).squeeze(-1)
            logdet = 2 * log_chol.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
            mahalanobis = vecdot(z, z, dim=-1)
            return -0.5 * (dim * _LOG2PI + logdet + mahalanobis)

        case other:
            assert_never(other)  # pyrefly: ignore[bad-argument-type]


def solve_proximal_kl(
    # (..., d), (..., d, d) -> (..., d), (..., d, d)
    grad_fn: Callable[[GaussianParams], GaussianParams],
    theta: GaussianParams,  # (..., d), (..., d, d)
    /,
    *,
    gamma: float | Tensor = 1.0,
    parametrization: str = "covariance",
) -> GaussianParams:  # (..., d), (..., d, d)
    r"""Return the Gaussian KL-proximal solution in the chosen parametrization."""
    # (g, G) = ∇_θ f(θ⁎) where g is the mean gradient
    # and G is the covariance/precision/Cholesky gradient.
    g, G = grad_fn(theta)

    mu, matrix = theta
    gamma = torch.as_tensor(gamma, dtype=matrix.dtype, device=matrix.device)
    scale = gamma.reciprocal()
    eps = torch.finfo(matrix.dtype).eps

    match CovarianceType(parametrization):
        case CovarianceType.COVARIANCE:
            # μ' = μ - γ⁻¹Σg,  Σ'⁻¹ = Σ⁻¹ + 2γ⁻¹ sym(G).
            cov = matrix
            G = 0.5 * (G + G.mT)  # project gradient
            mu_new = mu - torch.einsum("...ij, ...j -> ...i", cov, g * scale)

            L = cholesky(cov)
            Λ = cholesky_inverse(L)
            Λ_new = Λ + 2 * G * scale
            Λ_new = 0.5 * (Λ_new + Λ_new.mT)

            try:
                Λ_chol = cholesky(Λ_new)
            except RuntimeError as error:
                raise ValueError(
                    "The covariance-parametrized proximal Gaussian update does "
                    "not admit a finite minimizer. Try increasing gamma or "
                    "regularizing the covariance gradient."
                ) from error

            cov_new = cholesky_inverse(Λ_chol)
            return mu_new, cov_new

        case CovarianceType.PRECISION:
            # μ' = μ - γ⁻¹Σg,  Λ' = L U diag(2/(1 + √(1 + 8b/γ))) UᵀLᵀ
            # where sym(LᵀGL) = U diag(b) Uᵀ and Λ = LLᵀ.
            Λ = matrix
            G = 0.5 * (G + G.mT)  # project gradient
            L = cholesky(Λ)
            mu_new = mu - cholesky_solve((g * scale).unsqueeze(-1), L).squeeze(-1)

            local_grad = L.mT @ G @ L
            local_grad = 0.5 * (local_grad + local_grad.mT)
            eigs, V = torch.linalg.eigh(local_grad)
            tolerance = 16 * eps * eigs.abs().amax(dim=-1, keepdim=True).clamp_min(1)

            if torch.any(eigs < -tolerance).item():
                raise ValueError(
                    "The precision-parametrized proximal Gaussian update does "
                    "not admit a finite minimizer. Regularize the precision "
                    "gradient or use the Cholesky parametrization."
                )

            eigs = eigs.clamp_min(0)
            eig_scale = 2 / (1 + torch.sqrt(1 + 8 * eigs * scale))
            local_prec = V @ torch.diag_embed(eig_scale) @ V.mT
            Λ_new = L @ local_prec @ L.mT
            Λ_new = 0.5 * (Λ_new + Λ_new.mT)
            return mu_new, Λ_new

        case CovarianceType.CHOLESKY:
            # μ' = μ - γ⁻¹Σg,  L' = LM with
            # M = tril(-LᵀG/γ, -1) + diag((-a + √(a² + 4γ²))/(2γ)),
            # a = diag(LᵀG).
            L = matrix
            G = G.tril()  # project gradient
            cov = L @ L.mT
            mu_new = mu - torch.einsum("...ij, ...j -> ...i", cov, g * scale)

            local_grad = L.mT @ G
            diag_grad = local_grad.diagonal(dim1=-2, dim2=-1)
            diag_step = (
                0.5
                * scale
                * (-diag_grad + torch.sqrt(diag_grad.square() + 4 * gamma.square()))
            )
            local_chol = torch.diag_embed(diag_step) - scale * local_grad.tril(-1)
            chol_new = torch.tril(L @ local_chol)
            return mu_new, chol_new

        case CovarianceType.LOG_CHOLESKY:
            # μ' = μ - γ⁻¹Σg,  X' = logchol(LW) with
            # Wᵢⱼ = -(LᵀGₗ)ᵢⱼ/γ for i>j and
            # wᵢᵢ = (-aᵢ + √(aᵢ² + 4γ(γ-gᵢ)))/(2γ),
            # a = diag(LᵀGₗ), g = diag(G), Gₗ = tril(G, -1).
            log_chol = matrix
            L = log_chol.tril(diagonal=-1) + torch.diag_embed(
                log_chol.diagonal(dim1=-2, dim2=-1).exp()
            )
            cov = L @ L.mT
            mu_new = mu - torch.einsum("...ij,...j->...i", cov, g * scale)

            g_log_chol = G.tril()  # project gradient
            g_off = torch.tril(g_log_chol, diagonal=-1)
            diag_grad = g_log_chol.diagonal(dim1=-2, dim2=-1)
            lin = L.mT @ g_off
            diag_lin = lin.diagonal(dim1=-2, dim2=-1)

            tolerance = (
                16 * eps * (gamma.abs() + diag_grad.abs() + diag_lin.abs()).clamp_min(1)
            )

            if torch.any(diag_grad > gamma + tolerance).item():
                raise ValueError(
                    "The log-Cholesky-parametrized proximal Gaussian update does "
                    "not admit a finite minimizer. Try increasing gamma or "
                    "regularizing the diagonal log-Cholesky gradient."
                )

            radicand = diag_lin.square() + 4 * gamma * (gamma - diag_grad)
            diag_step = 0.5 * scale * (-diag_lin + torch.sqrt(radicand.clamp_min(0)))

            if torch.any(diag_step <= tolerance).item():
                raise ValueError(
                    "The log-Cholesky-parametrized proximal Gaussian update does "
                    "not admit a finite minimizer. Try increasing gamma or "
                    "regularizing the diagonal log-Cholesky gradient."
                )

            local_chol = torch.diag_embed(diag_step) - scale * lin.tril(-1)
            chol_new = torch.tril(L @ local_chol)
            log_chol_new = chol_new.tril(diagonal=-1) + torch.diag_embed(
                chol_new.diagonal(dim1=-2, dim2=-1).log()
            )
            return mu_new, log_chol_new

        case _:
            raise ValueError(
                "Expected parametrization to be one of "
                "{'covariance', 'precision', 'cholesky', 'log_cholesky'}, "
                f"got {parametrization!r}."
            )


def argmin_proximal_kl(
    loss_fn: Callable[[GaussianParams], Tensor],  # (..., d), (..., d, d) -> (...)
    theta: GaussianParams,  # (..., d), (..., d, d)
    /,
    *,
    gamma: float | Tensor = 1.0,
    parametrization: str = "covariance",
) -> GaussianParams:  # (..., d), (..., d, d)
    r"""Return the Gaussian KL-proximal minimizer in the chosen parametrization.

    This returns the solution of

    .. math:: \argmin_θ f(θ⁎) + ⟨∇f(θ⁎), θ - θ⁎⟩ + γ\kl(p(x∣θ), p(x∣θ⁎))

    where $θ$ is interpreted according to `parametrization` and $p(x∣θ) = 𝓝(μ, Σ)$.

    The implementation computes $∇f(θ⁎)$ with `torch.func.grad`
    and evaluates the exact closed-form minimizer in covariance,
    precision, Cholesky, or log-Cholesky coordinates.

    Args:
        loss_fn: Scalar objective function to linearize at `theta`.
        theta: Linearization point $θ⁎$ in the selected parametrization.
        gamma: KL regularization strength.
        parametrization: One of `"covariance"`, `"precision"`, `"cholesky"` or `"log-cholesky"`.
    """
    return solve_proximal_kl(
        # ∇_θ ∑ ℓ(θᵢ) = (∇_{θ₁} ℓ(θ₁), ..., ∇_{θₙ} ℓ(θₙ))
        torch.func.grad(lambda θ: loss_fn(θ).sum()),
        theta,
        gamma=gamma,
        parametrization=parametrization,
    )


def argmin_reverse_kl(
    z: Tensor,  # (..., d)
    theta: GaussianParams,  # (..., d), (..., d, d)
    /,
    *,
    gamma: Tensor | float,
    parametrization: str = "covariance",
) -> GaussianParams:  # (..., d), (..., d, d)
    r"""Return the exact Gaussian minimizer of NLL plus reverse-KL anchoring.

    .. math:: \argmin_θ -\log 𝓝(z; θ) + γ\kl(𝓝(θ), 𝓝(θ₋))

    where $θ₋$ is the input `theta`, interpreted according to `parametrization`,
    that is by default $θ=(μ, Σ)$.

    The closed-form solution is evaluated in covariance coordinates and then
    mapped back to the requested parametrization. A finite minimizer exists only
    for $γ > 0$.

    Args:
        z: Observation already pulled back to latent space.
        theta: Prior Gaussian parameters in the selected parametrization.
        gamma: Reverse-KL regularization strength, must be positive.
        parametrization: One of `"covariance"`, `"precision"`, `"cholesky"` or `"log-cholesky"`.
    """
    parametrization = CovarianceType(parametrization)
    mean_prior, cov_prior = parametrization.to_covariance(theta)
    gamma = torch.as_tensor(gamma, dtype=cov_prior.dtype, device=cov_prior.device)

    eta = (1 + gamma).reciprocal()
    delta = z - mean_prior
    outer = torch.einsum("...i, ...j -> ...ij", delta, delta)

    mean_post = mean_prior + eta * delta
    cov_post = (1 - eta) * cov_prior + (eta * (1 - eta)) * outer

    return parametrization.from_covariance((mean_post, cov_post))


def fisher(
    theta: GaussianParams,  # (..., d), (..., d, d)
    tangent: GaussianParams,  # (..., d), (..., d, d)
    /,
    *,
    parametrization: str = "covariance",
) -> GaussianParams:  # (..., d), (..., d, d)
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
            _, cov = theta
            d_mu, d_cov = tangent
            prec = cholesky_inverse(cholesky(cov))
            return (
                torch.einsum("...ij,...j->...i", prec, d_mu),
                0.25 * prec @ (d_cov + d_cov.mT) @ prec,
            )

        case CovarianceType.PRECISION:
            # F(δμ, δΛ) = (Λδμ, ½Σ sym(δΛ) Σ), where Σ = Λ⁻¹.
            _, prec = theta
            d_mu, d_prec = tangent
            cov = cholesky_inverse(cholesky(prec))
            return (
                torch.einsum("...ij,...j->...i", prec, d_mu),
                0.25 * cov @ (d_prec + d_prec.mT) @ cov,
            )

        case CovarianceType.CHOLESKY:
            # F(δμ, δL) = (Σ⁻¹δμ, L⁻ᵀ(L⁻¹δL + diag(L⁻¹δL))) for lower-triangular δL.
            _, L = theta
            d_mu, d_chol = tangent
            prec = cholesky_inverse(L)
            local_chol = solve_triangular(L, d_chol, upper=False)
            diag = torch.diag_embed(local_chol.diagonal(dim1=-2, dim2=-1))
            return (
                torch.einsum("...ij,...j->...i", prec, d_mu),
                cholesky_solve(d_chol + L @ diag, L),
            )

        case CovarianceType.LOG_CHOLESKY:
            # F(δμ, δX) = (Σ⁻¹δμ, JₓᵀF_L(JₓδX)),
            # JₓδX = tril(δX, -1) + diag(Lᵢᵢ δxᵢᵢ).
            mean, log_chol = theta
            d_mu, d_log_chol = tangent
            L = log_chol.tril(diagonal=-1) + torch.diag_embed(
                log_chol.diagonal(dim1=-2, dim2=-1).exp()
            )
            diag = L.diagonal(dim1=-2, dim2=-1)
            d_chol = d_log_chol.tril(diagonal=-1) + torch.diag_embed(
                diag * d_log_chol.diagonal(dim1=-2, dim2=-1)
            )
            g_mu, g_chol = fisher(
                (mean, L),
                (d_mu, d_chol),
                parametrization="cholesky",
            )
            return (
                g_mu,
                g_chol.tril(diagonal=-1)
                + torch.diag_embed(diag * g_chol.diagonal(dim1=-2, dim2=-1)),
            )

        case _:
            raise ValueError(
                "Expected parametrization to be one of "
                "{'covariance', 'precision', 'cholesky', 'log-cholesky'}, "
                f"got {parametrization!r}."
            )


def inverse_fisher(
    theta: GaussianParams,  # (..., d), (..., d, d)
    cotangent: GaussianParams,  # (..., d), (..., d, d)
    /,
    *,
    parametrization: str = "covariance",
) -> GaussianParams:  # (..., d), (..., d, d)
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
            _, cov = theta
            g_mu, g_cov = cotangent
            return (
                torch.einsum("...ij,...j->...i", cov, g_mu),
                cov @ (g_cov + g_cov.mT) @ cov,
            )

        case CovarianceType.PRECISION:
            # F⁻¹(g, G) = (Σg, 2Λ sym(G) Λ), where Σ = Λ⁻¹.
            _, prec = theta
            g_mu, g_prec = cotangent
            return (
                torch.einsum(
                    "...ij,...j->...i",
                    cholesky_inverse(cholesky(prec)),
                    g_mu,
                ),
                prec @ (g_prec + g_prec.mT) @ prec,
            )

        case CovarianceType.CHOLESKY:
            # F⁻¹(g, G) = (Σg, L(tril(LᵀG) - ½diag(LᵀG))).
            _, L = theta
            g_mu, g_chol = cotangent
            LtG = L.mT @ g_chol
            diag = torch.diag_embed(LtG.diagonal(dim1=-2, dim2=-1))
            local_chol = torch.tril(LtG) - 0.5 * diag
            return (
                torch.einsum("...ij,...j->...i", L @ L.mT, g_mu),
                L @ local_chol,
            )

        case CovarianceType.LOG_CHOLESKY:
            # F⁻¹(g, G) = (Σg, Jₓ⁻¹F_L⁻¹(Jₓ^{-ᵀ}G)),
            # Jₓ⁻¹ΔL = tril(ΔL, -1) + diag(ΔLᵢᵢ / Lᵢᵢ).
            mean, log_chol = theta
            g_mu, g_log_chol = cotangent
            L = log_chol.tril(diagonal=-1) + torch.diag_embed(
                log_chol.diagonal(dim1=-2, dim2=-1).exp()
            )
            diag = L.diagonal(dim1=-2, dim2=-1)
            g_chol = g_log_chol.tril(diagonal=-1) + torch.diag_embed(
                g_log_chol.diagonal(dim1=-2, dim2=-1) / diag
            )
            d_mu, d_chol = inverse_fisher(
                (mean, L),
                (g_mu, g_chol),
                parametrization="cholesky",
            )
            return (
                d_mu,
                d_chol.tril(diagonal=-1)
                + torch.diag_embed(d_chol.diagonal(dim1=-2, dim2=-1) / diag),
            )

        case _:
            raise ValueError(
                "Expected parametrization to be one of "
                "{'covariance', 'precision', 'cholesky', 'log-cholesky'}, "
                f"got {parametrization!r}."
            )


def kl(
    p: GaussianParams,  # (..., d), (..., d, d)
    q: GaussianParams,  # (..., d), (..., d, d)
    /,
    *,
    parametrization: str = "covariance",
) -> Tensor:  # (...)
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
        normalization_constant = 0.5 * self.num_features * _LOG2PI  # -log (2π)^{-k/2}
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
        log_probs = self.normalization_constant + 0.5 * vecdot(z, z, dim=-1)  # (..., H)
        log_probs = log_probs - ldj  # (..., H)
        self.log_probs = log_probs  # store buffer for post-hoc analysis
        return log_probs

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
