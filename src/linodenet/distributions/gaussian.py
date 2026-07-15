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
    "argmin_forward_kl",
    "fisher",
    "inverse_fisher",
    "kl",
    "log_prob",
    "MultivariateNormal",
    "MultiHeadGaussian",
    "CovarianceType",
]

import math
from enum import StrEnum
from typing import Final, Optional, Self, assert_never

import torch
import torch.nn.functional as F
from torch import Tensor, cholesky_inverse, cholesky_solve, distributions as dist, nn
from torch.linalg import cholesky, solve_triangular, vecdot

from .base import DistributionBase

type GaussianParams = tuple[Tensor, Tensor]
type ScalarLike = Tensor | float


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
                chol = (
                    matrix.tril(diagonal=-1)
                    + matrix.diagonal(dim1=-2, dim2=-1).exp().diag_embed()
                )
                return mean, chol @ chol.mT

            case other:
                assert_never(other)

    def to_precision(self, theta: GaussianParams, /) -> GaussianParams:
        r"""Return the precision parametrization."""
        mean, matrix = theta

        match self:
            case CovarianceType.COVARIANCE:
                return mean, cholesky_inverse(cholesky(matrix))

            case CovarianceType.PRECISION:
                return theta

            case CovarianceType.CHOLESKY:
                return mean, cholesky_inverse(matrix)

            case CovarianceType.LOG_CHOLESKY:
                chol = (
                    matrix.tril(diagonal=-1)
                    + matrix.diagonal(dim1=-2, dim2=-1).exp().diag_embed()
                )
                return mean, cholesky_inverse(chol)

    def to_cholesky(self, theta: GaussianParams, /) -> GaussianParams:
        r"""Return the Cholesky parametrization."""
        mean, matrix = theta

        match self:
            case CovarianceType.COVARIANCE:
                return mean, cholesky(matrix)

            case CovarianceType.PRECISION:
                return mean, cholesky_inverse(cholesky(matrix))

            case CovarianceType.CHOLESKY:
                return theta

            case CovarianceType.LOG_CHOLESKY:
                return (
                    mean,
                    matrix.tril(diagonal=-1)
                    + matrix.diagonal(dim1=-2, dim2=-1).exp().diag_embed(),
                )

            case other:
                assert_never(other)

    def to_log_cholesky(self, theta: GaussianParams, /) -> GaussianParams:
        r"""Return the log-Cholesky parametrization."""
        mean, matrix = theta

        match self:
            case CovarianceType.COVARIANCE:
                chol = cholesky(matrix)
                return CovarianceType.CHOLESKY.to_log_cholesky((mean, chol))

            case CovarianceType.PRECISION:
                chol = cholesky_inverse(cholesky(matrix))
                return CovarianceType.CHOLESKY.to_log_cholesky((mean, chol))

            case CovarianceType.CHOLESKY:
                return (
                    mean,
                    matrix.diagonal(dim1=-2, dim2=-1).log().diag_embed()
                    + matrix.tril(diagonal=-1),
                )

            case CovarianceType.LOG_CHOLESKY:
                return theta

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
                L = cholesky(covariance)
                return (
                    mean,
                    L.tril(-1) + L.diagonal(dim1=-2, dim2=-1).log().diag_embed(),
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
            L = (
                log_chol.tril(diagonal=-1)
                + log_chol.diagonal(dim1=-2, dim2=-1).exp().diag_embed()
            )
            z = solve_triangular(L, residual.unsqueeze(-1), upper=False).squeeze(-1)
            logdet = 2 * log_chol.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
            mahalanobis = vecdot(z, z, dim=-1)
            return -0.5 * (dim * _LOG2PI + logdet + mahalanobis)

        case other:
            assert_never(other)  # pyrefly: ignore[bad-argument-type]


def _solve_w_closed_form(
    sq_dist: Tensor,  # (...)
    retention_mean: Tensor,  # (...)
    retention_cov: Tensor,  # (...)
    /,
    *,
    use_fp64: bool = True,
) -> Tensor:  # (...)
    r"""Solve the reverse-KL stationarity equation in retention coordinates.

    Returns $w = (1 - ρ_μ) + ρ_μ s$, where $s$ is the whitened posterior variance along the
    innovation, i.e. the admissible root of

        (s - ρ_Σ) / ((1 - ρ_Σ)·s) = ρ_μ²·q / ((1 - ρ_μ) + ρ_μ·s)²,   ρ_μ ∈ [0,1], ρ_Σ ∈ (0,1].

    Neither $λ_μ$ nor $λ_Σ$ appears: they cancel under $λ_μ = ρ_μ/(1 - ρ_μ)$,
    $λ_Σ = 1/(1 - ρ_Σ)$, which is what makes the endpoints $ρ_μ = 1$ ($λ_μ = ∞$) and
    $ρ_Σ = 1$ ($λ_Σ = ∞$) finite and exactly representable.

    With $A = (1 - ρ_μ) + ρ_μ ρ_Σ ∈ [ρ_Σ, 1]$, $κ = (1 - ρ_Σ)ρ_μ²q/A²$ and
    $τ = (1 - ρ_μ)/A ∈ [0, 1]$, substituting $w = A·v$ gives the scale-free monic cubic

        v³ - v² - κ·v + κ·τ = 0.

    Its roots are one negative, one in $(0, τ)$ and one $> 1$; only the last gives $s > 0$,
    so the admissible root is the largest. Depressing with $v = t + ⅓$ gives $p = -κ - ⅓ ≤ -⅓$
    unconditionally, so this is always the casus irreducibilis and the root is the $k = 0$
    cosine branch. $A ≥ ρ_Σ > 0$ and $m³ ≥ 1/27$, so neither division needs a guard.

    Note: Degenerate branch
        $\cos θ → 1⁻$ as $κ → 0$ (reached at $ρ_μ = 0$ or $q = 0$) and rounds to $≥ 1$ there,
        where $\arccos'(1) = -∞$. Since `torch.where` backpropagates through the discarded
        arm, a bare clamp would leave $0·(-∞) = \mathrm{NaN}$ in the gradient, so those
        entries take the series $v = 1 + κ(1 - τ) + O(κ²)$ *and* have their cosine argument
        neutralized. Unlike the $λ$ formulation there is no second branch: $s$ is never
        formed, only $ρ_μ s = w - (1 - ρ_μ)$, so nothing divides by $ρ_μ$.
    """
    q, ρ_μ, ρ_Σ = torch.broadcast_tensors(sq_dist, retention_mean, retention_cov)
    out_dtype = torch.promote_types(q.dtype, torch.promote_types(ρ_μ.dtype, ρ_Σ.dtype))
    work_dtype = (
        torch.float64 if use_fp64 else torch.promote_types(out_dtype, torch.float32)
    )

    q = q.to(work_dtype)
    ρ_μ = ρ_μ.to(work_dtype)
    ρ_Σ = ρ_Σ.to(work_dtype)
    eps = torch.finfo(work_dtype).eps

    A = 1.0 - ρ_μ * (1.0 - ρ_Σ)  # = (1 - ρ_μ) + ρ_μ ρ_Σ  ∈ [ρ_Σ, 1]
    κ = (1.0 - ρ_Σ) * ρ_μ.square() * q / A.square()
    τ = (1.0 - ρ_μ) / A  # ∈ [0, 1]

    # Series branch: exact at κ = 0, error O(κ²).
    v_series = 1.0 + κ * (1.0 - τ)

    # Depressed cubic for v³ - v² - κv + κτ via v = t + 1/3.
    p = -κ - 1.0 / 3.0  # ≤ -1/3
    r = -2.0 / 27.0 - κ / 3.0 + κ * τ
    m = torch.sqrt(-p / 3.0)  # ≥ 1/3, so m³ ≥ 1/27
    raw = -r / (2.0 * m.pow(3))

    # acos'(±1) = ∓∞ and `where` backprops through the dead arm: neutralize the argument
    # wherever the cosine sits on its endpoint, and take the series there instead.
    degenerate = (κ < eps**0.5) | (raw >= 1.0)
    cos_θ = torch.where(degenerate, torch.zeros_like(raw), raw).clamp(-1.0, 1.0)
    v_exact = 2.0 * m * torch.cos(torch.acos(cos_θ) / 3.0) + 1.0 / 3.0

    v = torch.where(degenerate, v_series, v_exact)
    return (A * v).to(dtype=out_dtype)


def argmin_reverse_kl(
    z: Tensor,  # (..., d)
    theta: GaussianParams,  # (..., d), (..., d, d)
    /,
    *,
    retention: ScalarLike | tuple[ScalarLike, ScalarLike],  # rho or (rho_mu, rho_sigma)
    parametrization: str = "covariance",
) -> GaussianParams:  # (..., d), (..., d, d)
    r"""Return the exact minimizer of NLL plus reverse-KL regularization term.

    This returns the exact minimizer of

    .. math:: θ₊ = \argmin_θ -\log 𝓝(z; θ) + λ⋅\kl(𝓝(θ) ∥ 𝓝(θ₋)) \\
                &= \argmin_θ -\log 𝓝(z; θ)
                    + λ_μ ½(μ - μ₋)ᵀΣ₋⁻¹(μ - μ₋)
                    + λ_Σ ½(\tr(ΣΣ₋⁻¹) - \log\det(ΣΣ₋⁻¹) - d)
                    \qquad (λ_μ = λ_Σ = λ)

    Here $θ₋$ is the input `theta`, interpreted according to `parametrization`.

    Update:
        Writing $δ = z - μ₋$, $Σ₋ = L₋L₋ᵀ$, $a = L₋⁻¹δ$, $q = aᵀa$, and letting
        $w = (1 - ρ_μ) + ρ_μ s$ where $s > 0$ solves the stationarity equation
        (see `_solve_w_closed_form`), the minimizer shares the prior-whitened eigenspaces
        of $aaᵀ$:

        - $μ₊ = μ₋ + \frac{1 - ρ_μ}{w}δ$
        - $Σ₊ = ρ_Σ⋅Σ₋ + \frac{(1 - ρ_Σ)ρ_μ⋅(ρ_μ s)}{w²}δδᵀ$,
          with $ρ_μ s = w - (1 - ρ_μ)$

        In precision coordinates, with $Λ₋ = Σ₋⁻¹$ and $v = Λ₋δ$,

        - $Λ₊ = ρ_Σ⁻¹Λ₋ - \frac{ρ_μ²(1 - ρ_Σ)}{ρ_Σ w²}vvᵀ$

        In Cholesky coordinates, $L₊ = L₋\chol(ρ_Σ⋅I + \frac{(1-ρ_Σ)ρ_μ(ρ_μ s)}{w²}aaᵀ)$,
        already lower-triangular with positive diagonal.

    Parametrization:
        Weights are supplied as *retentions* $ρ ∈ [0, 1]$, matching `argmin_forward_kl`:

        - $ρ_Σ = 1 - 1/λ_Σ$ is the exact covariance retention — the coefficient multiplying
          $Σ₋$ — so the constraint $λ_Σ > 1$ (below which no minimizer exists) becomes simply
          $ρ_Σ > 0$, isolated where floats are dense, instead of the rounding attractor at
          $λ_Σ = 1$.
        - $ρ_μ = λ_μ/(1 + λ_μ)$ is the *nominal* mean retention. The realized retention is
          $ρ_μ s / w$, satisfying $\mathrm{odds}(\text{realized}) = \mathrm{odds}(ρ_μ)⋅s$:
          $ρ_μ$ is what the mean would retain if the posterior variance along the innovation
          equalled the prior's ($s = 1$). Large innovations inflate $s$, so the mean is
          retained *more* — this is the bounded-influence property of the reverse KL, and it
          is why $ρ_μ$ cannot be an exact retention the way $ρ_Σ$ is.

        Passing a single `retention` sets $ρ_μ = ρ_Σ = ρ$. **This is not $λ_μ = λ_Σ$** — the
        two maps differ, and it corresponds to $λ_μ = λ_Σ - 1$. It does mean both the mean
        (nominally) and the covariance retain the same fraction, which is the useful reading.

    Args:
        z: Observation already pulled back to latent space.
        theta: Prior Gaussian parameters $θ₋$ in the selected parametrization.
        retention: Retention $ρ$, shared or split as $(ρ_μ, ρ_Σ)$. Broadcast against the
            batch shape of `theta`, so a per-sample schedule $ρ(Δt)$ is fine.
        parametrization: One of `"covariance"`, `"precision"`, `"cholesky"`, `"log-cholesky"`.

    Note: Admissible range
        $ρ_μ ∈ [0, 1]$ is closed: $0$ is $λ_μ = 0$, the unregularized jump $μ₊ = z$; $1$ is
        $λ_μ → ∞$, the identity $μ₊ = μ₋$. $ρ_Σ ∈ (0, 1]$: at $1$ the shape is frozen
        ($λ_Σ → ∞$), and $ρ_Σ → 0$ is $λ_Σ → 1⁺$, where $Σ₊$ degenerates. Both endpoints are
        exact, unlike in $λ$ coordinates where neither $∞$ is representable.

    See Also:
        `argmin_forward_kl`:
            Same observation term, opposite KL direction. Its $ρ_Σ = λ_Σ/(1 + λ_Σ)$ — a
            *different* map, since it has no $λ_Σ > 1$ constraint — but the same operational
            meaning (the coefficient multiplying $Σ₋$), so schedules transfer.
    """
    parametrization = CovarianceType(parametrization)
    μ, matrix = theta
    rho_mu, rho_sigma = (
        retention if isinstance(retention, tuple) else (retention, retention)
    )
    ρ_μ = torch.as_tensor(rho_mu, dtype=matrix.dtype, device=matrix.device)
    ρ_Σ = torch.as_tensor(rho_sigma, dtype=matrix.dtype, device=matrix.device)
    assert ((ρ_μ >= 0.0) & (ρ_μ <= 1.0)).all(), "requires rho_mu in [0, 1]"
    assert ((ρ_Σ > 0.0) & (ρ_Σ <= 1.0)).all(), "requires rho_sigma in (0, 1]"

    forget = 1.0 - ρ_Σ
    δ = z - μ
    dim = matrix.shape[-1]

    match parametrization:
        case CovarianceType.COVARIANCE:
            Σ = matrix
            L = cholesky(Σ)
            a = solve_triangular(L, δ.unsqueeze(-1), upper=False).squeeze(-1)
            q = vecdot(a, a, dim=-1)
            w = _solve_w_closed_form(q, ρ_μ, ρ_Σ)
            ρs = w - (1.0 - ρ_μ)  # = ρ_μ · s; no division by ρ_μ
            μ_new = μ + ((1.0 - ρ_μ) / w)[..., None] * δ
            outer = torch.einsum("...i, ...j -> ...ij", δ, δ)
            coefficient = forget * ρ_μ * ρs / w.square()
            Σ_new = ρ_Σ[..., None, None] * Σ + coefficient[..., None, None] * outer
            Σ_new = 0.5 * (Σ_new + Σ_new.mT)  # ensure symmetry
            return μ_new, Σ_new

        case CovarianceType.PRECISION:
            Λ = matrix
            projected = torch.einsum("...ij, ...j -> ...i", Λ, δ)
            q = vecdot(δ, projected, dim=-1)
            w = _solve_w_closed_form(q, ρ_μ, ρ_Σ)
            μ_new = μ + ((1.0 - ρ_μ) / w)[..., None] * δ
            outer = torch.einsum("...i, ...j -> ...ij", projected, projected)
            coefficient = ρ_μ.square() * forget / (ρ_Σ * w.square())
            Λ_new = (
                ρ_Σ.reciprocal()[..., None, None] * Λ
                - coefficient[..., None, None] * outer
            )  # fmt: skip
            Λ_new = 0.5 * (Λ_new + Λ_new.mT)  # ensure symmetry
            return μ_new, Λ_new

        case CovarianceType.CHOLESKY:
            L = matrix
            a = solve_triangular(L, δ.unsqueeze(-1), upper=False).squeeze(-1)
            q = vecdot(a, a, dim=-1)
            w = _solve_w_closed_form(q, ρ_μ, ρ_Σ)
            ρs = w - (1.0 - ρ_μ)
            μ_new = μ + ((1.0 - ρ_μ) / w)[..., None] * δ
            I = torch.eye(dim, dtype=L.dtype, device=L.device)
            outer = torch.einsum("...i, ...j -> ...ij", a, a)
            coefficient = forget * ρ_μ * ρs / w.square()
            local_cov = ρ_Σ[..., None, None] * I + coefficient[..., None, None] * outer
            chol_new = L @ cholesky(local_cov)  # already lower-triangular
            return μ_new, torch.tril(chol_new)

        case CovarianceType.LOG_CHOLESKY:
            log_chol_prior = matrix
            L = (
                log_chol_prior.tril(diagonal=-1)
                + log_chol_prior.diagonal(dim1=-2, dim2=-1).exp().diag_embed()
            )
            a = solve_triangular(L, δ.unsqueeze(-1), upper=False).squeeze(-1)
            q = vecdot(a, a, dim=-1)
            w = _solve_w_closed_form(q, ρ_μ, ρ_Σ)
            ρs = w - (1.0 - ρ_μ)
            μ_new = μ + ((1.0 - ρ_μ) / w)[..., None] * δ
            I = torch.eye(dim, dtype=L.dtype, device=L.device)
            outer = torch.einsum("...i, ...j -> ...ij", a, a)
            coefficient = forget * ρ_μ * ρs / w.square()
            local_cov = ρ_Σ[..., None, None] * I + coefficient[..., None, None] * outer
            chol_new = torch.tril(L @ cholesky(local_cov))
            log_chol_new = (
                chol_new.tril(diagonal=-1)
                + chol_new.diagonal(dim1=-2, dim2=-1).log().diag_embed()
            )
            return μ_new, log_chol_new

        case _:
            raise ValueError(
                "Expected parametrization to be one of "
                "{'covariance', 'precision', 'cholesky', 'log-cholesky'}, "
                f"got {parametrization!r}."
            )


def argmin_forward_kl(
    z: Tensor,  # (..., d)
    theta: GaussianParams,  # (..., d), (..., d, d)
    /,
    *,
    retention: ScalarLike | tuple[ScalarLike, ScalarLike],  # rho or (rho_mu, rho_sigma)
    parametrization: str = "covariance",
) -> GaussianParams:  # (..., d), (..., d, d)
    r"""Return the exact minimizer of NLL plus separable forward-KL anchoring.

    This returns the exact minimizer of

    .. math:: θ₊ = \argmin_θ -\log 𝓝(z; θ) + λ⋅\kl(𝓝(θ₋)，𝓝(θ)) \\
                &= \argmin_θ -\log 𝓝(z; θ)
                    + λ_μ ½(μ - μ₋)ᵀΣ⁻¹(μ - μ₋)
                    + λ_Σ ½(\tr(Σ₋Σ⁻¹) - \log\det(Σ₋Σ⁻¹) - d)
                    \qquad (λ_μ = λ_Σ = λ)

    Here $θ₋$ is the input `theta`, interpreted according to `parametrization`.

    Update:
        With $δ = z - μ₋$, $\text{keep} = ρ_Σ$ and $\text{gain} = (1 - ρ_Σ)ρ_μ$:

        - $μ₊ = ρ_μ μ₋ + (1 - ρ_μ)z$
        - $Σ₊ = \text{keep}⋅Σ₋ + \text{gain}⋅δδᵀ$

        Note $ρ_μδ = z - μ₊$: the covariance is driven by the *posterior* residual, the part
        of the innovation the mean did not absorb. Each parametrization uses its structure:

        - `covariance`: apply directly.
        - `precision`: with $Λ₋ = Σ₋⁻¹$, $v = Λ₋δ$, $q = δᵀv$, Sherman-Morrison gives
          $Λ₊ = \text{keep}⁻¹(Λ₋ - \frac{\text{gain}}{\text{keep} + \text{gain}⋅q}vvᵀ)$.
          The denominator is bounded below by $\text{keep} > 0$, so no guard is needed.
        - `cholesky`: with $Σ₋ = L₋L₋ᵀ$, $a = L₋⁻¹δ$, the factor is
          $L₊ = L₋\chol(\text{keep}⋅I + \text{gain}⋅aaᵀ)$, already lower-triangular.
        - `log-cholesky`: as `cholesky`, then store the diagonal in log form.

    Parametrization:
        Weights are the *retentions* $ρ = λ/(1 + λ) ∈ [0, 1]$, i.e. $λ = ρ/(1 - ρ)$: the
        fraction of the prior surviving the update, so the iterate is an EWMA of the
        sufficient statistics with effective memory $1/(1 - ρ)$ — the forgetting factor of
        RLS / RiskMetrics. Passing a single `retention` ties $ρ_μ = ρ_Σ = ρ$, i.e. $λ_μ = λ_Σ$.

        $ρ$ rather than $λ$ or $1 - ρ$ because it alone composes across time,
        $ρ(Δt₁ + Δt₂) = ρ(Δt₁)ρ(Δt₂)$, making the schedule exactly $ρ(Δt) = e^{-rΔt}$ with
        half-life $\ln 2/r$; because the identity $ρ = 1$ is representable while $λ = ∞$ is
        not; and because $\text{keep} = ρ_Σ$ is read off without subtraction, leaving the
        singular point at $ρ_Σ = 0$, where floats are dense, instead of at $1$, which nearby
        legal values round *onto*.

    Args:
        z: Observation already pulled back to latent space. Any decoder log-Jacobian is
            constant in $θ$ and does not affect the minimizer.
        theta: Prior Gaussian parameters $θ₋$ in the selected parametrization.
        retention: Retention $ρ$, shared or split as $(ρ_μ, ρ_Σ)$. Broadcast against the
            batch shape of `theta`, so a per-sample schedule $ρ(Δt)$ is fine.
        parametrization: One of `"covariance"`, `"precision"`, `"cholesky"`, `"log-cholesky"`.

    Note: Admissible range
        $ρ_μ ∈ [0, 1]$ is closed: $0$ is the jump $μ₊ = z$, $1$ the identity $μ₊ = μ₋$.
        $ρ_Σ ∈ (0, 1]$: at $1$ the shape is frozen; $ρ_Σ → 0$ is the $λ_Σ → 0$ limit where
        $Σ₊$ degenerates to the rank-one $ρ_μδδᵀ$. Both are asserted, but $ρ_Σ > 0$ is only
        the mathematical bound — $\mathrm{cond}(Σ₊) ≈ 1 + \text{gain}⋅q/\text{keep}$ exceeds
        $1/ε$ near $ρ_Σ ≈ 10⁻⁷$ in fp32, which is already a $10⁷{:}1$ forgetting ratio.

    Warning:
        Do not tie $ρ$ in a $Δt$-dependent schedule. With $ρ_μ = ρ_Σ = e^{-rΔt}$, a long gap
        sends $Σ₊ → (1 - ρ)ρ⋅δδᵀ → 0$: zero covariance, infinite confidence, infinite NLL.
        Untie them, letting $ρ_μ$ decay faster than $ρ_Σ$.

    See Also:
        `argmin_reverse_kl`:
            Same observation term, opposite KL direction. That update is *not* an affine
            interpolation — its coefficients are the admissible root of a cubic — so
            "retention" has no referent and it keeps the $λ$ convention. Its constraint
            $λ_Σ > 1$ reads as $ρ_Σ > ½$: the reverse KL can never forget more than half the
            prior covariance. The signatures are *not* interchangeable.
        `argmin_proximal_kl`:
            Generic reverse-KL proximal solver for a linearized loss.
    """
    parametrization = CovarianceType(parametrization)
    μ, matrix = theta
    rho_mu, rho_sigma = (
        retention if isinstance(retention, tuple) else (retention, retention)
    )
    ρ_μ = torch.as_tensor(rho_mu, dtype=matrix.dtype, device=matrix.device)
    ρ_Σ = torch.as_tensor(rho_sigma, dtype=matrix.dtype, device=matrix.device)
    assert ((ρ_μ >= 0.0) & (ρ_μ <= 1.0)).all(), "requires rho_mu in [0, 1]"
    assert ((ρ_Σ > 0.0) & (ρ_Σ <= 1.0)).all(), "requires rho_sigma in (0, 1]"

    keep = ρ_Σ
    gain = (1.0 - ρ_Σ) * ρ_μ
    δ = z - μ
    μ_new = μ + (1.0 - ρ_μ)[..., None] * δ

    match parametrization:
        case CovarianceType.COVARIANCE:
            Σ = matrix
            δδT = torch.einsum("...i, ...j -> ...ij", δ, δ)
            Σ_new = (
                keep[..., None, None] * Σ
                + gain[..., None, None] * δδT
            )  # fmt: skip
            Σ_new = 0.5 * (Σ_new + Σ_new.mT)  # ensure symmetry
            return μ_new, Σ_new

        case CovarianceType.PRECISION:
            Λ = matrix
            projected = torch.einsum("...ij, ...j -> ...i", Λ, δ)
            mahalanobis = vecdot(δ, projected, dim=-1)
            outer = torch.einsum("...i, ...j -> ...ij", projected, projected)
            denom = keep + gain * mahalanobis
            Λ_new = (
                keep[..., None, None].reciprocal()
                * (Λ - (gain / denom)[..., None, None] * outer)
            )  # fmt: skip
            Λ_new = 0.5 * (Λ_new + Λ_new.mT)  # ensure symmetry
            return μ_new, Λ_new

        case CovarianceType.CHOLESKY:
            L = matrix
            u = solve_triangular(L, δ.unsqueeze(-1), upper=False).squeeze(-1)
            I = torch.eye(L.shape[-1], dtype=L.dtype, device=L.device)
            uuT = torch.einsum("...i, ...j -> ...ij", u, u)
            local_cov = keep[..., None, None] * I + gain[..., None, None] * uuT
            chol_new = L @ cholesky(local_cov)
            return μ_new, torch.tril(chol_new)

        case CovarianceType.LOG_CHOLESKY:
            log_chol_prior = matrix
            L = (
                log_chol_prior.tril(diagonal=-1)
                + log_chol_prior.diagonal(dim1=-2, dim2=-1).exp().diag_embed()
            )
            u = solve_triangular(L, δ.unsqueeze(-1), upper=False).squeeze(-1)
            I = torch.eye(L.shape[-1], dtype=L.dtype, device=L.device)
            uuT = torch.einsum("...i, ...j -> ...ij", u, u)
            local_cov = keep[..., None, None] * I + gain[..., None, None] * uuT
            chol_new = L @ cholesky(local_cov)
            log_chol_new = (
                chol_new.tril(diagonal=-1)
                + chol_new.diagonal(dim1=-2, dim2=-1).log().diag_embed()
            )
            return μ_new, log_chol_new

        case _:
            raise ValueError(
                "Expected parametrization to be one of "
                "{'covariance', 'precision', 'cholesky', 'log-cholesky'}, "
                f"got {parametrization!r}."
            )


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
            L = (
                log_chol.tril(diagonal=-1)
                + log_chol.diagonal(dim1=-2, dim2=-1).exp().diag_embed()
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
            L = (
                log_chol.tril(diagonal=-1)
                + log_chol.diagonal(dim1=-2, dim2=-1).exp().diag_embed()
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
            chol_p = (
                log_chol_p.tril(diagonal=-1)
                + log_chol_p.diagonal(dim1=-2, dim2=-1).exp().diag_embed()
            )
            chol_q = (
                log_chol_q.tril(diagonal=-1)
                + log_chol_q.diagonal(dim1=-2, dim2=-1).exp().diag_embed()
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
