r"""Gaussian distributions and parameter updates.

!!! note "Marginalizing parametrized distributions"

    Analytic marginalization of a distribution may not directly define an update
    of its unconstrained weights. The distribution's parameters must satisfy
    constraints, such as positive definiteness of a covariance matrix.

    We often use an unconstrained weight tensor $w$ and obtain the actual
    parameters $θ$ by applying a transformation $θ = f(w)$ called the parametrization.

    Marginalization updates the parameters $θ$. To express it in weight space,
    we need a corresponding update of $w$ before applying the parametrization.

    In essence, this means we need to be able to find a mapping $w → w'$ such that the
    following diagram commutes:

    ```text
          f
    w ───────► θ
    │         │
    │ ϕ       │ m
    ▼         ▼
    w′───────► θ′
          f
    ```

    We therefore need a mapping $ϕ$ on the unconstrained weights such that
    $f(ϕ(w)) = m(f(w))$, where $m$ is the marginalization operation on the
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
    "GaussianParams",
    "CovarianceParams",
    "PrecisionParams",
    "CholeskyParams",
    "LogCholeskyParams",
]

import math
from enum import StrEnum
from typing import Final, Literal, NamedTuple, Optional, Self, assert_never

import torch
import torch.nn.functional as F
from torch import Tensor, cholesky_inverse, cholesky_solve, distributions as dist, nn
from torch.linalg import cholesky, solve_triangular, vecdot

from .base import DistributionBase

type GaussianParams = tuple[Tensor, Tensor]
r"""Gaussian tensor pair; representation-specific parameter tuples also satisfy this type."""
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

    @classmethod
    def from_params(
        cls,
        theta: tuple[Tensor, Tensor],
        /,
        parametrization: str | None = None,
    ) -> CovarianceType:
        r"""Infer a parameter pair's representation and reject conflicting overrides.

        Our parameter tuples provide their own covariance type. Other tensor pairs
        require an explicit parametrization. Missing, unknown, or conflicting
        parametrizations raise `ValueError`.
        """
        if isinstance(
            theta,
            (CovarianceParams, PrecisionParams, CholeskyParams, LogCholeskyParams),
        ):
            if parametrization is None:
                return cls(theta.covariance_type)
            covariance_type = cls(parametrization)
            if covariance_type != theta.covariance_type:
                raise ValueError(
                    f"Parametrization {covariance_type.value!r} does not match "
                    f"{type(theta).__name__} ({theta.covariance_type.value!r})."
                )
            return covariance_type
        return CovarianceType(parametrization)

    def _as_params(
        self, theta: GaussianParams, /
    ) -> CovarianceParams | PrecisionParams | CholeskyParams | LogCholeskyParams:
        r"""Wrap a tensor pair in this representation's parameter type."""
        match self:
            case CovarianceType.COVARIANCE:
                return CovarianceParams(*theta)
            case CovarianceType.PRECISION:
                return PrecisionParams(*theta)
            case CovarianceType.CHOLESKY:
                return CholeskyParams(*theta)
            case CovarianceType.LOG_CHOLESKY:
                return LogCholeskyParams(*theta)
            case other:
                assert_never(other)

    def to_covariance(self, theta: GaussianParams, /) -> CovarianceParams:
        r"""Return Gaussian parameters in covariance parametrization."""
        return self._as_params(theta).to_covariance()

    def to_precision(self, theta: GaussianParams, /) -> PrecisionParams:
        r"""Return the precision parametrization."""
        return self._as_params(theta).to_precision()

    def to_cholesky(self, theta: GaussianParams, /) -> CholeskyParams:
        r"""Return the Cholesky parametrization."""
        return self._as_params(theta).to_cholesky()

    def to_log_cholesky(self, theta: GaussianParams, /) -> LogCholeskyParams:
        r"""Return the log-Cholesky parametrization."""
        return self._as_params(theta).to_log_cholesky()

    def from_covariance(self, theta: tuple[Tensor, Tensor], /) -> GaussianParams:
        r"""Return covariance-parametrized Gaussian parameters in this form."""
        params = CovarianceParams(*theta)
        match self:
            case CovarianceType.COVARIANCE:
                return params
            case CovarianceType.PRECISION:
                return params.to_precision()
            case CovarianceType.CHOLESKY:
                return params.to_cholesky()
            case CovarianceType.LOG_CHOLESKY:
                return params.to_log_cholesky()
            case other:
                assert_never(other)


class CovarianceParams(NamedTuple):
    r"""Covariance parameters `(mean, covariance)` with positive-definite covariance.

    Construction does not validate or broadcast inputs. Tensors must have
    compatible batch shapes, dtype, and device. The tuple is immutable, but its
    tensors remain mutable.
    """

    mean: Tensor
    r"""Mean vector with shape `(..., D)`."""
    covariance: Tensor
    r"""Covariance matrix with shape `(..., D, D)`."""

    @property
    def covariance_type(self) -> Literal[CovarianceType.COVARIANCE]:
        r"""Representation of the second tensor."""
        return CovarianceType.COVARIANCE

    def log_prob(self, value: Tensor, /) -> Tensor:
        r"""Evaluate the density via [`log_prob`][imtskit.distributions.gaussian.log_prob]."""
        return log_prob(value, self)

    def kl(self, other: GaussianParams, /) -> Tensor:
        r"""Compute KL to `other`, which must use the same representation."""
        return kl(self, other, self.covariance_type)

    def fisher(self, tangent: GaussianParams, /) -> GaussianParams:
        r"""Apply [`fisher`][imtskit.distributions.gaussian.fisher] to a direction pair."""
        return fisher(self, tangent)

    def inverse_fisher(self, cotangent: GaussianParams, /) -> GaussianParams:
        r"""Apply [`inverse_fisher`][imtskit.distributions.gaussian.inverse_fisher]."""
        return inverse_fisher(self, cotangent)

    def argmin_forward_kl(
        self,
        value: Tensor,
        /,
        *,
        retention: ScalarLike | tuple[ScalarLike, ScalarLike],
        mask: Tensor | None = None,
    ) -> CovarianceParams:
        r"""Return the [forward-KL update][imtskit.distributions.gaussian.argmin_forward_kl]."""
        result = argmin_forward_kl(value, self, retention=retention, mask=mask)
        return CovarianceParams(*result)

    def argmin_reverse_kl(
        self,
        value: Tensor,
        /,
        *,
        retention: ScalarLike | tuple[ScalarLike, ScalarLike],
        mask: Tensor | None = None,
    ) -> CovarianceParams:
        r"""Return the [reverse-KL update][imtskit.distributions.gaussian.argmin_reverse_kl]."""
        result = argmin_reverse_kl(value, self, retention=retention, mask=mask)
        return CovarianceParams(*result)

    def to_covariance(self) -> CovarianceParams:
        r"""Convert to covariance parameters."""
        return self

    def to_precision(self) -> PrecisionParams:
        r"""Convert to precision parameters."""
        return PrecisionParams(self.mean, cholesky_inverse(cholesky(self.covariance)))

    def to_cholesky(self) -> CholeskyParams:
        r"""Convert to Cholesky parameters."""
        return CholeskyParams(self.mean, cholesky(self.covariance))

    def to_log_cholesky(self) -> LogCholeskyParams:
        r"""Convert to parameters with logarithms on the Cholesky diagonal."""
        return self.to_cholesky().to_log_cholesky()


class PrecisionParams(NamedTuple):
    r"""Precision parameters `(mean, precision)` with positive-definite precision.

    Construction does not validate or broadcast inputs. Tensors must have
    compatible batch shapes, dtype, and device. The tuple is immutable, but its
    tensors remain mutable.
    """

    mean: Tensor
    r"""Mean vector with shape `(..., D)`."""
    precision: Tensor
    r"""Precision matrix with shape `(..., D, D)`."""

    @property
    def covariance_type(self) -> Literal[CovarianceType.PRECISION]:
        r"""Representation of the second tensor."""
        return CovarianceType.PRECISION

    def log_prob(self, value: Tensor, /) -> Tensor:
        r"""Evaluate the density via [`log_prob`][imtskit.distributions.gaussian.log_prob]."""
        return log_prob(value, self)

    def kl(self, other: GaussianParams, /) -> Tensor:
        r"""Compute KL to `other`, which must use the same representation."""
        return kl(self, other, self.covariance_type)

    def fisher(self, tangent: GaussianParams, /) -> GaussianParams:
        r"""Apply [`fisher`][imtskit.distributions.gaussian.fisher] to a direction pair."""
        return fisher(self, tangent)

    def inverse_fisher(self, cotangent: GaussianParams, /) -> GaussianParams:
        r"""Apply [`inverse_fisher`][imtskit.distributions.gaussian.inverse_fisher]."""
        return inverse_fisher(self, cotangent)

    def argmin_forward_kl(
        self,
        value: Tensor,
        /,
        *,
        retention: ScalarLike | tuple[ScalarLike, ScalarLike],
        mask: Tensor | None = None,
    ) -> Self:
        r"""Return the [forward-KL update][imtskit.distributions.gaussian.argmin_forward_kl]."""
        return self.__class__(
            *argmin_forward_kl(value, self, retention=retention, mask=mask)
        )

    def argmin_reverse_kl(
        self,
        value: Tensor,
        /,
        *,
        retention: ScalarLike | tuple[ScalarLike, ScalarLike],
        mask: Tensor | None = None,
    ) -> Self:
        r"""Return the [reverse-KL update][imtskit.distributions.gaussian.argmin_reverse_kl]."""
        return self.__class__(
            *argmin_reverse_kl(value, self, retention=retention, mask=mask)
        )

    def to_covariance(self) -> CovarianceParams:
        r"""Convert to covariance parameters."""
        return CovarianceParams(self.mean, cholesky_inverse(cholesky(self.precision)))

    def to_precision(self) -> PrecisionParams:
        r"""Convert to precision parameters."""
        return self

    def to_cholesky(self) -> CholeskyParams:
        r"""Convert to Cholesky parameters."""
        return self.to_covariance().to_cholesky()

    def to_log_cholesky(self) -> LogCholeskyParams:
        r"""Convert to parameters with logarithms on the Cholesky diagonal."""
        return self.to_cholesky().to_log_cholesky()


class CholeskyParams(NamedTuple):
    r"""Cholesky parameters `(mean, scale_tril)` with a positive-diagonal factor.

    Construction does not validate or broadcast inputs. Tensors must have
    compatible batch shapes, dtype, and device. The tuple is immutable, but its
    tensors remain mutable.
    """

    mean: Tensor
    r"""Mean vector with shape `(..., D)`."""
    scale_tril: Tensor
    r"""Lower-triangular factor of the covariance with positive diagonal."""

    @property
    def covariance_type(self) -> Literal[CovarianceType.CHOLESKY]:
        r"""Representation of the second tensor."""
        return CovarianceType.CHOLESKY

    def log_prob(self, value: Tensor, /) -> Tensor:
        r"""Evaluate the density via [`log_prob`][imtskit.distributions.gaussian.log_prob]."""
        return log_prob(value, self)

    def kl(self, other: GaussianParams, /) -> Tensor:
        r"""Compute KL to `other`, which must use the same representation."""
        return kl(self, other, self.covariance_type)

    def fisher(self, tangent: GaussianParams, /) -> GaussianParams:
        r"""Apply [`fisher`][imtskit.distributions.gaussian.fisher] to a direction pair."""
        return fisher(self, tangent)

    def inverse_fisher(self, cotangent: GaussianParams, /) -> GaussianParams:
        r"""Apply [`inverse_fisher`][imtskit.distributions.gaussian.inverse_fisher]."""
        return inverse_fisher(self, cotangent)

    def argmin_forward_kl(
        self,
        value: Tensor,
        /,
        *,
        retention: ScalarLike | tuple[ScalarLike, ScalarLike],
        mask: Tensor | None = None,
    ) -> Self:
        r"""Return the [forward-KL update][imtskit.distributions.gaussian.argmin_forward_kl]."""
        return self.__class__(
            *argmin_forward_kl(value, self, retention=retention, mask=mask)
        )

    def argmin_reverse_kl(
        self,
        value: Tensor,
        /,
        *,
        retention: ScalarLike | tuple[ScalarLike, ScalarLike],
        mask: Tensor | None = None,
    ) -> Self:
        r"""Return the [reverse-KL update][imtskit.distributions.gaussian.argmin_reverse_kl]."""
        return self.__class__(
            *argmin_reverse_kl(value, self, retention=retention, mask=mask)
        )

    def to_covariance(self) -> CovarianceParams:
        r"""Convert to covariance parameters."""
        return CovarianceParams(self.mean, self.scale_tril @ self.scale_tril.mT)

    def to_precision(self) -> PrecisionParams:
        r"""Convert to precision parameters."""
        return PrecisionParams(self.mean, cholesky_inverse(self.scale_tril))

    def to_cholesky(self) -> CholeskyParams:
        r"""Convert to Cholesky parameters."""
        return self

    def to_log_cholesky(self) -> LogCholeskyParams:
        r"""Convert to parameters with logarithms on the Cholesky diagonal."""
        return LogCholeskyParams(
            self.mean,
            self.scale_tril.diagonal(dim1=-2, dim2=-1).log().diag_embed()
            + self.scale_tril.tril(diagonal=-1),
        )


class LogCholeskyParams(NamedTuple):
    r"""Log-Cholesky parameters with logarithms stored only on the diagonal.

    Construction does not validate or broadcast inputs. Tensors must have
    compatible batch shapes, dtype, and device. The tuple is immutable, but its
    tensors remain mutable.
    """

    mean: Tensor
    r"""Mean vector with shape `(..., D)`."""
    log_scale_tril: Tensor
    r"""Lower-triangular matrix with log diagonal and unchanged off-diagonal entries."""

    @property
    def covariance_type(self) -> Literal[CovarianceType.LOG_CHOLESKY]:
        r"""Representation of the second tensor."""
        return CovarianceType.LOG_CHOLESKY

    def log_prob(self, value: Tensor, /) -> Tensor:
        r"""Evaluate the density via [`log_prob`][imtskit.distributions.gaussian.log_prob]."""
        return log_prob(value, self)

    def kl(self, other: GaussianParams, /) -> Tensor:
        r"""Compute KL to `other`, which must use the same representation."""
        return kl(self, other, self.covariance_type)

    def fisher(self, tangent: GaussianParams, /) -> GaussianParams:
        r"""Apply [`fisher`][imtskit.distributions.gaussian.fisher] to a direction pair."""
        return fisher(self, tangent)

    def inverse_fisher(self, cotangent: GaussianParams, /) -> GaussianParams:
        r"""Apply [`inverse_fisher`][imtskit.distributions.gaussian.inverse_fisher]."""
        return inverse_fisher(self, cotangent)

    def argmin_forward_kl(
        self,
        value: Tensor,
        /,
        *,
        retention: ScalarLike | tuple[ScalarLike, ScalarLike],
        mask: Tensor | None = None,
    ) -> Self:
        r"""Return the [forward-KL update][imtskit.distributions.gaussian.argmin_forward_kl]."""
        return self.__class__(
            *argmin_forward_kl(value, self, retention=retention, mask=mask)
        )

    def argmin_reverse_kl(
        self,
        value: Tensor,
        /,
        *,
        retention: ScalarLike | tuple[ScalarLike, ScalarLike],
        mask: Tensor | None = None,
    ) -> Self:
        r"""Return the [reverse-KL update][imtskit.distributions.gaussian.argmin_reverse_kl]."""
        return self.__class__(
            *argmin_reverse_kl(value, self, retention=retention, mask=mask)
        )

    def to_covariance(self) -> CovarianceParams:
        r"""Convert to covariance parameters."""
        return self.to_cholesky().to_covariance()

    def to_precision(self) -> PrecisionParams:
        r"""Convert to precision parameters."""
        return self.to_cholesky().to_precision()

    def to_cholesky(self) -> CholeskyParams:
        r"""Convert to Cholesky parameters."""
        return CholeskyParams(
            self.mean,
            self.log_scale_tril.tril(diagonal=-1)
            + self.log_scale_tril.diagonal(dim1=-2, dim2=-1).exp().diag_embed(),
        )

    def to_log_cholesky(self) -> LogCholeskyParams:
        r"""Convert to parameters with logarithms on the Cholesky diagonal."""
        return self


def log_prob(
    x: Tensor,  # (*S, ..., D)
    theta: GaussianParams,  # (..., D), (..., D, D)
    /,
    parametrization: str | None = None,
) -> Tensor:  # (*S, ...)
    r"""Compute the Gaussian log-density at `x` in the chosen parametrization.

    Parametrization is inferred from tagged parameter tuples when omitted. Plain
    tensor pairs require an explicit parametrization. Conflicting representations
    raise `ValueError`.
    """
    mean, matrix = theta
    residual = x - mean
    dim = residual.shape[-1]

    match CovarianceType.from_params(theta, parametrization):
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
            L = LogCholeskyParams(mean, log_chol).to_cholesky().scale_tril
            z = solve_triangular(L, residual.unsqueeze(-1), upper=False).squeeze(-1)
            logdet = 2 * log_chol.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
            mahalanobis = vecdot(z, z, dim=-1)
            return -0.5 * (dim * _LOG2PI + logdet + mahalanobis)

        case other:
            assert_never(other)


def _solve_w_closed_form(
    sq_dist: Tensor,  # (...)
    retention_mean: Tensor,  # (...)
    retention_cov: Tensor,  # (...)
    /,
    *,
    use_fp64: bool = True,
) -> Tensor:  # (...)
    r"""Solve the reverse-KL stationarity equation in retention coordinates.

    Return $w = (1 - ρ_μ) + ρ_μ s$, where $s$ is the whitened posterior variance along the
    innovation, i.e. the admissible root of

    $$  (s - ρ_Σ)((1 - ρ_μ) + ρ_μ s)² = (1 - ρ_Σ)sρ_μ²q.    $$

    The admissible ranges are $ρ_μ ∈ [0, 1]$ and $ρ_Σ ∈ (0, 1]$.

    Neither $λ_μ$ nor $λ_Σ$ appears: they cancel under $λ_μ = ρ_μ/(1 - ρ_μ)$,
    $λ_Σ = 1/(1 - ρ_Σ)$, which is what makes the endpoints $ρ_μ = 1$ ($λ_μ = ∞$) and
    $ρ_Σ = 1$ ($λ_Σ = ∞$) finite and exactly representable.

    With $A = (1 - ρ_μ) + ρ_μ ρ_Σ ∈ [ρ_Σ, 1]$, $κ = (1 - ρ_Σ)ρ_μ²q/A²$ and
    $τ = (1 - ρ_μ)/A ∈ [0, 1]$, substituting $w = A·v$ gives the scale-free monic cubic

    $$  v³ - v² - κv + κτ = 0.  $$

    For $κ > 0$ and $0 < τ < 1$, its roots are one negative, one in $(0, τ)$ and
    one $> 1$. Only the largest gives the admissible positive variance; endpoint
    cases follow by continuity. Depressing with $v = t + ⅓$ gives $p = -κ - ⅓ ≤ -⅓$.
    The cubic has three real roots, including repeated roots at degeneracies, and
    the largest root is the $k = 0$
    cosine branch. $A ≥ ρ_Σ > 0$ and $m³ ≥ 1/27$, so neither division needs a guard.

    !!! note "Degenerate branch"

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
    cos_θ = torch.where(degenerate, 0.0, raw).clamp(-1.0, 1.0)
    v_exact = 2.0 * m * torch.cos(torch.acos(cos_θ) / 3.0) + 1.0 / 3.0

    v = torch.where(degenerate, v_series, v_exact)
    return (A * v).to(dtype=out_dtype)


def _lift_masked_marginal_update(
    mean: Tensor,  # (..., d)
    covariance: Tensor,  # (..., d, d)
    /,
    marginal_covariance: Tensor,  # (..., d, d)
    delta_marginal_mean: Tensor,  # (..., d)
    delta_marginal_covariance: Tensor,  # (..., d, d)
    mask: Tensor,  # (..., d)
) -> GaussianParams:
    r"""Lift an observed-marginal update while preserving the prior conditional."""
    rhs = torch.where(mask[..., None, :], covariance, covariance.new_zeros(()))
    L = cholesky(marginal_covariance)
    gain = cholesky_solve(rhs.mT, L).mT

    mean_new = mean + torch.einsum("...ij, ...j -> ...i", gain, delta_marginal_mean)
    Σ_new = covariance + gain @ delta_marginal_covariance @ gain.mT
    Σ_new = 0.5 * (Σ_new + Σ_new.mT)
    return mean_new, Σ_new


def argmin_reverse_kl(
    z: Tensor,  # (..., d)
    theta: GaussianParams,  # (..., d), (..., d, d)
    /,
    parametrization: str | None = None,
    *,
    retention: ScalarLike | tuple[ScalarLike, ScalarLike],  # rho or (rho_mu, rho_sigma)
    mask: Tensor | None = None,  # (..., d), True where observed
) -> GaussianParams:  # (..., d), (..., d, d)
    r"""Return the exact minimizer of NLL plus reverse-KL regularization term.

    Parametrization is inferred from tagged parameter tuples when omitted. Plain
    tensor pairs require an explicit parametrization. Conflicting representations
    raise `ValueError`.

    This returns the exact minimizer of

    $$\begin{aligned}
    θ₊  &= \argmin_θ -\log 𝓝(z; θ) + λ⋅\KL(𝓝(θ) ∥ 𝓝(θ₋)) \\
        &= \argmin_θ -\log 𝓝(z; θ)
            + λ_μ ½(μ - μ₋)ᵀΣ₋⁻¹(μ - μ₋)
            + λ_Σ ½(\tr(ΣΣ₋⁻¹) - \log\det(ΣΣ₋⁻¹) - d)
            \qquad (λ_μ = λ_Σ = λ)
    \end{aligned}$$

    Here $θ₋$ is the input `theta`, interpreted according to `parametrization`.

    ### Update

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

    ### Parametrization

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
        mask: Optional mask indicating observed coordinates of `z`. Currently supported
            only for covariance-parametrized inputs. Missing coordinates are handled by
            optimizing the observed marginal likelihood and preserving the prior conditional.
        parametrization: One of `"covariance"`, `"precision"`, `"cholesky"`, `"log-cholesky"`.

    !!! note "Admissible range"

        $ρ_μ ∈ [0, 1]$ is closed: $0$ is $λ_μ = 0$, the unregularized jump $μ₊ = z$; $1$ is
        $λ_μ → ∞$, the identity $μ₊ = μ₋$. $ρ_Σ ∈ (0, 1]$: at $1$ the shape is frozen
        ($λ_Σ → ∞$), and $ρ_Σ → 0$ is $λ_Σ → 1⁺$, where $Σ₊$ degenerates.
        Unit retention is represented exactly without evaluating infinite weights.

    ### See also

    [`argmin_forward_kl`][imtskit.distributions.gaussian.argmin_forward_kl] uses
    the same observation term and the opposite KL direction.
    Its covariance retention map is $ρ_Σ = λ_Σ/(1 + λ_Σ)$, since it has no
    $λ_Σ > 1$ constraint. In both updates, $ρ_Σ$ is the coefficient multiplying $Σ₋$.
    """
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

    match CovarianceType.from_params(theta, parametrization):
        case CovarianceType.COVARIANCE:
            Σ = matrix
            if mask is not None:
                observed_mask = torch.as_tensor(mask, dtype=torch.bool, device=Σ.device)
                δ = torch.where(observed_mask, δ, δ.new_zeros(()))
                eye = torch.eye(dim, dtype=Σ.dtype, device=Σ.device)
                observed = observed_mask[..., :, None] & observed_mask[..., None, :]
                observed_covariance = torch.where(observed, Σ, Σ.new_zeros(()))
                marginal_covariance = torch.where(observed, Σ, eye)
                L = cholesky(marginal_covariance)
                solved = cholesky_solve(δ.unsqueeze(-1), L).squeeze(-1)
                q = vecdot(δ, solved, dim=-1)
                w = _solve_w_closed_form(q, ρ_μ, ρ_Σ)
                ρs = w - (1.0 - ρ_μ)
                delta_mean = ((1.0 - ρ_μ) / w)[..., None] * δ
                outer = torch.einsum("...i, ...j -> ...ij", δ, δ)
                coefficient = forget * ρ_μ * ρs / w.square()
                delta_covariance = (
                    -forget[..., None, None] * observed_covariance
                    + coefficient[..., None, None] * outer
                )
                return _lift_masked_marginal_update(
                    μ,
                    Σ,
                    marginal_covariance,
                    delta_mean,
                    delta_covariance,
                    observed_mask,
                )

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
            if mask is not None:
                raise NotImplementedError
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
            if mask is not None:
                raise NotImplementedError
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
            if mask is not None:
                raise NotImplementedError
            L = LogCholeskyParams(μ, matrix).to_cholesky().scale_tril
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

        case other:
            assert_never(other)


def argmin_forward_kl(
    z: Tensor,  # (..., d)
    theta: GaussianParams,  # (..., d), (..., d, d)
    /,
    parametrization: str | None = None,
    *,
    retention: ScalarLike | tuple[ScalarLike, ScalarLike],  # rho or (rho_mu, rho_sigma)
    mask: Tensor | None = None,  # (..., d), True where observed
) -> GaussianParams:  # (..., d), (..., d, d)
    r"""Return the exact minimizer of NLL plus separable forward-KL anchoring.

    Parametrization is inferred from tagged parameter tuples when omitted. Plain
    tensor pairs require an explicit parametrization. Conflicting representations
    raise `ValueError`.

    This returns the exact minimizer of

    $$\begin{aligned}
    θ₊  &= \argmin_θ -\log 𝓝(z; θ) + λ⋅\KL(𝓝(θ₋) ∥ 𝓝(θ)) \\
        &= \argmin_θ -\log 𝓝(z; θ)
            + λ_μ ½(μ - μ₋)ᵀΣ⁻¹(μ - μ₋)
            + λ_Σ ½(\tr(Σ₋Σ⁻¹) - \log\det(Σ₋Σ⁻¹) - d)
            \qquad (λ_μ = λ_Σ = λ)
    \end{aligned}$$

    Here $θ₋$ is the input `theta`, interpreted according to `parametrization`.

    ### Update

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
    - `"log-cholesky"`: as `"cholesky"`, then store the diagonal in log form.

    ### Parametrization

    Weights are the *retentions* $ρ = λ/(1 + λ) ∈ [0, 1]$, i.e. $λ = ρ/(1 - ρ)$: the
    fraction of the prior surviving the update, so the iterate is an EWMA of the
    sufficient statistics with effective memory $1/(1 - ρ)$ — the forgetting factor of
    RLS / RiskMetrics. Passing a single `retention` ties $ρ_μ = ρ_Σ = ρ$, i.e. $λ_μ = λ_Σ$.

    We parameterize the update by $ρ$ because exponential retention schedules
    compose across time: $ρ(Δt₁ + Δt₂) = ρ(Δt₁)ρ(Δt₂)$ for $ρ(Δt) = e^{-rΔt}$, with
    half-life $\ln 2/r$. The identity $ρ = 1$ is directly representable, whereas
    an infinite regularization weight is handled as a limit. In addition,
    $\text{keep} = ρ_Σ$ is read off without subtraction, leaving the
    singular point at $ρ_Σ = 0$, where floats are dense, instead of at $1$, which nearby
    legal values round *onto*.

    Args:
        z: Observation already pulled back to latent space. Any decoder log-Jacobian is
            constant in $θ$ and does not affect the minimizer.
        theta: Prior Gaussian parameters $θ₋$ in the selected parametrization.
        retention: Retention $ρ$, shared or split as $(ρ_μ, ρ_Σ)$. Broadcast against the
            batch shape of `theta`, so a per-sample schedule $ρ(Δt)$ is fine.
        mask: Optional mask indicating observed coordinates of `z`. Currently supported
            only for covariance-parametrized inputs. Missing coordinates are handled by
            optimizing the observed marginal likelihood and preserving the prior conditional.
        parametrization: One of `"covariance"`, `"precision"`, `"cholesky"`, `"log-cholesky"`.

    !!! note "Admissible range"

        $ρ_μ ∈ [0, 1]$ is closed: $0$ is the jump $μ₊ = z$, $1$ the identity $μ₊ = μ₋$.
        $ρ_Σ ∈ (0, 1]$: at $1$ the shape is frozen; $ρ_Σ → 0$ is the $λ_Σ → 0$ limit where
        $Σ₊$ degenerates to the rank-one $ρ_μδδᵀ$. Both are asserted, but $ρ_Σ > 0$ is only
        the mathematical bound — $\mathrm{cond}(Σ₊) ≈ 1 + \text{gain}⋅q/\text{keep}$ exceeds
        $1/ε$ near $ρ_Σ ≈ 10⁻⁷$ in fp32, which is already a $10⁷{:}1$ forgetting ratio.

    !!! warning

        Do not tie $ρ$ in a $Δt$-dependent schedule. With $ρ_μ = ρ_Σ = e^{-rΔt}$, a long gap
        sends $Σ₊ → (1 - ρ)ρ⋅δδᵀ → 0$: zero covariance, infinite confidence, infinite NLL.
        Untie them, letting $ρ_μ$ decay faster than $ρ_Σ$.

    ### See also

    [`argmin_reverse_kl`][imtskit.distributions.gaussian.argmin_reverse_kl] uses
    the same observation term and the opposite KL direction.
    Its coefficients depend on the admissible root of a cubic. Both functions accept
    `retention`, but reverse KL uses $ρ_Σ = 1 - 1/λ_Σ$, whereas forward KL uses
    $ρ_Σ = λ_Σ/(1 + λ_Σ)$. The nominal mean retention map is the same in both cases.
    """
    μ, matrix = theta
    rho_mu, rho_sigma = (
        retention if isinstance(retention, tuple) else (retention, retention)
    )
    ρ_μ = torch.as_tensor(rho_mu, dtype=matrix.dtype, device=matrix.device)
    ρ_Σ = torch.as_tensor(rho_sigma, dtype=matrix.dtype, device=matrix.device)
    assert ((ρ_μ >= 0.0) & (ρ_μ <= 1.0)).all(), "requires rho_mu in [0, 1]"
    assert ((ρ_Σ > 0.0) & (ρ_Σ <= 1.0)).all(), "requires rho_sigma in (0, 1]"

    keep = ρ_Σ
    forget = 1.0 - ρ_Σ
    gain = (1.0 - ρ_Σ) * ρ_μ
    δ = z - μ
    μ_new = μ + (1.0 - ρ_μ)[..., None] * δ

    match CovarianceType.from_params(theta, parametrization):
        case CovarianceType.COVARIANCE:
            Σ = matrix
            if mask is not None:
                observed_mask = torch.as_tensor(mask, dtype=torch.bool, device=Σ.device)
                δ = torch.where(observed_mask, δ, δ.new_zeros(()))
                eye = torch.eye(Σ.shape[-1], dtype=Σ.dtype, device=Σ.device)
                observed = observed_mask[..., :, None] & observed_mask[..., None, :]
                observed_covariance = torch.where(observed, Σ, Σ.new_zeros(()))
                marginal_covariance = torch.where(observed, Σ, eye)
                delta_mean = (1.0 - ρ_μ)[..., None] * δ
                δδT = torch.einsum("...i, ...j -> ...ij", δ, δ)
                delta_covariance = (
                    -forget[..., None, None] * observed_covariance
                    + gain[..., None, None] * δδT
                )
                return _lift_masked_marginal_update(
                    μ,
                    Σ,
                    marginal_covariance,
                    delta_mean,
                    delta_covariance,
                    observed_mask,
                )

            δδT = torch.einsum("...i, ...j -> ...ij", δ, δ)
            Σ_new = (
                keep[..., None, None] * Σ
                + gain[..., None, None] * δδT
            )  # fmt: skip
            Σ_new = 0.5 * (Σ_new + Σ_new.mT)  # ensure symmetry
            return μ_new, Σ_new

        case CovarianceType.PRECISION:
            if mask is not None:
                raise NotImplementedError
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
            if mask is not None:
                raise NotImplementedError
            L = matrix
            u = solve_triangular(L, δ.unsqueeze(-1), upper=False).squeeze(-1)
            I = torch.eye(L.shape[-1], dtype=L.dtype, device=L.device)
            uuT = torch.einsum("...i, ...j -> ...ij", u, u)
            local_cov = keep[..., None, None] * I + gain[..., None, None] * uuT
            chol_new = L @ cholesky(local_cov)
            return μ_new, torch.tril(chol_new)

        case CovarianceType.LOG_CHOLESKY:
            if mask is not None:
                raise NotImplementedError
            L = LogCholeskyParams(μ, matrix).to_cholesky().scale_tril
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

        case other:
            assert_never(other)


def fisher(
    theta: GaussianParams,  # (..., d), (..., d, d)
    tangent: GaussianParams,  # (..., d), (..., d, d)
    /,
    parametrization: str | None = None,
) -> GaussianParams:  # (..., d), (..., d, d)
    r"""Apply the Gaussian Fisher/KL metric in the chosen parametrization.

    Parametrization is inferred from tagged parameter tuples when omitted. Plain
    tensor pairs require an explicit parametrization. Conflicting representations
    raise `ValueError`.

    $$\begin{aligned}
        F_{(μ, Σ)}(δμ, δΣ) &= (Σ⁻¹δμ, ½Σ⁻¹\sym(δΣ)Σ⁻¹)
        \\  F_{(μ, Λ)}(δμ, δΛ) &= (Λδμ, ½Σ\sym(δΛ)Σ), \qquad Σ = Λ⁻¹
        \\  F_{(μ, L)}(δμ, δL) &= (Σ⁻¹δμ, L⁻ᵀ(L⁻¹δL + \diag(L⁻¹δL))), \qquad Σ = LLᵀ
        \\  F_{(μ, X)}(δμ, δX) &= (Σ⁻¹δμ, Jₓᵀ F_{(μ, L)}(JₓδX)),
        \qquad L = \tril(X, -1) + \diag(e^{\diag X})
    \end{aligned}$$

    Args:
        theta: Gaussian parameters in the selected parametrization.
        tangent: Tangent/cotangent-like direction to which the metric is applied.
        parametrization: One of `"covariance"`, `"precision"`, `"cholesky"` or `"log-cholesky"`.
    """
    match CovarianceType.from_params(theta, parametrization):
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
            L = LogCholeskyParams(mean, log_chol).to_cholesky().scale_tril
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

        case other:
            assert_never(other)


def inverse_fisher(
    theta: GaussianParams,  # (..., d), (..., d, d)
    cotangent: GaussianParams,  # (..., d), (..., d, d)
    /,
    parametrization: str | None = None,
) -> GaussianParams:  # (..., d), (..., d, d)
    r"""Apply the inverse Gaussian Fisher/KL metric in the chosen parametrization.

    Parametrization is inferred from tagged parameter tuples when omitted. Plain
    tensor pairs require an explicit parametrization. Conflicting representations
    raise `ValueError`.

    $$\begin{aligned}
        F_{(μ, Σ)}⁻¹(g, G) &= (Σg, 2Σ\sym(G)Σ)
        \\  F_{(μ, Λ)}⁻¹(g, G) &= (Σg, 2Λ\sym(G)Λ), \qquad Σ = Λ⁻¹
        \\  F_{(μ, L)}⁻¹(g, G) &= (Σg, L(\tril(LᵀG) - ½\diag(LᵀG))), \qquad Σ = LLᵀ
        \\  F_{(μ, X)}⁻¹(g, G) &= (Σg, Jₓ⁻¹F_{(μ, L)}⁻¹(Jₓ⁻ᵀG)),
            \qquad L = \tril(X, -1) + \diag(e^{\diag X})
    \end{aligned}$$

    Args:
        theta: Gaussian parameters in the selected parametrization.
        cotangent: Cotangent-like direction to which the inverse metric is applied.
        parametrization: One of `"covariance"`, `"precision"`, `"cholesky"`, or `"log-cholesky"`.
    """
    match CovarianceType.from_params(theta, parametrization):
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
            L = LogCholeskyParams(mean, log_chol).to_cholesky().scale_tril
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

        case other:
            assert_never(other)


def kl(
    p: GaussianParams,  # (..., d), (..., d, d)
    q: GaussianParams,  # (..., d), (..., d, d)
    /,
    parametrization: str | None = None,
) -> Tensor:  # (...)
    r"""Return the KL divergence from Gaussian distribution `p` to `q`.

    Parametrization is inferred from tagged parameter tuples when omitted. Plain
    tensor pairs require an explicit parametrization. Conflicting representations
    raise `ValueError`.

    Args:
        p: Gaussian parameters in the selected parametrization.
        q: Gaussian parameters in the selected parametrization.
        parametrization: One of `"covariance"`, `"precision"`, `"cholesky"`,
            or `"log-cholesky"`.

    Returns:
        The KL divergence $\KL(p，q)$.
    """
    covariance_type = CovarianceType.from_params(p, parametrization)
    if CovarianceType.from_params(q, parametrization) != covariance_type:
        raise ValueError("Gaussian parameters must use the same parametrization.")

    match covariance_type:
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
            chol_p = LogCholeskyParams(mean_p, log_chol_p).to_cholesky().scale_tril
            chol_q = LogCholeskyParams(mean_q, log_chol_q).to_cholesky().scale_tril
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

        case other:
            assert_never(other)


class MultivariateNormal(dist.MultivariateNormal):
    r"""Augmented Multivariate Normal distribution.

    We add some utilities to the base class.
    """

    CovarianceParams: Final = CovarianceParams
    r"""Constructor for covariance parameters."""
    PrecisionParams: Final = PrecisionParams
    r"""Constructor for precision parameters."""
    CholeskyParams: Final = CholeskyParams
    r"""Constructor for Cholesky parameters."""
    LogCholeskyParams: Final = LogCholeskyParams
    r"""Constructor for log-Cholesky parameters."""

    def __add__(self, bias: float | Tensor, /) -> Self:
        r"""Add a tensor to the mean."""
        return self.__class__(
            self.mean + bias,
            self.covariance_matrix,  # ty: ignore[invalid-argument-type]
        )

    def __mul__(self, scale: float | Tensor, /) -> Self:
        r"""Multiply by a tensor."""
        return self.__class__(
            scale * self.mean,
            scale**2 * self.covariance_matrix,  # ty: ignore[unsupported-operator]
        )

    def __matmul__(self, scale: Tensor, /) -> Self:
        r"""Multiply by a tensor."""
        return self.__class__(
            scale @ self.mean,
            scale @ self.covariance_matrix @ scale.T,  # ty: ignore[unsupported-operator]
        )


class MultiHeadGaussian(DistributionBase):
    r"""Implements a multi-head Gaussian distribution."""

    CovarianceParams: Final = CovarianceParams
    r"""Constructor for covariance parameters."""
    PrecisionParams: Final = PrecisionParams
    r"""Constructor for precision parameters."""
    CholeskyParams: Final = CholeskyParams
    r"""Constructor for Cholesky parameters."""
    LogCholeskyParams: Final = LogCholeskyParams
    r"""Constructor for log-Cholesky parameters."""

    normalization_constant: Tensor
    r"""CONST: Normalization constant of a Gaussian distribution."""
    num_heads: Final[int]
    r"""CONST: Number of Gaussian heads."""
    num_features: Final[int]
    r"""CONST: Number of features in input."""

    # parameters/buffers
    means: Tensor
    r"""PARAM: Means of the Gaussian heads."""
    scale_tril: Tensor  # shape: (n_gaussians, n_inputs, n_inputs)
    r"""PARAM: Parameters determining the covariances."""

    # non-permanent buffers
    eye: Tensor
    r"""BUFFER: Identity matrix."""
    covs: Tensor
    r"""BUFFER: Covariances of the Gaussian heads."""
    cholesky_factor: Tensor  # shape: (n_gaussians, n_inputs, n_inputs)
    r"""BUFFER: Cholesky factor of the covariance matrix."""
    samples: Tensor
    r"""BUFFER: Stored samples when sampling."""
    latents: Tensor
    r"""BUFFER: Inputs stored during the most recent `log_prob` evaluation."""
    log_probs: Tensor
    r"""BUFFER: Values stored during the most recent `log_prob` evaluation."""

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
        self.num_heads = n_heads
        self.num_features = n_feats
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
        r"""Compute the Cholesky factor of each covariance matrix."""
        lower = self.scale_tril.tril()
        diag = lower.diagonal(dim1=-2, dim2=-1)
        # need to make the diagonal positive
        new_diag = F.softplus(diag) + 1e-6  # (M, D)
        # (D, D), (M, D, 1), (M, D, D) -> (M, D, D)
        self.cholesky_factor = torch.where(self.eye, new_diag.unsqueeze(-1), lower)
        return self.cholesky_factor

    def get_covariance(self) -> Tensor:
        r"""Compute each covariance matrix from its Cholesky factor."""
        L = self.get_cholesky()  # M x D x D
        self.covs = torch.einsum("mij,mkj->mik", L, L)
        return self.covs

    def forward(self, x: Tensor, /) -> Tensor:
        r"""Apply the affine transformation $y = Lx + μ$.

        Args:
            x: Input tensor with shape `(..., H, D)`.

        Returns:
            Tensor: Transformed tensor with shape `(..., H, D)`.
        """
        L = self.get_cholesky()
        y = self.means + torch.einsum("...mj, mij -> ...mi", x, L)
        return y

    def inverse(self, y: Tensor, /) -> tuple[Tensor, Tensor]:
        r"""Apply the inverse affine transformation $x = L⁻¹(y - μ)$.

        Args:
            y: Input tensor with shape `(B, H, D)`.

        Returns:
            tuple[Tensor, Tensor]: Transformed tensor with shape `(B, H, D)` and
                the log absolute determinant of the inverse Jacobian with shape `(H,)`.
        """
        L = self.get_cholesky()

        # compute z = L⁻¹(x-μ)
        y = y - self.means
        y = y.movedim(0, -1)  # (B, H, D) -> (H, D, B)
        # (H, D, D), (H, D, B) -> (H, D, B)
        u = solve_triangular(L, y, upper=False)
        u = u.movedim(-1, 0)  # (H, D, B) -> (B, H, D)

        # log |det L⁻¹| = -∑ᵢ log Lᵢᵢ.
        ldj = -L.diagonal(dim1=-2, dim2=-1).log().sum(-1)
        return u, ldj

    def sample(self, size: int | tuple[int, ...] = (), /) -> Tensor:
        r"""Sample from the model.

        Args:
            size: Sample count or sample shape. An empty tuple requests one sample
                per head without a leading sample dimension.

        Returns:
            Tensor: Samples with shape `(*size, H, D)`.
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
            u: Input tensor with shape `(B, H, D)`.

        Returns:
            Tensor: Log densities with shape `(B, H)`.
        """
        self.latents = u  # store buffer for post-hoc analysis

        # parse through the gaussians
        z, ldj = self.inverse(u)

        # log p(u) = log 𝓝(z; 0, I) + log |det L⁻¹|.
        log_probs = -self.normalization_constant - 0.5 * vecdot(z, z, dim=-1)
        log_probs = log_probs + ldj  # (..., H)
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
