r"""Trace estimators."""

__all__ = [
    # Protocols
    "AbstractSampler",
    "AbstractTraceEstimator",
    "AbstractLogAbsDetEstimator",
    # Enums
    "Samplers",
    "TraceEstimators",
    "LogAbsDetEstimators",
    # samplers
    "GaussianSampler",
    "OrthSampler",
    "SignSampler",
    "SphereSampler",
    # estimators
    "ExactTrace",
    "ExactLogabsdet",
    "HutchPP_Estimator",
    "HutchinsonEstimator",
    "LogabsdetSeriesEstimator",
    "TraceEstimator",
    "XTraceEstimator",
    # functional api
    "exact_logabsdet",
    "exact_powers",
    "exact_trace",
    "hutchpp_estimator",
    "hutchinson_estimator",
    "logabsdet_series",
    "xtrace_estimator",
    "xtrace_estimator_matlab",
    "xtrace_naive_estimator",
]

import logging
import math
from abc import abstractmethod
from collections.abc import Callable as Fn, Iterator
from enum import StrEnum
from typing import Any, Final, Protocol, overload

import torch
from torch import Tensor, nn, vmap
from torch.func import linearize, vjp
from torch.linalg import qr, solve_triangular, vecdot, vector_norm

from signatures import signature

logging.basicConfig(level=logging.WARNING)
__logger__ = logging.getLogger(__package__)


class Samplers(StrEnum):
    r"""Enum of the provided probe vector samplers for stochastic trace estimators."""

    GAUSSIAN = "gaussian"
    SIGN = "sign"
    SPHERE = "sphere"
    ORTH = "orth"

    @classmethod
    def new(cls, sampler: str | AbstractSampler, /) -> AbstractSampler:
        r"""Construct a built-in sampler or forward a custom sampler as-is."""
        if callable(sampler):
            assert not isinstance(sampler, str)
            return sampler

        match Samplers(sampler):
            case Samplers.GAUSSIAN:
                return GaussianSampler()
            case Samplers.SIGN:
                return SignSampler()
            case Samplers.SPHERE:
                return SphereSampler()
            case Samplers.ORTH:
                return OrthSampler()
            case _:
                raise ValueError


class TraceEstimators(StrEnum):
    r"""Enum of the provided trace estimators."""

    EXACT = "exact"
    HUTCH = "hutch"
    HUTCH_PP = "hutch++"
    XTRACE = "xtrace"

    @classmethod
    def _missing_(cls, value: object) -> TraceEstimators | None:
        if value == "hutchinson":
            return cls.HUTCH
        return None

    @classmethod
    def new(
        cls,
        estimator: str,
        /,
        num_matvecs: int,
        mode: str,
        sampler: str | AbstractSampler,
        **kwargs: Any,
    ) -> TraceEstimator:
        match TraceEstimators(estimator):
            case TraceEstimators.EXACT:
                __logger__.warning("Estimator 'exact' was chosen, ignoring passed args")
                return ExactTrace()
            case TraceEstimators.HUTCH:
                return HutchinsonEstimator(num_matvecs, sampler=sampler, mode=mode)
            case TraceEstimators.HUTCH_PP:
                return HutchPP_Estimator(num_matvecs, sampler=sampler, mode=mode)
            case TraceEstimators.XTRACE:
                return XTraceEstimator(
                    num_matvecs, sampler=sampler, mode=mode, **kwargs
                )
            case _:
                raise ValueError


class LogAbsDetEstimators(StrEnum):
    r"""Enum of the provided log-absolute-determinant estimators."""

    EXACT = "exact"
    HUTCH = "hutch"
    HUTCH_PP = "hutch++"

    @classmethod
    def _missing_(cls, value: object) -> LogAbsDetEstimators | None:
        if value == "hutchinson":
            return cls.HUTCH
        return None

    @classmethod
    def new(
        cls,
        estimator: str,
        *,
        num_matvecs: int,
        num_terms: int,
        sampler: str | AbstractSampler = "sphere",
        mode: str = "reverse",
    ) -> ExactLogabsdet | LogabsdetSeriesEstimator:
        match e := LogAbsDetEstimators(estimator):
            case LogAbsDetEstimators.EXACT:
                __logger__.warning("Estimator 'exact' was chosen, ignoring passed args")
                return ExactLogabsdet()
            case _:
                return LogabsdetSeriesEstimator(
                    e.value,
                    num_matvecs=num_matvecs,
                    num_terms=num_terms,
                    sampler=sampler,
                    mode=mode,
                )


class AbstractSampler(Protocol):
    r"""Abstract probe vector sampler for stochastic trace estimators."""

    def __call__(
        self,
        shape: tuple[int, ...],
        num: int,
        *,
        dtype: torch.dtype,
        device: str | torch.device,
    ) -> Tensor: ...


class AbstractTraceEstimator(Protocol):
    r"""Protocol for Jacobian trace estimation."""

    @signature("[{(..., d) -> (..., d)}, (..., d)] -> (...)")
    def __call__(self, op: Fn[[Tensor], Tensor], x: Tensor, /) -> Tensor:
        r"""Returns an estimate of $\tr(𝐃f(x))$.

        Args:
            op: Function $f$ whose Jacobian trace should be estimated at $x$.
            x: Evaluation point. Its shape, dtype, and device define the domain.
        """
        ...


class AbstractLogAbsDetEstimator(Protocol):
    r"""Protocol for log-absolute-determinant Jacobian estimation."""

    @signature("[{(..., d) -> (..., d)}, (..., d)] -> [(..., d), (...)]")
    def __call__(self, op: Fn[[Tensor], Tensor], x: Tensor, /) -> tuple[Tensor, Tensor]:
        r"""Returns $f(x)$ and an estimate of $\log|\det(𝕀 + 𝐃f(x))|$.

        Args:
            op: Function $f$ whose Jacobian log-absolute-determinant should be estimated at $x$.
            x: Evaluation point. Its shape, dtype, and device define the domain.
        """
        ...


class GaussianSampler(nn.Module):
    r"""Sample $vᵢ∼𝓝(0,𝕀)$."""

    def forward(
        self,
        shape: tuple[int, ...],
        num: int,
        *,
        dtype: torch.dtype,
        device: str | torch.device,
    ) -> Tensor:
        r"""Sample $vᵢ∼𝓝(0,𝕀ₙ)$."""
        return torch.randn((*shape, num), device=device, dtype=dtype)


class SignSampler(nn.Module):
    r"""Sample $vᵢ∼Unif\{±1\}ⁿ$."""

    def forward(
        self,
        shape: tuple[int, ...],
        num: int,
        *,
        dtype: torch.dtype,
        device: str | torch.device,
    ) -> Tensor:
        values = torch.tensor([-1, +1], device=device, dtype=dtype)
        indices = torch.randint(0, 2, (*shape, num), device=device)
        return values[indices]


class SphereSampler(nn.Module):
    r"""Sample uniformly on sphere with radius $√n$."""

    def forward(
        self,
        shape: tuple[int, ...],
        num: int,
        *,
        dtype: torch.dtype,
        device: str | torch.device,
    ) -> Tensor:
        n = shape[-1]
        v = torch.randn((*shape, num), device=device, dtype=dtype)
        v = v * (math.sqrt(n) / vector_norm(v, dim=-2, keepdim=True))
        return v


class OrthSampler(nn.Module):
    r"""Sample orthogonal probe vectors with norm $√n$.

    Reduced QR produces at most $n$ vectors, even when more are requested.
    """

    def forward(
        self,
        shape: tuple[int, ...],
        num: int,
        *,
        dtype: torch.dtype,
        device: str | torch.device,
    ) -> Tensor:
        n = shape[-1]
        v = torch.randn((*shape, num), device=device, dtype=dtype)
        q, _ = qr(v, mode="reduced")
        return math.sqrt(n) * q


@signature(
    "[{(..., d) -> (..., d)}, (..., d), str] -> [(..., d), {(..., d, n) -> (..., d, n)}]"
)
def _make_batched_op(
    op: Fn[[Tensor], Tensor], x: Tensor, /, mode: str
) -> tuple[Tensor, Fn[[Tensor], Tensor]]:
    r"""Construct `op(x)` together with a batched Jacobian action $V ↦ AV$ or $V ↦ AᵀV$."""
    match mode:
        case "forward":
            y, jvp_fn = linearize(op, x)
            return y, vmap(jvp_fn, -1, -1)
        case "reverse":
            y, vjp_fn, *_ = vjp(op, x)
            return y, vmap(lambda z: vjp_fn(z)[0], -1, -1)
        case _:
            raise ValueError(f"invalid mode {mode!r}")


@signature("[{(..., d) -> (..., d)}, (..., d)] -> (...)")
def exact_trace(
    op: Fn[[Tensor], Tensor], x: Tensor, /, *, mode: str = "reverse"
) -> Tensor:
    r"""Estimate $\tr(𝐃f(x))$ by explicitly materializing the Jacobian.

    Args:
        op: Function $f$ whose Jacobian trace should be estimated at $x$.
        x: Evaluation point. Its shape, dtype, and device define the domain.
        mode: Whether to materialize the forward or reverse Jacobian action.

    Returns:
        The exact trace $\tr(𝐃f(x))$, computed from the full Jacobian.
    """
    dim = x.shape[-1]
    eye = torch.eye(dim, device=x.device, dtype=x.dtype).expand(*x.shape, dim)
    _, batched_op = _make_batched_op(op, x, mode)
    matrix = batched_op(eye)
    return torch.einsum("...ii -> ...", matrix)


@signature("[{(..., d) -> (..., d)}, (..., d)] -> (...)")
def exact_powers(
    op: Fn[[Tensor], Tensor], x: Tensor, /, max_power: int, *, mode: str = "reverse"
) -> Iterator[Tensor]:
    r"""Yield $\tr(𝐃f(x)ᵏ)$ for powers from one through `max_power`."""
    dim = x.shape[-1]
    eye = torch.eye(dim, device=x.device, dtype=x.dtype).expand(*x.shape, dim)
    _, batched_op = _make_batched_op(op, x, mode)
    matrix = batched_op(eye)
    eigenvalues = torch.linalg.eigvals(matrix)
    for power in range(1, max_power + 1):
        trace_power = eigenvalues.pow(power).sum(dim=-1)
        yield trace_power.real if not matrix.is_complex() else trace_power


@signature("[{(..., d) -> (..., d)}, (..., d)] -> (...)")
def exact_logabsdet(
    op: Fn[[Tensor], Tensor], x: Tensor, /, *, mode: str = "reverse"
) -> tuple[Tensor, Tensor]:
    r"""Compute $\log|\det(𝕀 + 𝐃f(x))|$ by materializing the Jacobian matrix.

    For the eigenvalues $λᵢ$ of $A = 𝐃f(x)$,

    $$ \log|\det(𝕀 + A)| = ∑ᵢ\log|1+λᵢ| $$

    This computation does not require $A$ to be a contraction.
    A singular $𝕀 + A$ yields $-∞$.
    Dense factorization costs $𝓞(N³)$, in addition to materializing the Jacobian.

    Args:
        op: Function $f$ whose Jacobian log-absolute-determinant is computed at $x$.
        x: Evaluation point. Its shape, dtype, and device define the domain.
        mode: Whether to materialize the forward or reverse Jacobian action.

    Returns:
        y: $f(x)$
        s: $\log|\det(𝕀 + 𝐃f(x))|$
    """
    dim = x.shape[-1]
    eye = torch.eye(dim, device=x.device, dtype=x.dtype).expand(*x.shape, dim)
    y, batched_op = _make_batched_op(op, x, mode)
    reverse_matrix = batched_op(eye)
    matrix = eye + reverse_matrix
    _, value = torch.linalg.slogdet(matrix)
    return y, value


@signature("[{(..., d) -> (..., d)}, (..., d)] -> (...)")
def logabsdet_series(
    op: Fn[[Tensor], Tensor], x: Tensor, /, num_terms: int, *, estimator: TraceEstimator
) -> tuple[Tensor, Tensor]:
    r"""Estimate $\log|\det(𝕀 + 𝐃f(x))|$ via a truncated power series.

    For $A = 𝐃f(x)$,

    $$ \log|\det(𝕀 + A)| = \Re(∑ₖ₌₁^∞ (-1)ᵏ⁺¹\tr(Aᵏ)/k) $$

    The series is truncated after `num_terms` terms. Each trace is replaced
    with the corresponding value from `estimator.powers`.

    Note:
        Assumes $𝐃f(x)$ is a contraction, i.e. $‖𝐃f(x)‖₂ < 1$.

    For dense Jacobian actions, the cost is $𝓞(kmN²)$, where $N$ is the domain
    dimension, $k$ is `num_terms`, and $m$ is the matrix-vector product budget
    per power. Actual cost depends on the operator and estimator.

    Args:
        op: Function $f$ whose Jacobian log-absolute-determinant is estimated at $x$.
        x: Evaluation point. Its shape, dtype, and device define the domain.
        num_terms: Number of power-series terms. Zero terms yield a zero estimate.
        estimator: Trace estimator supporting `powers`.

    Returns:
        y: $f(x)$
        ldj: estimated $\log|\det(𝕀 + 𝐃f(x))|$
    """
    y, jvp_fn = linearize(op, x)

    result = torch.zeros(x.shape[:-1], device=x.device, dtype=x.dtype)
    sign = 1.0
    for k, tr_k in enumerate(estimator.powers(jvp_fn, x, num_terms), start=1):
        # log(1 + x) = x - ½x² + ⅓x³ - ¼x⁴ + …
        result = result + (sign / k) * tr_k
        sign = -sign

    return y, result.real


@signature("[{(..., d) -> (..., d)}, (..., d)] -> (...)")
def hutchinson_estimator(
    op: Fn[[Tensor], Tensor],
    x: Tensor,
    /,
    num_matvecs: int,
    *,
    sampler: AbstractSampler,
    mode: str = "reverse",
) -> Tensor:
    r"""Estimate $\tr(𝐃f(x))$ with Hutchinson's estimator.

    For probes with $𝐄[vvᵀ] = 𝕀$,

    $$ \tr(A) = 𝐄[vᵀAv] $$

    Args:
        op: Function $f$ whose Jacobian trace should be estimated at $x$.
        x: Evaluation point. Its shape, dtype, and device define the domain.
        num_matvecs: Number of matrix-vector products,
            equal to the number of probe vectors. Must be at least one.
        sampler: Callable that produces probe vectors.
        mode: Whether to apply forward or reverse Jacobian actions.

    Returns:
        A Hutchinson estimate of $\tr(𝐃f(x))$.
    """
    if num_matvecs < 1:
        raise ValueError("num_samples must be at least 1")

    probes = sampler(x.shape, num_matvecs, dtype=x.dtype, device=x.device)
    _, batched_op = _make_batched_op(op, x, mode)
    reverse_probes = batched_op(probes)
    estimate = vecdot(probes, reverse_probes, dim=-2).mean(dim=-1)
    return estimate


@signature("[{(..., d) -> (..., d)}, (..., d)] -> (...)")
def hutchpp_estimator(
    op: Fn[[Tensor], Tensor],
    x: Tensor,
    /,
    num_matvecs: int,
    *,
    sampler: AbstractSampler,
    mode: str = "reverse",
) -> Tensor:
    r"""Estimate $\tr(𝐃f(x))$ with Hutch++.

    $$ \tr(A) = \tr(QᵀAQ) + 𝐄[vᵀ(𝕀-QQᵀ)A(𝕀-QQᵀ)v] $$

    Args:
        op: Function $f$ whose Jacobian trace should be estimated at $x$.
        x: Evaluation point. Its shape, dtype, and device define the domain.
        num_matvecs: Total matrix-vector product budget.
        sampler: Callable that produces probe vectors.
        mode: Whether to apply forward or reverse Jacobian actions.

    Returns:
        A Hutch++ estimate of $\tr(𝐃f(x))$.
    """
    if num_matvecs < 3:
        raise ValueError("num_matvecs must be at least 3")

    num_samples = num_matvecs // 3
    samples = sampler(x.shape, num_samples, device=x.device, dtype=x.dtype)
    residual_samples = sampler(x.shape, num_samples, device=x.device, dtype=x.dtype)

    _, batched_op = _make_batched_op(op, x, mode)
    sketch = batched_op(samples)
    q, _ = qr(sketch, mode="reduced")  # (...dr)

    residual_samples = residual_samples - q @ (q.mH @ residual_samples)
    low_rank_q = batched_op(q)
    residual_action = batched_op(residual_samples)
    low_rank = vecdot(q, low_rank_q, dim=-2).sum(dim=-1)
    residual = vecdot(residual_samples, residual_action, dim=-2).mean(dim=-1)
    estimate = low_rank + residual
    return estimate.real if not estimate.is_complex() else estimate


@signature("[{(..., d) -> (..., d)}, (..., d)] -> (...)")
def xtrace_naive_estimator(
    op: Fn[[Tensor], Tensor],
    x: Tensor,
    /,
    num_matvecs: int,
    *,
    sampler: AbstractSampler,
    mode: str = "reverse",
    renormalize: bool = True,
) -> Tensor:
    r"""Naive XTrace estimate of $\tr(𝐃f(x))$ for debugging."""
    if num_matvecs < 2:
        raise ValueError("num_matvecs must be at least 2")

    num_samples = num_matvecs // 2
    *batch, N = x.shape
    k = min(N, num_samples)

    _, batched_op = _make_batched_op(op, x, mode)
    tr = torch.zeros(batch, dtype=x.dtype, device=x.device)

    samples = sampler(x.shape, k, device=x.device, dtype=x.dtype)
    Y = batched_op(samples)  # (...Nm)

    mus = []
    for i in range(k):
        col_indices = torch.arange(k, device=Y.device)
        Q_i, _ = qr(Y[..., i != col_indices], mode="reduced")
        ω_i = samples[..., [i]]
        μ_i = ω_i - Q_i @ (Q_i.mH @ ω_i)
        mus.append(μ_i)
        tr = tr + vecdot(Q_i, batched_op(Q_i), dim=-2).sum(dim=-1)
    μ = torch.cat(mus, dim=-1)
    μ_norm_sq = vecdot(μ, μ, dim=-2)
    scale = 1.0 - renormalize * (1.0 - (N - k + 1) / μ_norm_sq)
    μ = μ * scale.unsqueeze(-2)
    residual = vecdot(μ, batched_op(μ), dim=-2).mean(dim=-1)
    return tr / k + residual


@signature("[{(..., d) -> (..., d)}, (..., d)] -> (...)")
def xtrace_estimator(
    op: Fn[[Tensor], Tensor],
    x: Tensor,
    /,
    num_matvecs: int,
    *,
    sampler: AbstractSampler,
    mode: str = "reverse",
    renormalize: bool = True,
) -> Tensor:
    r"""Estimate $\tr(𝐃f(x))$ with the fast XTrace estimator.

    Args:
        op: Function $f$ whose Jacobian trace should be estimated at $x$.
        x: Evaluation point. Its shape, dtype, and device define the domain.
        num_matvecs: Total matrix-vector product budget. XTrace uses
            `num_matvecs // 2` probe vectors internally.
        sampler: Callable that produces probe vectors.
        mode: Whether to apply forward or reverse Jacobian actions.
        renormalize: Whether to apply the XTrace renormalization correction.

    Returns:
        An XTrace estimate of $\tr(𝐃f(x))$.
    """
    _, batched_op = _make_batched_op(op, x, mode)
    if num_matvecs < 2:
        raise ValueError("num_matvecs must be at least 2")

    *_, N = x.shape
    num_samples = num_matvecs // 2
    k = min(N, num_samples)

    samples = sampler(x.shape, k, device=x.device, dtype=x.dtype)
    Y = batched_op(samples)  # (...dk)
    Q, R = qr(Y, mode="reduced")  # (...dk), (...kk)
    # Q has normalized cols <-> Q.norm(dim=-2) = 1

    Z = batched_op(Q)  # (...dk)
    H = Q.mH @ Z  # (...kk)
    W = Q.mH @ samples  # (...kk)
    T = Z.mH @ samples  # (...kk)

    identity = torch.eye(k, dtype=samples.dtype, device=samples.device)
    # solve R^* S = Iₖ
    S = solve_triangular(R.mH, identity, upper=False, left=True)  # (...kk)
    # normalize COLS
    S = S / vector_norm(S, dim=-2, keepdim=True)  # (...kk)

    sw = vecdot(S, W, dim=-2)  # (...k)
    if renormalize:
        scale = (N - k + 1) / (
            vecdot(samples, samples, dim=-2) - vecdot(W, W, dim=-2) + sw.abs().square()
        )
    else:
        scale = 1.0

    shs = torch.einsum("...ik, ...kl, ...li -> ...i", S.mH, H, S)
    hw = H @ W
    term1 = sw.abs().square() * shs  # |⟨sᵢ∣wᵢ⟩|²⟨sᵢ∣Hsᵢ⟩
    term2 = sw.conj() * vecdot(S, R - hw, dim=-2)  # ⟨wᵢ∣sᵢ⟩⟨sᵢ∣rᵢ - Hwᵢ⟩
    x_term = W - sw.unsqueeze(-2) * S
    term3 = -vecdot(T - H.mH @ W, x_term, dim=-2)  # -⟨tᵢ - Hᴴwᵢ∣wᵢ - ⟨sᵢ∣wᵢ⟩sᵢ⟩
    trs = -shs + scale * (term1 + term2 + term3)

    estimate = H.diagonal(dim1=-2, dim2=-1).sum(dim=-1) + trs.mean(dim=-1)
    return estimate


@signature("[{(..., d) -> (..., d)}, (..., d), int] -> (...)")
def xtrace_estimator_matlab(
    op: Fn[[Tensor], Tensor],
    x: Tensor,
    /,
    num_matvecs: int,
    *,
    sampler: AbstractSampler,
    mode: str = "reverse",
    renormalize: bool = False,
) -> Tensor:
    r"""Estimate the trace of a matrix using the original XTrace MATLAB algorithm.

    This is a direct Torch transcription of the reference MATLAB code.
    """
    if num_matvecs < 2:
        raise ValueError("num_matvecs must be at least 2")

    samples = sampler(x.shape, num_matvecs // 2, dtype=x.dtype, device=x.device)
    _, batched_op = _make_batched_op(op, x, mode)

    *_, d, m = samples.shape

    # MATLAB: Om = sqrt(N) * cnormc(randn(N, m))
    # Here we reuse the provided probes as the m columns of Ω.
    # Omega: (..., d, m)
    omega = d**0.5 * samples.mH / vector_norm(samples.mH, dim=-2, keepdim=True)

    # MATLAB: Y = A * Om
    # Y: (..., d, m)
    y = batched_op(omega.mH).mH
    # MATLAB: [Q, R] = qr(Y, 0)
    # Q: (..., d, m), R: (..., m, m)
    q, r = qr(y, mode="reduced")

    # MATLAB: W = Q' * Om
    # W: (..., m, m)
    w = torch.einsum("...dm, ...dn -> ...mn", q.conj(), omega)

    # MATLAB: S = cnormc(inv(R)')
    # S: (..., m, m), columns of (R^{-1})ᴴ normalized to unit norm.
    identity = torch.eye(m, dtype=samples.dtype, device=samples.device)
    s = solve_triangular(r.mH, identity, upper=False)
    s = s / vector_norm(s, dim=-2, keepdim=True)

    # MATLAB:
    # scale = (N - m + 1) ./ (N - ||w_i||² + |<s_i, w_i> ||s_i|| |²)
    # column norms / diagonal products: (..., m)
    w_norm_sq = vector_norm(w, dim=-2).square()
    s_norm = vector_norm(s, dim=-2)
    d_sw = torch.einsum("...dm, ...dm -> ...m", s.conj(), w)
    scale = (d - m + 1) / (d - w_norm_sq + (d_sw * s_norm).abs().square())
    scale = scale if renormalize else 1.0

    # MATLAB: Z = A * Q
    # Z: (..., d, m)
    z = batched_op(q.mH).mH
    # MATLAB: H = Q' * Z
    # H: (..., m, m)
    h = torch.einsum("...dm, ...dn -> ...mn", q.conj(), z)
    # MATLAB: HW = H * W
    # HW: (..., m, m)
    hw = h @ w
    # MATLAB: T = Z' * Om
    # T: (..., m, m)
    t = torch.einsum("...dm, ...dn -> ...mn", z.conj(), omega)

    # Column-wise diagonal contractions used by the estimator correction terms.
    # All shapes below are (..., m).
    d_shs = torch.einsum("...dm, ...dm -> ...m", s.conj(), h @ s)
    d_tw = torch.einsum("...dm, ...dm -> ...m", t.conj(), w)
    d_whw = torch.einsum("...dm, ...dm -> ...m", w.conj(), hw)
    d_s_r_minus_hw = torch.einsum("...dm, ...dm -> ...m", s.conj(), r - hw)
    d_t_minus_hhw_s = torch.einsum("...dm, ...dm -> ...m", (t - h.mH @ w).conj(), s)

    # MATLAB:
    # ests_i = tr(H)
    #        - <sᵢ, H sᵢ>
    #        + ( <wᵢ, H wᵢ> - <tᵢ, wᵢ>
    #            + <tᵢ - H' wᵢ, sᵢ><sᵢ, wᵢ>
    #            + |<sᵢ, wᵢ>|² <sᵢ, H sᵢ>
    #            + conj(<sᵢ, wᵢ>) <sᵢ, rᵢ - H wᵢ> ) * scaleᵢ
    trace_h = h.diagonal(dim1=-2, dim2=-1).sum(dim=-1, keepdim=True)
    ests = (
        trace_h
        - d_shs
        + (
            d_whw
            - d_tw
            + d_t_minus_hhw_s * d_sw
            + d_sw.abs().square() * d_shs
            + d_sw.conj() * d_s_r_minus_hw
        )
        * scale
    )

    # MATLAB: t = mean(ests)
    return ests.mean(dim=-1)


class TraceEstimator(nn.Module):
    r"""Base class for Jacobian trace estimators.

    Concrete estimators operate on a function `f` together with an evaluation point `x`.
    In the common case, `f` is a nonlinear map and the estimator approximates trace-like
    quantities of its Jacobian $𝐃f(x)$. Linear operators fit this API as a special case:
    when $f(z) = Az$, the Jacobian is constant and equal to $A$.

    Subclasses must implement `forward`, which estimates $\tr(𝐃f(x))$,
    and `powers`, which yields estimates for powers of the Jacobian or raises
    `NotImplementedError` if power traces are unsupported.
    """

    @signature("[{(..., d) -> (..., d)}, (..., d)] -> (...)")
    @abstractmethod
    def forward(self, op: Fn[[Tensor], Tensor], x: Tensor, /) -> Tensor:
        r"""Return an estimate of $\tr(𝐃f(x))$.

        Args:
            op: Function $f$ whose Jacobian trace should be estimated at $x$.
            x: Evaluation point. Its shape, dtype, and device define the domain.
        """
        raise NotImplementedError

    @signature("[{(..., d) -> (..., d)}, (..., d)] -> (...)")
    @abstractmethod
    def powers(
        self, op: Fn[[Tensor], Tensor], x: Tensor, /, max_power: int
    ) -> Iterator[Tensor]:
        r"""Yield estimates of $\tr(𝐃f(x)ᵏ)$ through `max_power`.

        Subclasses must provide an implementation or raise `NotImplementedError`
        when power traces are unsupported.

        Args:
            op: Function $f$ whose Jacobian power traces should be estimated at $x$.
            x: Evaluation point. Its shape, dtype, and device define the domain.
            max_power: Largest Jacobian power to estimate.
        """
        raise NotImplementedError


class ExactTrace(TraceEstimator):
    r"""Estimate traces by explicitly materializing the Jacobian.

    This `nn.Module` wrapper implements the same estimator as
    `exact_trace` and exposes `powers` and `logabsdet`, both derived from
    the exact Jacobian. Computing the trace costs $𝓞(N)$ after materializing
    the Jacobian; `powers` and `logabsdet` use dense factorizations costing
    $𝓞(N³)$, where $N$ is the domain dimension.
    """

    @signature("[{(..., d) -> (..., d)}, (..., d)] -> (...)")
    def forward(self, op: Fn[[Tensor], Tensor], x: Tensor, /) -> Tensor:
        return exact_trace(op, x)

    @signature("[{(..., d) -> (..., d)}, (..., d)] -> (...)")
    def powers(
        self, op: Fn[[Tensor], Tensor], x: Tensor, /, max_power: int
    ) -> Iterator[Tensor]:
        yield from exact_powers(op, x, max_power)

    @signature("[{(..., d) -> (..., d)}, (..., d)] -> [(..., d), (...)]")
    def logabsdet(
        self, op: Fn[[Tensor], Tensor], x: Tensor, /
    ) -> tuple[Tensor, Tensor]:
        r"""Computes $f(x)$ and $\log|\det(𝕀+𝐃f(x))|$ from the materialized Jacobian."""
        return exact_logabsdet(op, x)


class ExactLogabsdet(nn.Module):
    r"""Compute $\log|\det(𝕀 + 𝐃f(x))|$ by materializing the Jacobian matrix.

    For the eigenvalues $λᵢ$ of $A = 𝐃f(x)$,

    $$ \log|\det(𝕀 + A)| = ∑ᵢ\log|1+λᵢ| $$

    This wraps `exact_logabsdet` and does not require a contraction.
    A singular $𝕀 + A$ yields $-∞$. Dense factorization costs $𝓞(N³)$,
    in addition to materializing the Jacobian.
    """

    @signature("[{(..., d) -> (..., d)}, (..., d)] -> (...)")
    def forward(self, op: Fn[[Tensor], Tensor], x: Tensor, /) -> tuple[Tensor, Tensor]:
        return exact_logabsdet(op, x)


class HutchinsonEstimator(TraceEstimator):
    r"""Estimate traces with Hutchinson's estimator.

    For dense Jacobian actions, each estimate costs $𝓞(mN²)$, where
    $m$ is `num_samples` (equal to `num_matvecs`) and $N$ is the domain dimension.
    The orthogonal sampler adds a QR factorization cost of $𝓞(m²N)$ for $m ≤ N$.

    This module wraps the same trace estimator as `hutchinson_estimator`.
    The `forward` method estimates $\tr(𝐃f(x))$ and `powers` extends the
    same sampling scheme to powers of the Jacobian by repeatedly applying Jacobian
    or reverse actions according to `mode`.

    Args:
        num_matvecs: Number of matrix-vector products, equal to the number of probe
            vectors used per estimate.
        sampler: Probe sampler, either a built-in sampler name or a custom callable.
        mode: Whether to use forward Jacobian-vector products, reverse vector-Jacobian
            products, or a symmetric alternating scheme.
    """

    MODES: Final[frozenset[str]] = frozenset({"forward", "reverse", "symmetric"})

    num_matvecs: Final[int]
    num_samples: Final[int]
    mode: Final[str]
    sampler: Final[AbstractSampler]

    @overload
    def __init__(
        self, num_matvecs: int, *, sampler: str | AbstractSampler = ..., mode: str = ...
    ) -> None: ...
    @overload
    def __init__(
        self, *, num_samples: int, sampler: str | AbstractSampler = ..., mode: str = ...
    ) -> None: ...
    def __init__(
        self,
        num_matvecs: int | None = None,
        *,
        num_samples: int | None = None,
        sampler: str | AbstractSampler = Samplers.SPHERE,
        mode: str = "reverse",
    ) -> None:
        match num_samples, num_matvecs:
            case None, None:
                raise ValueError("Expected one of num_samples or num_matvecs")
            case int(m), None:
                num_matvecs = m
                num_samples = m
            case None, int(n):
                num_matvecs = n
                num_samples = n
            case _:
                raise ValueError("Expected one of num_samples or num_matvecs")
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {self.MODES}, got {mode!r}")

        super().__init__()
        self.num_matvecs = num_matvecs
        self.num_samples = num_samples
        self.mode = mode
        self.sampler = Samplers.new(sampler)

    @signature("[{(..., d) -> (..., d)}, (..., d)] -> (...)")
    def forward(self, op: Fn[[Tensor], Tensor], x: Tensor, /) -> Tensor:
        r"""Return a Hutchinson estimate of $\tr(𝐃f(x))$.

        Args:
            op: Function $f$ whose Jacobian trace should be estimated at $x$.
            x: Evaluation point. Its shape, dtype, and device define the domain.
        """
        return next(self.powers(op, x, 1))

    @signature("[{(..., d) -> (..., d)}, (..., d)] -> (...)")
    def powers(
        self, op: Fn[[Tensor], Tensor], x: Tensor, /, max_power: int
    ) -> Iterator[Tensor]:
        r"""Yield Hutchinson estimates of $\tr(𝐃f(x)ᵏ)$ through `max_power`.

        Args:
            op: Function $f$ whose Jacobian power traces should be estimated at $x$.
            x: Evaluation point. Its shape, dtype, and device define the domain.
            max_power: Largest Jacobian power to estimate.
        """
        right_samples = self.sampler(
            x.shape,
            self.num_samples,
            device=x.device,
            dtype=x.dtype,
        )
        left_samples = right_samples.clone()

        match self.mode:
            case "forward":
                # use x ↦ Ax only
                _, batched_jvp_fn = _make_batched_op(op, x, "forward")

                for _ in range(max_power):
                    right_samples = batched_jvp_fn(right_samples)
                    yield vecdot(left_samples, right_samples, dim=-2).mean(dim=-1)

            case "reverse":
                # use x ↦ Aᵀx only
                _, batched_vjp_fn = _make_batched_op(op, x, "reverse")

                for _ in range(max_power):
                    left_samples = batched_vjp_fn(left_samples)
                    yield vecdot(left_samples, right_samples, dim=-2).mean(dim=-1)

            case "symmetric":
                # alternate between Ax and Aᵀx, which is good for forward sensitivity.
                # as it grows exponentially in the number of matvecs.
                _, batched_jvp_fn = _make_batched_op(op, x, "forward")
                _, batched_vjp_fn = _make_batched_op(op, x, "reverse")

                power = 0
                while power < max_power:
                    right_samples = batched_jvp_fn(right_samples)
                    power += 1
                    yield vecdot(left_samples, right_samples, dim=-2).mean(dim=-1)

                    if power == max_power:
                        break

                    left_samples = batched_vjp_fn(left_samples)
                    power += 1
                    yield vecdot(left_samples, right_samples, dim=-2).mean(dim=-1)
            case _:
                raise ValueError(f"invalid mode {self.mode!r}")


class HutchPP_Estimator(TraceEstimator):
    r"""Estimate traces with the Hutch++ variance-reduced estimator.

    For dense Jacobian actions, the cost is $𝓞(mN² + m²N + m³)$, where
    $m$ is three times `num_samples` and $N$ is the domain dimension.

    This module wraps the same trace estimator as `hutchpp_estimator`.
    The `forward` method estimates $\tr(𝐃f(x))$ and `powers` reuses the
    same low-rank-plus-residual decomposition for powers of the Jacobian.

    Args:
        num_matvecs: Total matrix-vector product budget. The estimator uses `num_matvecs // 3`
            probe vectors for the sketch and the same number for the residual term.
        sampler: Probe sampler, either a built-in sampler name or a custom callable.
        mode: Whether to use forward Jacobian-vector products, reverse vector-Jacobian
            products, or a symmetric alternating scheme.

    References:
        - Hutch++: Optimal Stochastic Trace Estimation \
          Raphael A. Meyer, Cameron Musco, Christopher Musco, and David P. Woodruff. \
          Symposium on Simplicity in Algorithms (SOSA), 2021. \
          <https://doi.org/10.1137/1.9781611976496.16>
    """

    MODES: Final[frozenset[str]] = frozenset({"forward", "reverse", "symmetric"})

    num_matvecs: Final[int]
    num_samples: Final[int]
    mode: Final[str]
    sampler: Final[AbstractSampler]

    @overload
    def __init__(
        self, num_matvecs: int, *, sampler: str | AbstractSampler = ..., mode: str = ...
    ) -> None: ...
    @overload
    def __init__(
        self, *, num_samples: int, sampler: str | AbstractSampler = ..., mode: str = ...
    ) -> None: ...
    def __init__(
        self,
        num_matvecs: int | None = None,
        *,
        num_samples: int | None = None,
        sampler: str | AbstractSampler = Samplers.SPHERE,
        mode: str = "reverse",
    ) -> None:
        match num_samples, num_matvecs:
            case None, None:
                raise ValueError("Expected one of num_samples or num_matvecs")
            case int(m), None:
                num_matvecs = m * 3
                num_samples = m
            case None, int(n):
                num_matvecs = n
                num_samples = n // 3
            case _:
                raise ValueError("Expected one of num_samples or num_matvecs")
        if num_matvecs < 3:
            raise ValueError("num_matvecs must be at least 3")
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {self.MODES}, got {mode!r}")

        super().__init__()
        self.num_matvecs = num_matvecs
        self.num_samples = num_samples
        self.mode = mode
        self.sampler = Samplers.new(sampler)

    @signature("[{(..., d) -> (..., d)}, (..., d)] -> (...)")
    def forward(self, op: Fn[[Tensor], Tensor], x: Tensor, /) -> Tensor:
        r"""Return a Hutch++ estimate of $\tr(𝐃f(x))$.

        Args:
            op: Function $f$ whose Jacobian trace should be estimated at $x$.
            x: Evaluation point. Its shape, dtype, and device define the domain.
        """
        return next(self.powers(op, x, 1))

    @signature("[{(..., d) -> (..., d)}, (..., d)] -> (...)")
    def powers(
        self, op: Fn[[Tensor], Tensor], x: Tensor, /, max_power: int
    ) -> Iterator[Tensor]:
        r"""Yield Hutch++ estimates of $\tr(𝐃f(x)ᵏ)$ through `max_power`.

        Args:
            op: Function $f$ whose Jacobian power traces should be estimated at $x$.
            x: Evaluation point. Its shape, dtype, and device define the domain.
            max_power: Largest Jacobian power to estimate.
        """
        samples = self.sampler(
            x.shape,
            self.num_samples,
            device=x.device,
            dtype=x.dtype,
        )
        residual_samples = self.sampler(
            x.shape,
            self.num_samples,
            device=x.device,
            dtype=x.dtype,
        )

        match self.mode:
            case "forward":
                _, batched_jvp_fn = _make_batched_op(op, x, "forward")
                sketch = batched_jvp_fn(samples)
                Q, _ = qr(sketch, mode="reduced")  # (...dr)

                projected_samples = Q
                projected_l = projected_samples.clone()
                projected_r = projected_samples.clone()
                residual_samples = residual_samples - Q @ (Q.mH @ residual_samples)
                residual_l = residual_samples.clone()
                residual_r = residual_samples.clone()

                for _ in range(max_power):
                    projected_r = batched_jvp_fn(projected_r)
                    residual_r = batched_jvp_fn(residual_r)

                    low_rank = vecdot(projected_l, projected_r, dim=-2).sum(dim=-1)
                    residual = vecdot(residual_l, residual_r, dim=-2).mean(dim=-1)
                    yield low_rank + residual

            case "reverse":
                _, batched_vjp_fn = _make_batched_op(op, x, "reverse")
                sketch = batched_vjp_fn(samples)
                Q, _ = qr(sketch, mode="reduced")  # (...dr)

                projected_samples = Q
                projected_l = projected_samples.clone()
                projected_r = projected_samples.clone()
                residual_samples = residual_samples - Q @ (Q.mH @ residual_samples)
                residual_l = residual_samples.clone()
                residual_r = residual_samples.clone()

                for _ in range(max_power):
                    projected_l = batched_vjp_fn(projected_l)
                    residual_l = batched_vjp_fn(residual_l)

                    low_rank = vecdot(projected_l, projected_r, dim=-2).sum(dim=-1)
                    residual = vecdot(residual_l, residual_r, dim=-2).mean(dim=-1)
                    yield low_rank + residual

            case "symmetric":
                _, batched_jvp_fn = _make_batched_op(op, x, "forward")
                _, batched_vjp_fn = _make_batched_op(op, x, "reverse")

                # Hutch++ uses a fixed projector P = QQᵀ and the exact split
                #   tr(Aᵏ) = tr(Qᵀ Aᵏ Q) + tr((I-P) Aᵏ (I-P)).
                left_sketch, right_sketch = torch.tensor_split(samples, 2, dim=-1)
                left_sketch = batched_vjp_fn(left_sketch)
                right_sketch = batched_jvp_fn(right_sketch)
                sketch = torch.cat([left_sketch, right_sketch], dim=-1)
                Q, _ = qr(sketch, mode="reduced")  # (...dr)

                projected_samples = Q
                projected_l = projected_samples.clone()
                projected_r = projected_samples.clone()
                residual_samples = residual_samples - Q @ (Q.mH @ residual_samples)
                residual_l = residual_samples.clone()
                residual_r = residual_samples.clone()

                power = 0
                while power < max_power:
                    projected_r = batched_jvp_fn(projected_r)
                    residual_r = batched_jvp_fn(residual_r)
                    power += 1

                    low_rank = vecdot(projected_l, projected_r, dim=-2).sum(dim=-1)
                    residual = vecdot(residual_l, residual_r, dim=-2).mean(dim=-1)
                    yield low_rank + residual

                    if power == max_power:
                        break

                    projected_l = batched_vjp_fn(projected_l)
                    residual_l = batched_vjp_fn(residual_l)
                    power += 1

                    low_rank = vecdot(projected_l, projected_r, dim=-2).sum(dim=-1)
                    residual = vecdot(residual_l, residual_r, dim=-2).mean(dim=-1)
                    yield low_rank + residual

            case _:
                raise ValueError(f"invalid mode {self.mode!r}")


class XTraceEstimator(TraceEstimator):
    r"""Estimate traces with the XTrace estimator.

    This module wraps the same trace estimator as `xtrace_estimator` and supports
    both forward and reverse Jacobian actions. It does not support Jacobian power
    traces.

    Args:
        num_matvecs: Total matrix-vector product budget.
            XTrace uses `num_matvecs // 2` probe vectors internally.
        sampler: Probe sampler, either a built-in sampler name or a custom callable.
        renormalize: Whether to apply the paper's renormalization.
        mode: Jacobian action mode. Must be `"forward"` or `"reverse"`.

    For dense Jacobian actions, the cost is $𝓞(mN² + m²N + m³)$, where
    $m$ is twice `num_samples` and $N$ is the domain dimension.

    References:
        - XTrace: Making the Most of Every Sample in Stochastic Trace Estimation \
          Ethan N. Epperly, Joel A. Tropp, and Robert J. Webber. \
          SIAM Journal on Matrix Analysis and Applications, 2024. \
          <https://doi.org/10.1137/23M1548323>

    Note: Core idea
        Given probe columns $W = [w₁, …, wₖ]$, let $W₋ᵢ$ omit the $i$th column
        and set $Qᵢ = \orth(AW₋ᵢ)$. Each leave-one-out estimate is

        $$ \tr(QᵢᴴAQᵢ) + wᵢᴴ(𝕀-QᵢQᵢᴴ)A(𝕀-QᵢQᵢᴴ)wᵢ $$

        A rank-one update, $QᵢQᵢᴴ = Q(𝕀-sᵢsᵢᴴ)Qᴴ$, avoids recomputing
        every QR factorization. Here $sᵢ$ is the normalized $i$th column of $R⁻ᴴ$.

    Example: Algorithm without renormalization
        ```text
        1. Set k ← min(N, m//2) and draw Ω ∼ Unif{±1} with shape N×k
        2. Y ← AΩ
        3. (Q, R) ← qr(Y, 'econ')
        4. Z ← AQ
        5. H ← QᴴZ, W ← QᴴΩ, T ← ZᴴΩ
        6. S ← R⁻ᴴ
        7. Normalize the columns of S to unit norm
        8. for i = 1 … k do
        9.     xᵢ ← wᵢ − ⟨sᵢ∣wᵢ⟩·sᵢ
        10.    trᵢ ← tr(H) − ⟨sᵢ|H sᵢ⟩
                   + ⟨wᵢ∣sᵢ⟩·⟨sᵢ∣rᵢ⟩ − ⟨tᵢ|xᵢ⟩ + ⟨xᵢ|Hxᵢ⟩
        11. end for
        12. tr ← mean(trᵢ: i=1…k)
        ```

        Here $wᵢ$, $tᵢ$, and $rᵢ$ are columns of $W$, $T$, and $R$.
        The actual probe distribution is selected by `sampler`.
    """

    MODES: Final[frozenset[str]] = frozenset({"forward", "reverse"})

    num_matvecs: Final[int]
    num_samples: Final[int]
    renormalize: Final[bool]
    r"""Whether to apply the renormalization from Section 2.3 of the XTrace paper."""
    sampler: Final[AbstractSampler]
    mode: Final[str]

    @overload
    def __init__(
        self,
        num_matvecs: int,
        *,
        sampler: str | AbstractSampler = ...,
        mode: str = ...,
        renormalize: bool = ...,
    ) -> None: ...
    @overload
    def __init__(
        self,
        *,
        num_samples: int,
        sampler: str | AbstractSampler = ...,
        mode: str = ...,
        renormalize: bool = ...,
    ) -> None: ...
    def __init__(
        self,
        num_matvecs: int | None = None,
        *,
        num_samples: int | None = None,
        sampler: str | AbstractSampler = Samplers.SPHERE,
        mode: str = "reverse",
        renormalize: bool = True,
    ) -> None:
        match num_samples, num_matvecs:
            case None, None:
                raise ValueError("Expected one of num_samples or num_matvecs")
            case int(m), None:
                num_matvecs = m * 2
                num_samples = m
            case None, int(n):
                num_matvecs = n
                num_samples = n // 2
            case _:
                raise ValueError("Expected one of num_samples or num_matvecs")
        if num_matvecs < 2:
            raise ValueError("num_matvecs must be at least 2")
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {self.MODES}, got {mode!r}")

        super().__init__()
        self.num_matvecs = num_matvecs
        self.num_samples = num_samples
        self.renormalize = renormalize
        self.mode = mode
        self.sampler = Samplers.new(sampler)

    @signature("[{(..., d) -> (..., d)}, (..., d)] -> (...)")
    def forward(self, op: Fn[[Tensor], Tensor], x: Tensor, /) -> Tensor:
        r"""Return an XTrace estimate of $\tr(𝐃f(x))$.

        Args:
            op: Function $f$ whose Jacobian trace should be estimated at $x$.
            x: Evaluation point. Its shape, dtype, and device define the domain.
        """
        return xtrace_estimator(
            op,
            x,
            self.num_matvecs,
            sampler=self.sampler,
            mode=self.mode,
            renormalize=self.renormalize,
        )

    @signature("[{(..., d) -> (..., d)}, (..., d)] -> (...)")
    def estimate_naive(self, op: Fn[[Tensor], Tensor], x: Tensor, /) -> Tensor:
        r"""Estimate $\tr(𝐃f(x))$ with the naive XTrace formulation.

        This method is mainly useful for debugging against the optimized
        implementation in `forward`.
        """
        return xtrace_naive_estimator(
            op,
            x,
            self.num_matvecs,
            sampler=self.sampler,
            mode=self.mode,
            renormalize=self.renormalize,
        )

    @signature("[{(..., d) -> (..., d)}, (..., d)] -> (...)")
    def powers(
        self, op: Fn[[Tensor], Tensor], x: Tensor, /, max_power: int
    ) -> Iterator[Tensor]:
        raise NotImplementedError("XTraceEstimator does not support power traces")


class LogabsdetSeriesEstimator(nn.Module):
    r"""Estimate $\log|\det(𝕀 + 𝐃f(x))|$ with a trace-estimator backend.

    For $A = 𝐃f(x)$,

    $$ \log|\det(𝕀 + A)| = \Re(∑ₖ₌₁^∞ (-1)ᵏ⁺¹\tr(Aᵏ)/k) $$

    Assumes $𝐃f(x)$ is a contraction, i.e. $‖𝐃f(x)‖₂ < 1$.

    Note:
        For a matrix logarithm $B$ with $ℯᴮ = A$,
        $\log|\det A| = \Re(\tr(B))$.
        For nonsingular $A$, this also equals $½\tr(\log(AᴴA))$.
        Singular matrices have log-absolute-determinant $-∞$.

    Args:
        estimator: Backend name: `"exact"`, `"hutch"` (or `"hutchinson"`),
            or `"hutch++"`. The backend must support Jacobian power traces;
            `"xtrace"` raises `NotImplementedError` during evaluation.
        num_matvecs: Budget of matrix-vector multiplications per series term.
        num_terms: Number of power-series terms.
        sampler: Probe sampler name or custom callable used by stochastic backends.
        mode: Jacobian action mode supported by the selected backend.
    """

    estimator: Final[TraceEstimator]
    num_matvecs: int
    num_terms: int

    def __init__(
        self,
        estimator: str,
        *,
        num_matvecs: int,
        num_terms: int,
        sampler: str | AbstractSampler = "sphere",
        mode: str = "reverse",
    ) -> None:
        super().__init__()
        self.num_matvecs = num_matvecs
        self.num_terms = num_terms
        self.estimator = TraceEstimators.new(
            estimator,
            num_matvecs=num_matvecs,
            mode=mode,
            sampler=sampler,
        )

    @signature("[{(..., d) -> (..., d)}, (..., d)] -> (...)")
    def forward(self, fn: Fn[[Tensor], Tensor], x: Tensor) -> tuple[Tensor, Tensor]:
        r"""Return `fn(x)` together with an estimate of $\log|\det(𝕀 + 𝐃f(x))|$.

        Args:
            fn: Function $f$ whose Jacobian log-absolute-determinant should be
                estimated at $x$.
            x: Evaluation point. Its shape, dtype, and device define the domain.

        Returns:
            A pair `(y, logabsdet)` with $y = f(x)$ and the corresponding
            log-absolute-determinant estimate.
        """
        return logabsdet_series(fn, x, self.num_terms, estimator=self.estimator)
