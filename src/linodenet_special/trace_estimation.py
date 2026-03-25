r"""Trace estimators.

Notes:
    Let vₖ = Aᵏv₀,     uₖ = (Aᵀ)ᵏv₀
    then: tr(A²ᵏ) = E[uₖᵀvₖ],  tr(A²ᵏ⁺¹) = E[uₖᵀAvₖ]
"""

__all__ = [
    # samplers
    "GaussianSampler",
    "OrthSampler",
    "AbstractSampler",
    "SignSampler",
    "SphereSampler",
    # estimators
    "AbstractTraceEstimator",
    "TraceEstimator",
    "ExactEstimator",
    "HutchPlusPlusEstimator",
    "HutchinsonEstimator",
    "LogabsdetSeriesEstimator",
    "Sampler",
    "XTraceEstimator",
    # functional api
    "exact_trace_estimator",
    "hutchinson_estimator",
    "hutchplusplus_estimator",
    "logabsdet_series",
    "xtrace_estimator",
    "xtrace_estimator_matlab",
    "xtrace_naive_estimator",
]


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


class AbstractSampler(Protocol):
    def __call__(
        self,
        shape: tuple[int, ...],
        num: int,
        *,
        dtype: torch.dtype,
        device: str | torch.device,
    ) -> Tensor: ...


class AbstractTraceEstimator(Protocol):
    @signature("[{(..., d) -> (..., d)}, (..., d)] -> (...)")
    def __call__(
        self, op: Fn[[Tensor], Tensor], x: Tensor, /, *args: Any, **kwargs: Any
    ) -> Tensor:
        r"""Returns an estimate of $\tr(Df(x))$.

        Args:
            op: Function $f$ whose Jacobian trace should be estimated at $x$.
            x: Evaluation point. Its shape, dtype, and device define the domain.
            *args: partial signature, implementation may require extra arguments.
            **kwargs: extra arguments, implementation may require extra arguments.
        """
        ...


class Sampler(StrEnum):
    r"""Built-in probe vector samplers for stochastic trace estimators."""

    GAUSSIAN = "gaussian"
    SIGN = "sign"
    SPHERE = "sphere"
    ORTH = "orth"

    @classmethod
    def new(cls, sampler: str | AbstractSampler, /) -> AbstractSampler:
        r"""Construct a built-in sampler or forward a custom sampler as-is."""
        if callable(sampler):
            return sampler
        match cls(sampler):
            case cls.GAUSSIAN:
                return GaussianSampler()
            case cls.SIGN:
                return SignSampler()
            case cls.SPHERE:
                return SphereSampler()
            case cls.ORTH:
                return OrthSampler()


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
        r"""Sample $vᵢ∼𝓝(0,𝕀ₙ)."""
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
    r"""Sample orthogonal with norm $n$."""

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


@signature("[{(..., d) -> (..., d)}, (..., d)] -> (...)")
def exact_trace_estimator(op: Fn[[Tensor], Tensor], x: Tensor, /) -> Tensor:
    r"""Estimate $\tr(Df(x))$ by explicitly materializing the Jacobian."""
    if x.ndim == 0:
        raise ValueError("x must be at least one-dimensional")

    _, jvp_fn = linearize(op, x)
    dim = x.shape[-1]
    identity = torch.eye(dim, device=x.device, dtype=x.dtype).expand(
        *x.shape[:-1], dim, dim
    )
    matrix = vmap(jvp_fn, -1, -1)(identity)
    trace = torch.einsum("...ii -> ...", matrix)
    return trace.real if not matrix.is_complex() else trace


@signature("[{(..., d) -> (..., d)}, (..., d)] -> (...)")
def hutchinson_estimator(
    op: Fn[[Tensor], Tensor],
    x: Tensor,
    /,
    num_matvecs: int,
    *,
    sampler: str | AbstractSampler = "sphere",
) -> Tensor:
    r"""Estimate $\tr(Df(x))$ with Hutchinson's estimator.

    .. math:: \tr(A) = E[uᵀAv], where E[uvᵀ] = 𝕀

    Args:
        op: Function $f$ whose Jacobian trace should be estimated at $x$.
        x: Evaluation point. Its shape, dtype, and device define the domain.
        num_matvecs: number of matrix-vector products to use for the estimator.
            This is equivalent to the number of probe vectors.
        sampler: Probe sampler, either a built-in sampler name or a custom callable.

    Returns:
        A Hutchinson estimate of $\tr(Df(x))$.
    """
    if x.ndim == 0:
        raise ValueError("x must be at least one-dimensional")
    if num_matvecs < 1:
        raise ValueError("num_samples must be at least 1")

    sampler = Sampler.new(sampler)
    probes = sampler(x.shape, num_matvecs, dtype=x.dtype, device=x.device)
    _, jvp_fn = linearize(op, x)
    batched_jvp_fn = vmap(jvp_fn, -1, -1)
    estimate = vecdot(probes, batched_jvp_fn(probes), dim=-2).mean(dim=-1)
    return estimate


@signature("[{(..., d) -> (..., d)}, (..., d)] -> (...)")
def hutchplusplus_estimator(
    op: Fn[[Tensor], Tensor],
    x: Tensor,
    /,
    num_matvecs: int,
    *,
    sampler: str | AbstractSampler = "sphere",
) -> Tensor:
    r"""Estimate $\tr(Df(x))$ with Hutch++.

    .. math:: \tr(A) = \tr(QᵀAQ) + 𝐄[vᵀ(𝕀-QQᵀ)A(𝕀-QQᵀ)v]

    Args:
        op: Function $f$ whose Jacobian trace should be estimated at $x$.
        x: Evaluation point. Its shape, dtype, and device define the domain.
        num_matvecs: Total matrix-vector product budget.
        sampler: Probe sampler, either a built-in sampler name or a custom callable.
    """
    if x.ndim == 0:
        raise ValueError("x must be at least one-dimensional")
    if num_matvecs < 3:
        raise ValueError("num_matvecs must be at least 3")

    num_samples = num_matvecs // 3
    sampler = Sampler.new(sampler)
    samples = sampler(x.shape, num_samples, device=x.device, dtype=x.dtype)
    residual_samples = sampler(x.shape, num_samples, device=x.device, dtype=x.dtype)

    _, jvp_fn = linearize(op, x)
    batched_jvp_fn = vmap(jvp_fn, -1, -1)  # (...dn) -> (...dn)
    sketch = batched_jvp_fn(samples)
    q, _ = qr(sketch, mode="reduced")  # (...dr)

    residual_samples = residual_samples - q @ (q.mH @ residual_samples)
    low_rank = vecdot(q, batched_jvp_fn(q), dim=-2).sum(dim=-1)
    residual = vecdot(residual_samples, batched_jvp_fn(residual_samples), dim=-2).mean(
        dim=-1
    )
    estimate = low_rank + residual
    return estimate.real if not estimate.is_complex() else estimate


@signature("[{(..., d) -> (..., d)}, (..., d)] -> (...)")
def xtrace_naive_estimator(
    op: Fn[[Tensor], Tensor],
    x: Tensor,
    /,
    num_matvecs: int,
    *,
    sampler: str | AbstractSampler = "sphere",
    renormalize: bool = False,
) -> Tensor:
    r"""Naive XTrace estimate of $\tr(Df(x))$ for debugging."""
    if num_matvecs < 2:
        raise ValueError("num_matvecs must be at least 2")

    num_samples = num_matvecs // 2
    *batch, N = x.shape
    k = min(N, num_samples)

    _, jvp_fn = linearize(op, x)
    batched_op = vmap(jvp_fn, -1, -1)  # (...Nm) -> (...Nm)
    tr = torch.zeros(batch, dtype=x.dtype, device=x.device)

    sampler = Sampler.new(sampler)
    samples = sampler(x.shape, k, device=x.device, dtype=x.dtype)
    Y = batched_op(samples)  # (...Nm)

    mus = []
    for i in range(num_samples):
        col_indices = torch.arange(num_samples, device=Y.device)
        Q_i, _ = qr(Y[..., i != col_indices], mode="reduced")
        ω_i = samples[..., [i]]
        μ_i = ω_i - Q_i @ (Q_i.mH @ ω_i)
        mus.append(μ_i)
        tr = tr + vecdot(Q_i, batched_op(Q_i), dim=-2).sum(dim=-1)
    μ = torch.cat(mus, dim=-1)
    scale = 1.0 - renormalize * (1.0 - (N - k + 1) / vecdot(μ, μ, dim=-2, keepdim=True))
    μ = μ * scale
    residual = vecdot(μ, batched_op(μ), dim=-2).mean(dim=-1)
    return tr / k + residual


@signature("[{(..., d) -> (..., d)}, (..., d)] -> (...)")
def xtrace_estimator(
    op: Fn[[Tensor], Tensor],
    x: Tensor,
    /,
    num_matvecs: int,
    *,
    sampler: str | AbstractSampler = "sphere",
    renormalize: bool = False,
) -> Tensor:
    r"""Estimate $\tr(Df(x))$ with the fast XTrace estimator."""
    if x.ndim == 0:
        raise ValueError("x must be at least one-dimensional")
    if num_matvecs < 2:
        raise ValueError("num_matvecs must be at least 2")

    *_, N = x.shape
    num_samples = num_matvecs // 2
    k = min(N, num_samples)

    sampler = Sampler.new(sampler)
    samples = sampler(x.shape, k, device=x.device, dtype=x.dtype)
    _, jvp_fn = linearize(op, x)
    batched_op = vmap(jvp_fn, -1, -1)  # (...dk) -> (...dk)

    Y = batched_op(samples)  # (...dk)
    Q, R = qr(Y, mode="reduced")  # (...dk), (...kk)
    Z = batched_op(Q)  # (...dk)
    H = Q.mH @ Z  # (...kk)
    W = Q.mH @ samples  # (...kk)
    T = Z.mH @ samples  # (...kk)

    identity = torch.eye(k, dtype=samples.dtype, device=samples.device)
    S = solve_triangular(R.mH, identity, upper=False, left=True)  # (...kk)
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
    term1 = sw.abs().square() * shs
    term2 = sw.conj() * vecdot(S, R - hw, dim=-2)
    x_term = W - sw.unsqueeze(-2) * S
    term3 = -vecdot(T - H.mH @ W, x_term, dim=-2)
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
    sampler: str | AbstractSampler = "sphere",
    renormalize: bool = False,
) -> Tensor:
    r"""Estimate the trace of a matrix using the original XTrace MATLAB algorithm.

    This is a direct Torch transcription of the reference MATLAB code.
    """
    if x.ndim == 0:
        raise ValueError("x must be at least one-dimensional")
    if num_matvecs < 2:
        raise ValueError("num_matvecs must be at least 2")

    sampler = Sampler.new(sampler)
    samples = sampler(x.shape, num_matvecs // 2, dtype=x.dtype, device=x.device)
    _, jvp_fn = linearize(op, x)
    fn = vmap(jvp_fn, -1, -1)

    *_, m, d = samples.shape

    # MATLAB: Om = sqrt(N) * cnormc(randn(N, m))
    # Here we reuse the provided probes as the m columns of Ω.
    # Omega: (..., d, m)
    omega = d**0.5 * samples.mH / vector_norm(samples.mH, dim=-2, keepdim=True)

    # MATLAB: Y = A * Om
    # Y: (..., d, m)
    y = fn(omega.mH).mH
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
    z = fn(q.mH).mH
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
    quantities of its Jacobian $Df(x)$. Linear operators fit this API as a special case:
    when $f(z) = Az$, the Jacobian is constant and equal to $A$.

    Subclasses must implement `estimate`, which returns an estimate of $\tr(Df(x))$.
    The default `estimate_powers` implementation builds on top of `estimate`; concrete
    estimators may override it with more efficient algorithms.
    """

    @signature("[{(..., d) -> (..., d)}, (..., d)] -> (...)")
    @abstractmethod
    def forward(
        self,
        op: Fn[[Tensor], Tensor],
        x: Tensor,
        /,
    ) -> Tensor:
        r"""Return an estimate of $\tr(Df(x))$.

        Args:
            op: Function $f$ whose Jacobian trace should be estimated at $x$.
            x: Evaluation point. Its shape, dtype, and device define the domain.
        """
        raise NotImplementedError

    @signature("[{(..., d) -> (..., d)}, (..., d)] -> (...)")
    def estimate_powers(
        self,
        op: Fn[[Tensor], Tensor],
        x: Tensor,
        /,
        max_power: int,
    ) -> Iterator[Tensor]:
        r"""Yield estimates of $\tr(Df(x)ᵏ)$ for $k = 1, …, \text{max_power}$.

        The default implementation repeatedly composes $f$ with itself and delegates to
        `estimate`. This is mainly a compatibility fallback; specialized estimators can
        usually implement this more efficiently and more accurately.
        """
        if max_power < 1:
            raise ValueError("max_power must be at least 1")

        power_op = op
        for _ in range(max_power):
            yield self(power_op, x)
            previous_op = power_op
            power_op = lambda z, prev=previous_op: op(prev(z))


class ExactEstimator(TraceEstimator):
    r"""Estimate traces by explicitly materializing the operator matrix.

    Cost: N³
        N is the dimension of the operator

    Args:
        mode: Whether to materialize the Jacobian from forward Jacobian-vector products
            or adjoint vector-Jacobian products.
    """

    mode: Final[str]

    def __init__(self, mode: str = "forward") -> None:
        super().__init__()
        if mode not in {"forward", "adjoint"}:
            raise ValueError(f"mode must be 'forward' or 'adjoint', got {mode!r}")
        self.mode = mode

    def _materialize(
        self,
        op: Fn[[Tensor], Tensor],
        x: Tensor,
        /,
    ) -> Tensor:
        if x.ndim == 0:
            raise ValueError("x must be at least one-dimensional")

        dim = x.shape[-1]
        identity = torch.eye(dim, device=x.device, dtype=x.dtype).expand(
            *x.shape[:-1], dim, dim
        )

        match self.mode:
            case "forward":
                _, jvp_fn = linearize(op, x)
                batched_op = vmap(jvp_fn, -1, -1)  # (...dn) -> (...dn)
                return batched_op(identity)

            case "adjoint":
                _, vjp_fn, *_ = vjp(op, x)
                batched_adj = vmap(vjp_fn, -1, -1)  # (...dn) -> tuple[(...dn)]
                (matrix,) = batched_adj(identity)
                return matrix

            case _:
                raise AssertionError("unreachable")

    @signature("[{(..., d) -> (..., d)}, (..., d)] -> (...)")
    def forward(
        self,
        op: Fn[[Tensor], Tensor],
        x: Tensor,
        /,
    ) -> Tensor:
        matrix = self._materialize(op, x)
        return torch.einsum("...ii -> ...", matrix)

    @signature("[{(..., d) -> (..., d)}, (..., d)] -> (...)")
    def estimate_powers(
        self,
        op: Fn[[Tensor], Tensor],
        x: Tensor,
        /,
        max_power: int,
    ) -> Iterator[Tensor]:
        if max_power < 1:
            raise ValueError("max_power must be at least 1")

        matrix = self._materialize(op, x)
        eigenvalues = torch.linalg.eigvals(matrix)
        for power in range(1, max_power + 1):
            trace_power = eigenvalues.pow(power).sum(dim=-1)
            yield trace_power.real if not matrix.is_complex() else trace_power

    @signature("[{(..., d) -> (..., d)}, (..., d)] -> (...)")
    def estimate_logabsdet(
        self,
        op: Fn[[Tensor], Tensor],
        x: Tensor,
        /,
    ) -> Tensor:
        matrix = self._materialize(op, x)
        eigenvalues = torch.linalg.eigvals(matrix)
        logabsdet = torch.log(torch.abs(1 + eigenvalues)).sum(dim=-1)
        return logabsdet.real if not matrix.is_complex() else logabsdet


class HutchinsonEstimator(TraceEstimator):
    r"""Estimate traces with Hutchinson's estimator.

    Cost: mN² + O(m²N + m³)
        m is the number of matvecs (=`num_samples`),
        N is the dimension of the operator

    Args:
        num_samples: Number of probe vectors.
        num_matvecs: Alias for `num_samples`.
        sampler: Probe vector sampler.
        mode: Whether to use forward Jacobian-vector products, adjoint vector-Jacobian
            products, or a symmetric alternating scheme.
    """

    num_matvecs: Final[int]
    num_samples: Final[int]
    mode: Final[str]
    sampler: AbstractSampler

    @overload
    def __init__(
        self,
        num_samples: int,
        *,
        sampler: str | AbstractSampler = "sphere",
        mode: str = "symmetric",
    ) -> None: ...
    @overload
    def __init__(
        self,
        *,
        num_matvecs: int,
        sampler: str | AbstractSampler = "sphere",
        mode: str = "symmetric",
    ) -> None: ...
    def __init__(
        self,
        num_samples: int | None = None,
        *,
        num_matvecs: int | None = None,
        sampler: str | AbstractSampler = Sampler.SPHERE,
        mode: str = "symmetric",
    ) -> None:
        super().__init__()

        match num_samples, num_matvecs:
            case None, None:
                raise ValueError("either num_samples or num_matvecs must be provided")
            case n, None:
                self.num_matvecs = n
                self.num_samples = n
            case None, n:
                self.num_matvecs = n
                self.num_samples = n
            case _, _:
                raise ValueError(
                    "Only one of num_samples or num_matvecs should be provided, but got both."
                )

        if mode not in {"forward", "adjoint", "symmetric"}:
            raise ValueError(
                f"mode must be 'forward', 'adjoint', or 'symmetric', got {mode!r}"
            )

        self.mode = mode
        self.sampler = Sampler.new(sampler)

    @signature("[{(..., d) -> (..., d)}, (..., d)] -> (...)")
    def forward(
        self,
        op: Fn[[Tensor], Tensor],
        x: Tensor,
        /,
    ) -> Tensor:
        r"""Return an estimate of $\tr(Df(x))$."""
        return next(self.estimate_powers(op, x, 1))

    @signature("[{(..., d) -> (..., d)}, (..., d)] -> (...)")
    def estimate_powers(
        self,
        op: Fn[[Tensor], Tensor],
        x: Tensor,
        /,
        max_power: int,
    ) -> Iterator[Tensor]:
        r"""Yield estimates of $\tr(Df(x)ᵏ)$ for $k = 1, …, \text{max_power}$."""
        if max_power < 1:
            raise ValueError("max_power must be at least 1")
        if x.ndim == 0:
            raise ValueError("x must be at least one-dimensional")

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
                _, jvp_fn = linearize(op, x)  # (...d) -> (...d)
                batched_jvp_fn = vmap(jvp_fn, -1, -1)  # (...dn) -> (...dn)

                for _ in range(max_power):
                    right_samples = batched_jvp_fn(right_samples)
                    yield vecdot(left_samples, right_samples, dim=-2).mean(dim=-1)

            case "adjoint":
                # use x ↦ Aᵀx only
                _, vjp_fn, *_ = vjp(op, x)  # (...d) -> tuple[(...d)]
                batched_vjp_fn = vmap(vjp_fn, -1, -1)  # (...dn) -> tuple[(...dn)]

                for _ in range(max_power):
                    (left_samples,) = batched_vjp_fn(left_samples)
                    yield vecdot(left_samples, right_samples, dim=-2).mean(dim=-1)

            case "symmetric":
                # alternate between Ax and Aᵀx, which is good for forward sensitivity.
                # as it grows exponentially in the number of matvecs.
                _, jvp_fn = linearize(op, x)  # (...d) -> (...d)
                _, vjp_fn, *_ = vjp(op, x)  # (...d) -> tuple[(...d)]
                batched_jvp_fn = vmap(jvp_fn, -1, -1)  # (...dn) -> (...dn)
                batched_vjp_fn = vmap(vjp_fn, -1, -1)  # (...dn) -> tuple[(...dn)]

                power = 0
                while power < max_power:
                    right_samples = batched_jvp_fn(right_samples)
                    power += 1
                    yield vecdot(left_samples, right_samples, dim=-2).mean(dim=-1)

                    if power == max_power:
                        break

                    (left_samples,) = batched_vjp_fn(left_samples)
                    power += 1
                    yield vecdot(left_samples, right_samples, dim=-2).mean(dim=-1)
            case _:
                raise ValueError(f"invalid mode {self.mode!r}")

    @signature("[{(..., d) -> (..., d)}, (..., d)] -> (...)")
    def estimate_logabsdet(
        self,
        op: Fn[[Tensor], Tensor],
        x: Tensor,
        /,
        num_series_terms: int,
    ) -> Tensor:
        if x.ndim == 0:
            raise ValueError("x must be at least one-dimensional")
        if self.mode != "forward":
            raise NotImplementedError(
                f"XTraceEstimator only supports mode='forward', got {self.mode!r}"
            )
        if num_series_terms < 1:
            raise ValueError("num_series_terms must be at least 1")

        _, jvp_fn = linearize(op, x)  # (...d) -> (...d)
        batched_jvp_fn = vmap(jvp_fn, -1, -1)  # (...dn) -> (...dn)
        samples = self.sampler(
            x.shape,
            self.num_samples,
            device=x.device,
            dtype=x.dtype,
        )
        result = torch.zeros(x.shape[:-1], device=x.device, dtype=x.dtype)
        sign = torch.tensor(-1.0, device=x.device, dtype=x.dtype)
        for k in range(1, num_series_terms + 1):
            sign = sign.neg()
            samples = batched_jvp_fn(samples)
            trace_power = xtrace_estimator(batched_jvp_fn, samples)
            result = result + (sign / k) * trace_power
        return result.real if not result.is_complex() else result


class HutchPlusPlusEstimator(TraceEstimator):
    r"""Estimate traces with the Hutch++ variance-reduced estimator.

    Cost: mN² + O(m²N + m³)
        m is the number of matvecs (=3×`num_samples`),
        N is the dimension of the operator

    Args:
        num_samples: Number of probe vectors.
        num_matvecs: Alias for the total matvec budget.
        sampler: Probe vector sampler.
        mode: Whether to use forward Jacobian-vector products, adjoint vector-Jacobian
            products, or a symmetric alternating scheme.
    """

    num_matvecs: Final[int]
    num_samples: Final[int]
    mode: Final[str]
    sampler: AbstractSampler

    @overload
    def __init__(
        self,
        num_samples: int,
        *,
        sampler: str | AbstractSampler = "sphere",
        mode: str = "symmetric",
    ) -> None: ...
    @overload
    def __init__(
        self,
        *,
        num_matvecs: int,
        sampler: str | AbstractSampler = "sphere",
        mode: str = "symmetric",
    ) -> None: ...
    def __init__(
        self,
        num_samples: int | None = None,
        *,
        num_matvecs: int | None = None,
        sampler: str | AbstractSampler = Sampler.SPHERE,
        mode: str = "symmetric",
    ) -> None:
        super().__init__()

        match num_samples, num_matvecs:
            case None, None:
                raise ValueError("either num_samples or num_matvecs must be provided")
            case n, None:
                self.num_matvecs = 3 * n
                self.num_samples = n
            case None, n:
                if num_matvecs < 3:
                    raise ValueError("num_matvecs must be at least 3")
                self.num_matvecs = n
                self.num_samples = n // 3
            case _, _:
                raise ValueError(
                    "Only one of num_samples or num_matvecs should be provided, but got both."
                )

        if mode not in {"forward", "adjoint", "symmetric"}:
            raise ValueError(
                f"mode must be 'forward', 'adjoint', or 'symmetric', got {mode!r}"
            )

        self.mode = mode
        self.sampler = Sampler.new(sampler)

    @signature("[{(..., d) -> (..., d)}, (..., d)] -> (...)")
    def forward(
        self,
        op: Fn[[Tensor], Tensor],
        x: Tensor,
        /,
    ) -> Tensor:
        r"""Return an estimate of $\tr(Df(x))$."""
        return next(self.estimate_powers(op, x, 1))

    @signature("[{(..., d) -> (..., d)}, (..., d)] -> (...)")
    def estimate_powers(
        self,
        op: Fn[[Tensor], Tensor],
        x: Tensor,
        /,
        max_power: int,
    ) -> Iterator[Tensor]:
        r"""Yield estimates of $\tr(Df(x)ᵏ)$ for $k = 1, …, \text{max_power}$."""
        if max_power < 1:
            raise ValueError("max_power must be at least 1")
        if x.ndim == 0:
            raise ValueError("x must be at least one-dimensional")

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
                _, jvp_fn = linearize(op, x)
                batched_jvp_fn = vmap(jvp_fn, -1, -1)  # (...dn) -> (...dn)
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

            case "adjoint":
                _, vjp_fn, *_ = vjp(op, x)
                batched_vjp_fn = vmap(vjp_fn, -1, -1)  # (...dn) -> tuple[(...dn)]
                (sketch,) = batched_vjp_fn(samples)
                Q, _ = qr(sketch, mode="reduced")  # (...dr)

                projected_samples = Q
                projected_l = projected_samples.clone()
                projected_r = projected_samples.clone()
                residual_samples = residual_samples - Q @ (Q.mH @ residual_samples)
                residual_l = residual_samples.clone()
                residual_r = residual_samples.clone()

                for _ in range(max_power):
                    (projected_l,) = batched_vjp_fn(projected_l)
                    (residual_l,) = batched_vjp_fn(residual_l)

                    low_rank = vecdot(projected_l, projected_r, dim=-2).sum(dim=-1)
                    residual = vecdot(residual_l, residual_r, dim=-2).mean(dim=-1)
                    yield low_rank + residual

            case "symmetric":
                _, jvp_fn = linearize(op, x)
                _, vjp_fn, *_ = vjp(op, x)
                batched_jvp_fn = vmap(jvp_fn, -1, -1)  # (...dn) -> (...dn)
                batched_vjp_fn = vmap(vjp_fn, -1, -1)  # (...dn) -> tuple[(...dn)]

                # Hutch++ uses a fixed projector P = QQᵀ and the exact split
                #   tr(Aᵏ) = tr(Qᵀ Aᵏ Q) + tr((I-P) Aᵏ (I-P)).
                left_sketch, right_sketch = torch.tensor_split(samples, 2, dim=-1)
                (left_sketch,) = batched_vjp_fn(left_sketch)
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

                    (projected_l,) = batched_vjp_fn(projected_l)
                    (residual_l,) = batched_vjp_fn(residual_l)
                    power += 1

                    low_rank = vecdot(projected_l, projected_r, dim=-2).sum(dim=-1)
                    residual = vecdot(residual_l, residual_r, dim=-2).mean(dim=-1)
                    yield low_rank + residual

            case _:
                raise ValueError(f"invalid mode {self.mode!r}")

    @signature("[{(..., d) -> (..., d)}, (..., d)] -> (...)")
    def estimate_logabsdet(
        self,
        op: Fn[[Tensor], Tensor],
        x: Tensor,
        /,
        num_series_terms: int,
    ) -> Tensor:
        if x.ndim == 0:
            raise ValueError("x must be at least one-dimensional")
        if self.mode != "forward":
            raise NotImplementedError(
                f"XTraceEstimator only supports mode='forward', got {self.mode!r}"
            )
        if num_series_terms < 1:
            raise ValueError("num_series_terms must be at least 1")

        _, jvp_fn = linearize(op, x)
        batched_jvp_fn = vmap(jvp_fn, -1, -1)  # (...dn) -> (...dn)
        samples = self.sampler(
            x.shape,
            self.num_samples,
            device=x.device,
            dtype=x.dtype,
        )
        result = torch.zeros(x.shape[:-1], device=x.device, dtype=x.dtype)
        sign = torch.tensor(-1.0, device=x.device, dtype=x.dtype)
        for k in range(1, num_series_terms + 1):
            sign = sign.neg()
            samples = batched_jvp_fn(samples)
            trace_power = xtrace_estimator(batched_jvp_fn, samples)
            result = result + (sign / k) * trace_power
        return result.real if not result.is_complex() else result


class XTraceEstimator(TraceEstimator):
    r"""Estimate traces with the XTrace estimator.

    Cost: mN^2 + O(m^3)
        m is the number of matvecs (=2x`num_samples`),
        N is the dimension of the operator

    Args:
        num_samples: Number of probe vectors.
        num_matvecs: Alias for the total matvec budget.
        sampler: Probe vector sampler.
        renormalize: Whether to apply the paper's renormalization.
        mode: Jacobian action mode. Only `"forward"` is currently implemented.
    """

    num_matvecs: Final[int]
    num_samples: Final[int]
    renormalize: Final[bool]
    mode: Final[str]
    r"""Whether to apply renormalization from paper section 2.3"""

    sampler: AbstractSampler

    @overload
    def __init__(
        self,
        num_samples: int,
        *,
        sampler: str | AbstractSampler = ...,
        renormalize: bool = ...,
        mode: str = ...,
    ) -> None: ...
    @overload
    def __init__(
        self,
        *,
        num_matvecs: int,
        sampler: str | AbstractSampler = ...,
        renormalize: bool = ...,
        mode: str = ...,
    ) -> None: ...
    def __init__(
        self,
        num_samples: int | None = None,
        *,
        num_matvecs: int | None = None,
        sampler: str | AbstractSampler = Sampler.SPHERE,
        renormalize: bool = True,
        mode: str = "forward",
    ) -> None:
        super().__init__()

        match num_samples, num_matvecs:
            case None, None:
                raise ValueError("either num_samples or num_matvecs must be provided")
            case n, None:
                self.num_matvecs = 2 * n
                self.num_samples = n
            case None, n:
                if n < 2:
                    raise ValueError("num_matvecs must be at least 2")
                self.num_matvecs = n
                self.num_samples = n // 2
            case _, _:
                raise ValueError(
                    "Only one of num_samples or num_matvecs should be provided, but got both."
                )

        if mode not in {"forward", "adjoint", "symmetric"}:
            raise ValueError(
                f"mode must be 'forward', 'adjoint', or 'symmetric', got {mode!r}"
            )

        self.renormalize = bool(renormalize)
        self.mode = mode
        self.sampler = Sampler.new(sampler)

    @signature("[{(..., d) -> (..., d)}, (..., d)] -> (...)")
    def forward(
        self,
        op: Fn[[Tensor], Tensor],
        x: Tensor,
        /,
    ) -> Tensor:
        r"""Return an estimate of $\tr(Df(x))$."""
        return next(self.estimate_powers(op, x, 1))

    @signature("[{(..., d) -> (..., d)}, (..., d)] -> (...)")
    def estimate_naive(
        self,
        op: Fn[[Tensor], Tensor],
        x: Tensor,
        /,
    ) -> Tensor:
        r"""Use the naive implementation to estimate $\tr(Df(x))$."""
        if x.ndim == 0:
            raise ValueError("x must be at least one-dimensional")
        if self.mode != "forward":
            raise NotImplementedError(
                f"XTraceEstimator only supports mode='forward', got {self.mode!r}"
            )

        *batch, N = x.shape
        k = min(N, self.num_samples)
        samples = self.sampler(x.shape, k, device=x.device, dtype=x.dtype)
        tr = torch.zeros(batch, dtype=x.dtype, device=x.device)
        _, jvp_fn = linearize(op, x)
        batched_op = vmap(jvp_fn, -1, -1)  # (...Nm) -> (...Nm)
        Y = batched_op(samples)  # (...Nm)

        mus = []
        for i in range(self.num_samples):
            col_indices = torch.arange(self.num_samples, device=Y.device)
            Q_i, _ = qr(Y[..., i != col_indices], mode="reduced")
            ω_i = samples[..., [i]]
            μ_i = ω_i - Q_i @ (Q_i.mH @ ω_i)
            mus.append(μ_i)
            tr = tr + vecdot(Q_i, batched_op(Q_i), dim=-2).sum(dim=-1)
        μ = torch.cat(mus, dim=-1)
        scale = 1.0 - (1.0 - (N - k + 1) / vecdot(μ, μ, dim=-2)) * self.renormalize
        μ = μ * scale.unsqueeze(-2)
        residual = vecdot(μ, batched_op(μ), dim=-2).mean(dim=-1)
        return tr / k + residual

    @signature("[{(..., d) -> (..., d)}, (..., d)] -> (...)")
    def estimate_powers(
        self,
        op: Fn[[Tensor], Tensor],
        x: Tensor,
        /,
        max_power: int,
    ) -> Iterator[Tensor]:
        if max_power < 1:
            raise ValueError("max_power must be at least 1")
        if max_power > 1:
            raise NotImplementedError("XTraceEstimator currently only supports k=1")
        if x.ndim == 0:
            raise ValueError("x must be at least one-dimensional")
        if self.mode != "forward":
            raise NotImplementedError(
                f"XTraceEstimator only supports mode='forward', got {self.mode!r}"
            )

        *batch, N = x.shape
        k = min(N, self.num_samples)
        samples = self.sampler(x.shape, k, device=x.device, dtype=x.dtype)
        _, jvp_fn = linearize(op, x)
        batched_op = vmap(jvp_fn, -1, -1)  # (...Nm) -> (...Nm)
        Y = batched_op(samples)  # (...Nm)
        Q, R = qr(Y, mode="reduced")  # (...Nk), (...kk)
        # Q has normalized cols <-> Q.norm(dim=-2) = 1

        Z = batched_op(Q)  # (...Nk)
        H = Q.mH @ Z  # (...kk)
        W = Q.mH @ samples  # (...kk)
        T = Z.mH @ samples  # (...kk)

        # solve R^* S = Iₖ
        I = torch.eye(k, dtype=samples.dtype, device=samples.device)
        S = solve_triangular(R.mH, I, upper=False, left=True)  # lower triangular
        # normalize COLS
        S = S / vector_norm(S, dim=-2, keepdim=True)  # (...kk)

        SW = vecdot(S, W, dim=-2)  # (...i)
        SR = vecdot(S, R, dim=-2)  # (...i)
        X = W - SW.unsqueeze(-2) * S  # (...kk)
        TX = vecdot(T, X, dim=-2)  # (...i)
        XHX = torch.einsum("...ki, ...kl, ...li -> ...i", X.conj(), H, X)
        SHS = torch.einsum("...ki, ...kl, ...li -> ...i", S.conj(), H, S)

        if False:
            mus = []
            for i in range(self.num_samples):
                col_indices = torch.arange(self.num_samples, device=Y.device)
                Q_i, R = qr(Y[..., i != col_indices], mode="reduced")
                ω_i = samples[..., [i]]
                μ_i = ω_i - Q_i @ (Q_i.mH @ ω_i)
                mus.append(μ_i)
            μ = torch.cat(mus, dim=-1)
            mu_norm_sq_a = vecdot(μ, μ, dim=-2)  # column norm
            mu_norm_sq_b = (
                vecdot(samples, samples, dim=-2)
                - vecdot(W, W, dim=-2)
                + SW.abs().square()
            )

        if self.renormalize:
            scale = (N - k + 1) / (  #
                vecdot(samples, samples, dim=-2)
                - vecdot(W, W, dim=-2)
                + SW.abs().square()
            )
        else:
            scale = 1.0

        WS = SW.conj()  # (...i)
        trs = -SHS + scale * (XHX + WS * SR - TX)

        HW = H @ W
        term1 = SW.abs().square() * SHS  # |⟨sᵢ∣wᵢ⟩|²⟨sᵢ∣Hsᵢ⟩
        term2 = SW.conj() * vecdot(S, R - HW, dim=-2)  # ⟨wᵢ∣sᵢ⟩⟨sᵢ∣rᵢ - Hwᵢ⟩
        term3 = -vecdot(T - H.mH @ W, X, dim=-2)  # -⟨tᵢ - Hᴴwᵢ∣wᵢ - ⟨sᵢ∣wᵢ⟩sᵢ⟩
        trs = -SHS + scale * (term1 + term2 + term3)

        estimate = H.diagonal(dim1=-2, dim2=-1).sum(dim=-1) + trs.mean(dim=-1)
        yield estimate

    @signature("[{(..., d) -> (..., d)}, (..., d)] -> (...)")
    def estimate_logabsdet(
        self,
        op: Fn[[Tensor], Tensor],
        x: Tensor,
        /,
        num_series_terms: int,
    ) -> Tensor:
        if x.ndim == 0:
            raise ValueError("x must be at least one-dimensional")
        if self.mode != "forward":
            raise NotImplementedError(
                f"XTraceEstimator only supports mode='forward', got {self.mode!r}"
            )
        if num_series_terms < 1:
            raise ValueError("num_series_terms must be at least 1")

        _, jvp_fn = linearize(op, x)
        batched_jvp_fn = vmap(jvp_fn, -1, -1)  # (...dn) -> (...dn)
        samples = self.sampler(
            x.shape,
            self.num_samples,
            device=x.device,
            dtype=x.dtype,
        )
        result = torch.zeros(x.shape[:-1], device=x.device, dtype=x.dtype)
        sign = torch.tensor(-1.0, device=x.device, dtype=x.dtype)
        for k in range(1, num_series_terms + 1):
            sign = sign.neg()
            samples = batched_jvp_fn(samples)
            trace_power = xtrace_estimator(batched_jvp_fn, samples)
            result = result + (sign / k) * trace_power
        return result.real if not result.is_complex() else result


@signature("[{(..., d) -> (..., d)}, (..., d)] -> (...)")
def logabsdet_series(
    estimator: TraceEstimator,
    op: Fn[[Tensor], Tensor],
    x: Tensor,
    /,
    num_series_terms: int,
) -> Tensor:
    r"""Estimate $\log |\det(𝕀 + Df(x))|$ via a truncated power series.

    The helper uses

    .. math::  \log|\det(𝕀 + A)| = ∑ₖ(-1)ᵏ⁺¹/k\tr(Aᵏ)

    truncated after `num_series_terms` terms and replaces each trace power with the
    corresponding value from `estimator.estimate_powers`.
    """
    if num_series_terms < 1:
        raise ValueError("num_series_terms must be at least 1")

    trace_powers = estimator.estimate_powers(op, x, num_series_terms)
    first_power = next(trace_powers)
    result = first_power.clone()
    sign = -1.0
    for k, trace_power in enumerate(trace_powers, start=2):
        result = result + (sign / k) * trace_power
        sign = -sign
    return result.real if not result.is_complex() else result


class LogabsdetSeriesEstimator(nn.Module):
    r"""Estimate $\log|\det(𝕀 + Df(x))|$ with a trace-estimator backend.

    - \log|\det A| = \Re(\tr(\log A)) for any A in the image of the matrix exponential
    - \log|\det A| = ½\tr(\log AᴴA) for any A. (-∞ if A is singular)

    Args:
        estimator: Trace-estimator backend, or a string in {"exact", "hutch", "xtrace"}
            used to construct one.
        num_samples: Number of probe vectors for stochastic estimators when `estimator`
            is given as a string.
        num_series_terms: Number of power-series terms for stochastic estimators.

    Returns:
        y: $f(x)$
        logabsdet: Approximation of $\log|\det(𝕀 + Df(x))|$
    """

    estimator: ExactEstimator | AbstractTraceEstimator
    num_samples: int | None
    num_series_terms: int | None

    def __init__(
        self,
        estimator: str | AbstractTraceEstimator,
        num_samples: int | None,
        num_series_terms: int | None,
    ) -> None:
        super().__init__()

        if num_samples is not None and num_samples < 1:
            raise ValueError("num_samples must be at least 1 when provided")
        if num_series_terms is not None and num_series_terms < 1:
            raise ValueError("num_series_terms must be at least 1 when provided")

        self.num_samples = num_samples
        self.num_series_terms = num_series_terms

        match estimator:
            case "exact":
                self.estimator = ExactEstimator()
            case "hutch" | "hutchinson":
                if num_samples is None:
                    raise ValueError("num_samples is required for estimator='hutch'")
                if num_series_terms is None:
                    raise ValueError(
                        "num_series_terms is required for estimator='hutch'"
                    )
                self.estimator = HutchinsonEstimator(num_samples=num_samples)
            case "hutch++" | "hutchplusplus":
                if num_samples is None:
                    raise ValueError("num_samples is required for estimator='hutch++'")
                if num_series_terms is None:
                    raise ValueError(
                        "num_series_terms is required for estimator='hutch++'"
                    )
                self.estimator = HutchPlusPlusEstimator(num_samples=num_samples)
            case "xtrace":
                if num_samples is None:
                    raise ValueError("num_samples is required for estimator='xtrace'")
                if num_series_terms is None:
                    raise ValueError(
                        "num_series_terms is required for estimator='xtrace'"
                    )
                self.estimator = XTraceEstimator(num_samples=num_samples)
            case _ if hasattr(estimator, "estimate") and hasattr(
                estimator, "estimate_powers"
            ):
                self.estimator = estimator
            case _:
                raise TypeError(f"Unknown logabsdet estimator {estimator!r}")

    @signature("[{(..., d) -> (..., d)}, (..., d)] -> (...)")
    def forward(self, fn: Fn[[Tensor], Tensor], x: Tensor) -> tuple[Tensor, Tensor]:
        y = fn(x)
        match self.estimator:
            case ExactEstimator():
                return y, self.estimator.estimate_logabsdet(fn, x)
            case XTraceEstimator():
                if self.num_series_terms is None:
                    raise ValueError(
                        "num_series_terms is required for stochastic logabsdet estimation"
                    )
                return y, self.estimator.estimate_logabsdet(
                    fn, x, self.num_series_terms
                )
            case _:
                if self.num_series_terms is None:
                    raise ValueError(
                        "num_series_terms is required for stochastic logabsdet estimation"
                    )
                return y, logabsdet_series(self.estimator, fn, x, self.num_series_terms)
