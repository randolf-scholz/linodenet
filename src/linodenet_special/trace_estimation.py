r"""Trace estimators.

Notes:
    Let vₖ = Aᵏv₀,     uₖ = (Aᵀ)ᵏv₀
    then: tr(A²ᵏ) = E[uₖᵀvₖ],  tr(A²ᵏ⁺¹) = E[uₖᵀAvₖ]
"""

__all__ = [
    "ExactEstimator",
    "HutchPlusPlusEstimator",
    "HutchinsonEstimator",
    "LogAbsDetEstimator",
    "SamplerKind",
    "XTraceEstimator",
    # functions
    "btrace_estimator",
    "btrace_estimator_naive",
    "hutchinson_estimator",
    "naive_estimator",
    "xtrace_estimator",
    "xtrace_estimator_corrected",
]

import math
from collections.abc import Callable as Fn, Iterator
from enum import StrEnum
from typing import Final, Protocol, overload

import torch
from torch import Tensor, nn, vmap
from torch.func import jvp, linearize, vjp
from torch.linalg import qr, solve_triangular, vecdot, vector_norm

from signatures import signature


class Sampler(Protocol):
    def __call__(
        self,
        shape: tuple[int, ...],
        num: int,
        *,
        dtype: torch.dtype,
        device: str | torch.device,
    ) -> Tensor: ...


class SamplerKind(StrEnum):
    r"""Built-in probe vector samplers for stochastic trace estimators."""

    GAUSSIAN = "gaussian"
    SIGN = "sign"
    SPHERE = "sphere"
    ORTH = "orth"

    def make(self, *args: object, **kwargs: object) -> Sampler:
        r"""Instantiate the sampler implementation for this built-in sampler."""
        match self:
            case self.GAUSSIAN:
                return GaussianSampler(*args, **kwargs)
            case self.SIGN:
                return SignSampler(*args, **kwargs)
            case self.SPHERE:
                return SphereSampler(*args, **kwargs)
            case self.ORTH:
                return OrthSampler(*args, **kwargs)


class GaussianSampler(nn.Module):
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


class AbstractTraceEstimator(Protocol):
    @signature("[{(..., d) -> (..., d)}?, {(..., d) -> (..., d)}?] -> (...)")
    def estimate(
        self,
        op: Fn[[Tensor], Tensor] | None,
        adj_op: Fn[[Tensor], Tensor] | None,
        /,
        *,
        shape: tuple[int, ...],
    ) -> Tensor:
        r"""Returns an estimate of $\tr(A)$.

        Args:
            op: Linear Operator encoding x ↦ Ax
            adj_op: Linear Operator encode x ↦ Aᵀx
            shape: Shape of `x` the linear operator accepts (may include batch dimension)
        """
        ...

    @signature("[{(..., d) -> (..., d)}?, {(..., d) -> (..., d)}?] -> (...)")
    def estimate_powers(
        self,
        op: Fn[[Tensor], Tensor] | None,
        adj_op: Fn[[Tensor], Tensor] | None,
        /,
        max_power: int,
        *,
        shape: tuple[int, ...],
    ) -> Iterator[Tensor]:
        r"""Yields $\tr(A), \tr(A²), …, \tr(Aᵏ)$ for $k=1..max_power$."""
        ...


class ExactEstimator(nn.Module):
    r"""Estimate traces by explicitly materializing the operator matrix.

    Cost: N³
        N is the dimension of the operator
    """

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("_anchor", torch.empty(0), persistent=False)

    def _materialize(
        self,
        op: Fn[[Tensor], Tensor] | None,
        adj_op: Fn[[Tensor], Tensor] | None,
        /,
        *,
        shape: tuple[int, ...],
    ) -> Tensor:
        if not shape:
            raise ValueError("shape must be non-empty")

        dim = shape[-1]
        identity = torch.eye(
            dim, device=self._anchor.device, dtype=self._anchor.dtype
        ).expand(*shape[:-1], dim, dim)

        if op is not None:
            batched_op = vmap(op, in_dims=-1, out_dims=-1)  # (...dn) -> (...dn)
            return batched_op(identity)

        if adj_op is not None:
            batched_adj = vmap(adj_op, in_dims=-1, out_dims=-1)  # (...dn) -> (...dn)
            return batched_adj(identity)

        raise ValueError("at least one of op or adj_op must be provided")

    @signature("[{(..., d) -> (..., d)}?, {(..., d) -> (..., d)}?] -> (...)")
    def estimate(
        self,
        op: Fn[[Tensor], Tensor] | None,
        adj_op: Fn[[Tensor], Tensor] | None,
        /,
        *,
        shape: tuple[int, ...],
    ) -> Tensor:
        matrix = self._materialize(op, adj_op, shape=shape)
        return torch.einsum("...ii -> ...", matrix)

    @signature("[{(..., d) -> (..., d)}?, {(..., d) -> (..., d)}?] -> (...)")
    def estimate_powers(
        self,
        op: Fn[[Tensor], Tensor] | None,
        adj_op: Fn[[Tensor], Tensor] | None,
        /,
        max_power: int,
        *,
        shape: tuple[int, ...],
    ) -> Iterator[Tensor]:
        if max_power < 1:
            raise ValueError("max_power must be at least 1")

        matrix = self._materialize(op, adj_op, shape=shape)
        eigenvalues = torch.linalg.eigvals(matrix)
        for power in range(1, max_power + 1):
            trace_power = eigenvalues.pow(power).sum(dim=-1)
            yield trace_power.real if not matrix.is_complex() else trace_power


class HutchinsonEstimator(nn.Module):
    r"""Estimate traces with Hutchinson's estimator.

    Cost: mN² + O(m²N + m³)
        m is the number of matvecs (=`num_samples`),
        N is the dimension of the operator
    """

    num_matvecs: Final[int]
    num_samples: Final[int]
    sampler: Sampler

    @overload
    def __init__(
        self, num_samples: int, *, sampler: str | SamplerKind | Sampler = "sphere"
    ) -> None: ...
    @overload
    def __init__(
        self, *, num_matvecs: int, sampler: str | SamplerKind | Sampler = "sphere"
    ) -> None: ...
    def __init__(
        self,
        num_samples: int | None = None,
        *,
        num_matvecs: int | None = None,
        sampler: str | SamplerKind | Sampler = SamplerKind.SPHERE,
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

        self.sampler = (
            SamplerKind(sampler).make() if isinstance(sampler, str) else sampler
        )
        self.register_buffer("_anchor", torch.empty(0), persistent=False)

    @signature("[{(..., d) -> (..., d)}?, {(..., d) -> (..., d)}?] -> (...)")
    def estimate(
        self,
        op: Fn[[Tensor], Tensor] | None,
        adj_op: Fn[[Tensor], Tensor] | None,
        /,
        *,
        shape: tuple[int, ...],
    ) -> Tensor:
        r"""Returns an estimate of $\tr(A)$."""
        return next(self.estimate_powers(op, adj_op, 1, shape=shape))

    @signature("[{(..., d) -> (..., d)}?, {(..., d) -> (..., d)}?] -> (...)")
    def estimate_powers(
        self,
        op: Fn[[Tensor], Tensor] | None,
        adj_op: Fn[[Tensor], Tensor] | None,
        /,
        max_power: int,
        *,
        shape: tuple[int, ...],
    ) -> Iterator[Tensor]:
        r"""Yields $\tr(A), \tr(A²), …, \tr(Aᵏ)$ for $k=1..max_power$."""
        if max_power < 1:
            raise ValueError("max_power must be at least 1")
        if op is None and adj_op is None:
            raise ValueError("at least one of op or adj_op must be provided")
        if not shape:
            raise ValueError("shape must be non-empty")

        right_samples = self.sampler(
            shape,
            self.num_samples,
            device=self._anchor.device,
            dtype=self._anchor.dtype,
        )
        left_samples = right_samples.clone()

        if op is not None and adj_op is not None:
            # alternate between op and adj_op.
            # this is good for forward sensitivity,
            # which grows exponentially in the number of matvecs.
            batched_op = vmap(op, in_dims=-1, out_dims=-1)  # (...dn) -> (...dn)
            batched_adj = vmap(adj_op, in_dims=-1, out_dims=-1)  # (...dn) -> (...dn)

            power = 0
            while power < max_power:
                right_samples = batched_op(right_samples)
                power += 1
                yield vecdot(left_samples, right_samples, dim=-2).mean(dim=-1)

                if power == max_power:
                    return

                left_samples = batched_adj(left_samples)
                power += 1
                yield vecdot(left_samples, right_samples, dim=-2).mean(dim=-1)
            return

        if op is not None:
            batched_op = vmap(op, in_dims=-1, out_dims=-1)  # (...dn) -> (...dn)
            for _ in range(max_power):
                right_samples = batched_op(right_samples)
                yield vecdot(left_samples, right_samples, dim=-2).mean(dim=-1)
            return

        if adj_op is not None:
            batched_adj = vmap(adj_op, in_dims=-1, out_dims=-1)  # (...dn) -> (...dn)
            for _ in range(max_power):
                left_samples = batched_adj(left_samples)
                yield vecdot(left_samples, right_samples, dim=-2).mean(dim=-1)
            return

        raise AssertionError("unreachable")


class HutchPlusPlusEstimator(nn.Module):
    r"""Estimate traces with the Hutch++ variance-reduced estimator.

    Cost: mN² + O(m²N + m³)
        m is the number of matvecs (=3×`num_samples`),
        N is the dimension of the operator
    """

    num_matvecs: Final[int]
    num_samples: Final[int]
    sampler: Sampler

    @overload
    def __init__(
        self, num_samples: int, *, sampler: str | SamplerKind | Sampler = "sphere"
    ) -> None: ...
    @overload
    def __init__(
        self, *, num_matvecs: int, sampler: str | SamplerKind | Sampler = "sphere"
    ) -> None: ...
    def __init__(
        self,
        num_samples: int | None = None,
        *,
        num_matvecs: int | None = None,
        sampler: str | SamplerKind | Sampler = SamplerKind.SPHERE,
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

        self.sampler = (
            SamplerKind(sampler).make() if isinstance(sampler, str) else sampler
        )
        self.register_buffer("_anchor", torch.empty(0), persistent=False)

    @signature("[{(..., d) -> (..., d)}?, {(..., d) -> (..., d)}?] -> (...)")
    def estimate(
        self,
        op: Fn[[Tensor], Tensor] | None,
        adj_op: Fn[[Tensor], Tensor] | None,
        /,
        *,
        shape: tuple[int, ...],
    ) -> Tensor:
        r"""Returns an estimate of $\tr(A)$."""
        return next(self.estimate_powers(op, adj_op, 1, shape=shape))

    @signature("[{(..., d) -> (..., d)}?, {(..., d) -> (..., d)}?] -> (...)")
    def estimate_powers(
        self,
        op: Fn[[Tensor], Tensor] | None,
        adj_op: Fn[[Tensor], Tensor] | None,
        /,
        max_power: int,
        *,
        shape: tuple[int, ...],
    ) -> Iterator[Tensor]:
        if max_power < 1:
            raise ValueError("max_power must be at least 1")
        if op is None and adj_op is None:
            raise ValueError("at least one of op or adj_op must be provided")
        if not shape:
            raise ValueError("shape must be non-empty")

        samples = self.sampler(
            shape,
            self.num_samples,
            device=self._anchor.device,
            dtype=self._anchor.dtype,
        )
        residual_samples = self.sampler(
            shape,
            self.num_samples,
            device=self._anchor.device,
            dtype=self._anchor.dtype,
        )

        if op is not None and adj_op is not None:
            # Two-sided power estimator:
            #   tr(A^(2t-1)) = E[uₜ₋₁ᵀ vₜ]
            #   tr(A^(2t))   = E[uₜᵀ vₜ]
            #
            # Hutch++ uses a fixed projector P = QQᵀ and the exact split
            #   tr(Aᵏ) = tr(Qᵀ Aᵏ Q) + tr((I-P) Aᵏ (I-P)).
            #
            batched_op = vmap(op, in_dims=-1, out_dims=-1)  # (...dn) -> (...dn)
            batched_adj = vmap(adj_op, in_dims=-1, out_dims=-1)  # (...dn) -> (...dn)

            # We build Q from a shared two-sided sketch [AΩ, AᵀΩ]
            left_sketch = batched_adj(samples[..., : self.num_samples // 2])
            right_sketch = batched_adj(samples[..., self.num_samples // 2 :])
            sketch = torch.cat([left_sketch, right_sketch], dim=-1)
            Q, _ = qr(sketch, mode="reduced")  # (...dr)
            projected_samples = Q
            residual_samples = residual_samples - Q @ (Q.mH @ residual_samples)

            residual_l = residual_samples.clone()
            residual_r = residual_samples.clone()
            projected_l = projected_samples.clone()
            projected_r = projected_samples.clone()

            power = 0
            while power < max_power:
                projected_r = batched_op(projected_r)
                residual_r = batched_op(residual_r)
                power += 1

                low_rank = vecdot(projected_l, projected_r, dim=-2).sum(dim=-1)
                residual = vecdot(residual_l, residual_r, dim=-2).mean(dim=-1)
                yield low_rank + residual

                if power == max_power:
                    return

                projected_l = batched_adj(projected_l)
                residual_l = batched_adj(residual_l)
                power += 1

                low_rank = vecdot(projected_l, projected_r, dim=-2).sum(dim=-1)
                residual = vecdot(residual_l, residual_r, dim=-2).mean(dim=-1)
                yield low_rank + residual

            return

        if op is not None:
            # One-sided right-action estimator for tr(Aᵏ):
            #   tr(Aᵏ) = tr(Qᵀ Aᵏ Q) + E[g_⟂ᵀ Aᵏ g_⟂].
            batched_op = vmap(op, in_dims=-1, out_dims=-1)  # (...dn) -> (...dn)
            sketch = batched_op(samples)
            Q, _ = qr(sketch, mode="reduced")  # (...dr)
            projected_samples = Q
            residual_samples = residual_samples - Q @ (Q.mH @ residual_samples)

            residual_l = residual_samples.clone()
            residual_r = residual_samples.clone()
            projected_l = projected_samples.clone()
            projected_r = projected_samples.clone()

            for _ in range(max_power):
                projected_r = batched_op(projected_r)
                residual_r = batched_op(residual_r)

                low_rank = vecdot(projected_l, projected_r, dim=-2).sum(dim=-1)
                residual = vecdot(residual_l, residual_r, dim=-2).mean(dim=-1)
                yield low_rank + residual

            return

        if adj_op is not None:
            # One-sided left-action estimator, equivalently applied to Aᵀ:
            #   tr((Aᵀ)ᵏ) = tr(Aᵏ).
            batched_adj = vmap(adj_op, in_dims=-1, out_dims=-1)  # (...dn) -> (...dn)
            sketch = batched_adj(samples)
            Q, _ = qr(sketch, mode="reduced")  # (...dr)
            projected_samples = Q
            residual_samples = residual_samples - Q @ (Q.mH @ residual_samples)

            residual_l = residual_samples.clone()
            residual_r = residual_samples.clone()
            projected_l = projected_samples.clone()
            projected_r = projected_samples.clone()

            for _ in range(max_power):
                projected_l = batched_adj(projected_l)
                residual_l = batched_adj(residual_l)

                low_rank = vecdot(projected_l, projected_r, dim=-2).sum(dim=-1)
                residual = vecdot(residual_l, residual_r, dim=-2).mean(dim=-1)
                yield low_rank + residual
            return

        raise AssertionError("unreachable")


class XTraceEstimator(nn.Module):
    r"""Estimate traces with the XTrace estimator.

    Cost: mN^2 + O(m^3)
        m is the number of matvecs (=2x`num_samples`),
        N is the dimension of the operator
    """

    num_matvecs: Final[int]
    num_samples: Final[int]
    renormalize: Final[bool]
    r"""Whether to apply renormalization from paper section 2.3"""

    sampler: Sampler

    @overload
    def __init__(
        self,
        num_samples: int,
        *,
        sampler: str | SamplerKind | Sampler = ...,
        renormalize: bool = ...,
    ) -> None: ...
    @overload
    def __init__(
        self,
        *,
        num_matvecs: int,
        sampler: str | SamplerKind | Sampler = ...,
        renormalize: bool = ...,
    ) -> None: ...
    def __init__(
        self,
        num_samples: int | None = None,
        *,
        num_matvecs: int | None = None,
        sampler: str | SamplerKind | Sampler = SamplerKind.SPHERE,
        renormalize: bool = True,
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

        self.renormalize = bool(renormalize)
        self.sampler = (
            SamplerKind(sampler).make() if isinstance(sampler, str) else sampler
        )
        self.register_buffer("_anchor", torch.empty(0), persistent=False)

    @signature("[{(..., d) -> (..., d)}?, {(..., d) -> (..., d)}?] -> (...)")
    def estimate(
        self,
        op: Fn[[Tensor], Tensor] | None,
        adj_op: Fn[[Tensor], Tensor] | None,
        /,
        *,
        shape: tuple[int, ...],
    ) -> Tensor:
        r"""Returns an estimate of $\tr(A)$."""
        return next(self.estimate_powers(op, adj_op, 1, shape=shape))

    @signature("[{(..., d) -> (..., d)}?, {(..., d) -> (..., d)}?] -> (...)")
    def estimate_powers(
        self,
        op: Fn[[Tensor], Tensor] | None,
        adj_op: Fn[[Tensor], Tensor] | None,
        /,
        max_power: int,
        *,
        shape: tuple[int, ...],
    ) -> Iterator[Tensor]:
        if max_power < 1:
            raise ValueError("max_power must be at least 1")
        if max_power > 1:
            raise NotImplementedError("XTraceEstimator currently only supports k=1")
        if op is None and adj_op is None:
            raise ValueError("at least one of op or adj_op must be provided")
        if not shape:
            raise ValueError("shape must be non-empty")

        *batch, N = shape
        k = min(N, self.num_samples)
        samples = self.sampler(
            shape,
            k,
            device=self._anchor.device,
            dtype=self._anchor.dtype,
        )

        if op is not None and adj_op is not None:
            raise NotImplementedError

        if op is not None:
            batched_op = vmap(op, in_dims=-1, out_dims=-1)  # (...Nm) -> (...Nm)
            Y = batched_op(samples)  # (...Nm)

            # Q has normalized cols <-> Q.norm(dim=-2) = 1
            Q, R = qr(Y, mode="reduced")  # (...Nk), (...kk)
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
            yield estimate.real if not estimate.is_complex() else estimate
            return

        if adj_op is not None:
            raise NotImplementedError

        raise AssertionError("unreachable")

    @signature("[{(..., d) -> (..., d)}?, {(..., d) -> (..., d)}?] -> (...)")
    def estimate_alt(
        self,
        op: Fn[[Tensor], Tensor] | None,
        adj_op: Fn[[Tensor], Tensor] | None,
        /,
        *,
        shape: tuple[int, ...],
    ) -> Tensor:
        r"""Returns an estimate of $\tr(A)$."""
        assert op is not None

        *batch, N = shape
        k = min(N, self.num_samples)
        samples = self.sampler(
            shape,
            k,
            device=self._anchor.device,
            dtype=self._anchor.dtype,
        )
        batched_op = vmap(op, in_dims=-1, out_dims=-1)  # (...Nm) -> (...Nm)
        Y = batched_op(samples)  # (...Nm)

        # Q has normalized cols <-> Q.norm(dim=-2) = 1
        Q, R = qr(Y, mode="reduced")  # (...Nk), (...kk)
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
        X = W - SW.unsqueeze(-2) * S
        WS = SW.conj()  # (...i)
        TX = vecdot(T, X, dim=-2)  # (...i)
        XHX = torch.einsum("...ik, ...kl, ...il -> ...i", X.conj(), H, X)
        SHS = torch.einsum("...ik, ...kl, ...il -> ...i", S.conj(), H, S)

        if self.renormalize:
            scale = (N - k + 1) / (  #
                vecdot(samples, samples, dim=-2)
                - vecdot(W, W, dim=-2)
                + SW.abs().square()
            )
        else:
            scale = 1.0

        trs = -SHS + scale * (XHX + WS * SR - TX)

        HW = H @ W
        term1 = SW.abs().square() * SHS  # |⟨sᵢ∣wᵢ⟩|²⟨sᵢ∣Hsᵢ⟩
        term2 = vecdot(S, R - HW, dim=-2) * SW.conj()  # ⟨wᵢ∣sᵢ⟩⟨sᵢ∣rᵢ - Hwᵢ⟩
        term3 = vecdot(T - H.mH @ W, X)  #  -⟨tᵢ - Hᴴwᵢ∣wᵢ - ⟨sᵢ∣wᵢ⟩sᵢ⟩
        trs = -SHS + scale * (term1 + term2 + term3)

        return H.diagonal(dim1=-2, dim2=-1).sum(dim=-1) + trs.mean(dim=-1)


def trace_naive_estimator(
    fn: Fn[[Tensor], Tensor],
    samples: Tensor,
) -> Tensor:
    r"""Naive implementation of XTrace (useful for debugging)."""
    batched_fn = vmap(fn, in_dims=-1, out_dims=-1)  # (...Nm) -> (...Nm)
    Y = batched_fn(samples)  # (...Nm)
    *batch_size, N, m = Y.shape
    tr = torch.zeros(batch_size, device=Y.device, dtype=Y.dtype)
    col_indices = torch.arange(m, device=Y.device)
    for i in range(m):
        Q_i, R = qr(Y[..., i != col_indices], mode="reduced")
        ω_i = samples[..., i]
        μ_i = ω_i - Q_i @ Q_i.mH @ ω_i
        tr = (
            tr
            + torch.einsum("...mm -> ...", Q_i.mh @ fn(Q_i))
            + vecdot(μ_i, fn(μ_i), dim=-1)
        )
    tr = tr / m
    return tr.real if not tr.is_complex() else tr.conj()


@signature("(..., n, d) -> (...)")
def naive_estimator(fn: Fn[[Tensor], Tensor], samples: Tensor) -> Tensor:
    r"""Estimate the trace of a matric, realizing the full matrix."""
    I = torch.eye(samples.shape[-1], dtype=samples.dtype, device=samples.device)
    A = fn(I)
    return torch.einsum("...dd -> ...", A)


@signature("[{(..., n, d) -> (...)}, (..., n, d), (..., n, d)?] -> (...)")
def hutchinson_estimator(
    fn: Fn[[Tensor], Tensor],
    samples: Tensor,
    *,
    left_samples: Tensor | None = None,
) -> Tensor:
    r"""Estimate the trace of a matrix with Hutchinson's estimator.

    .. math:: \tr(A) = E[vᵀAv], where E[vvᵀ] = 𝕀
    .. math:: \tr(A) = E[uᵀAv], where E[uvᵀ] = 𝕀

    Args:
        fn: Matrix-vector product function, i.e. $x ↦ Ax$ (batched).
        samples: Random samples to use for the estimator.
            Shape: `(..., n, d)`, with `...` batch size, `n` number of samples,
            and `d` dimension.
        left_samples: Optional random samples for the left probe vectors in the bilinear estimator.

    Returns:
        Tensor: The estimated trace.
    """
    left_samples = samples if left_samples is None else left_samples
    return vecdot(left_samples, fn(samples), dim=-1).mean(dim=-1)


@signature("[{(..., n, d) -> (..., n, d)}, (..., n, d)] -> (...)")
def xtrace_estimator(fn: Fn[[Tensor], Tensor], samples: Tensor) -> Tensor:
    r"""Estimate the trace of a matric.

    Args:
        fn: matrix-vector product function, i.e. x ↦ Ax (batched)
        samples: random samples to use for the estimator.
            shape: (..., n, d), with `...` batch size, n: num_samples, d: dimension.

    Returns:
        Tensor: The estimated trace.

    core idea:
        samples: [w₁, ..., wₖ]
        compute Qᵢ = orth(AW₋ᵢ)
        compute: trᵢ = tr(QᵢᴴAQᵢ) + wᵢᴴ(I-QᵢQᵢᴴ) A (I-QᵢQᵢᴴ)wᵢ
        trick rank-1 update: QᵢQᵢᴴ = Q(I − sᵢ sᵢᴴ)Qᴴ

    Algorithm:
        1: Draw Ω ∼ Unif{±1}^{N×m/2}
        2: Y ← AΩ
        3: (Q, R) ← qr(Y, ’econ’)
        4: Z ← AQ
        5: H ← QᴴZ, W ← QᴴΩ, T ← ZᴴΩ
        6: S ← R⁻ᴴ
        7: S ← S · diag(∥sᵢ∥: i=1…m/2)
        8: for i = 1 … m/2 do
        9:     xᵢ ← wᵢ − ⟨sᵢ∣wᵢ⟩·sᵢ
        10:    trᵢ ← tr(H) − ⟨sᵢ|H sᵢ⟩ + ⟨wᵢ∣sᵢ⟩·⟨sᵢ∣rᵢ⟩ − ⟨tᵢ|xᵢ⟩ + ⟨xᵢ|Hxᵢ⟩
        11: end for
        12: tr ← mean(trᵢ: i=1…m/2)
    """
    V = samples.mH  # (..., d, n)
    *_, d, n = V.shape
    k = min(n, d)
    Y = fn(V.mH).mH  # (..., d, n)
    Q, R = qr(Y, mode="reduced")  # (..., d, k), (..., k, n)
    Z = fn(Q.mH).mH  # (..., d, k)
    H = torch.einsum("...kd, ...dj -> ...kj", Q.mH, Z)  # (..., k, k)
    W = torch.einsum("...kd, ...dn -> ...nk", Q.mH, V)  # (..., n, k)
    T = torch.einsum("...kd, ...dn -> ...nk", Z.mH, V)  # (..., n, k)

    # Note: compute S=R⁻¹ ⟺ S R = Iₖ  (or: R S = Iₙ)
    I = torch.eye(k, dtype=samples.dtype, device=samples.device)
    S = solve_triangular(I, R.mH, upper=True, left=False)  # (..., n, k)
    S = S / vector_norm(S, dim=-2, keepdim=True)  # (..., n, k)

    # compute xᵢ = wᵢ - ⟨sᵢ∣wᵢ⟩ sᵢ
    X = W - torch.einsum("...nk, ...nk, ...nl -> ...nl", S.conj(), W, S)  # (..., n, k)
    # compute tr_i = ⟨xᵢ|H|xᵢ⟩ - ⟨sᵢ|H|sᵢ⟩ + ⟨wᵢ∣sᵢ⟩⟨sᵢ∣rᵢ⟩ - ⟨tᵢ∣xᵢ⟩
    TRS = (
        torch.einsum("...nk, ...kl, ...nl -> ...n", X.conj(), H, X)  # ⟨xᵢ|H|xᵢ⟩
        - torch.einsum("...nk, ...kl, ...nl -> ...n", S.conj(), H, S)  # - ⟨sᵢ|H|sᵢ⟩
        - torch.einsum("...nk, ...nk -> ...n", T.conj(), X)  # - ⟨tᵢ∣xᵢ⟩
        + (
            torch.einsum("...nk, ...nk -> ...n", W.conj(), S)  # ⟨wᵢ∣sᵢ⟩
            * torch.einsum("...nk, ...kn -> ...n", S.conj(), R)  # ⟨sᵢ∣rᵢ⟩
        )
    )
    # compute tr = tr(H) + mean(tr_i)
    return H.diagonal(dim1=-2, dim2=-1).sum(dim=-1) + TRS.mean(dim=-1)


@signature("(..., n, d) -> (...)")
def xtrace_estimator_corrected(fn: Fn[[Tensor], Tensor], samples: Tensor) -> Tensor:
    r"""Estimate the trace of a matrix using the original XTrace MATLAB algorithm.

    This is a direct Torch transcription of the reference MATLAB code. The input
    `samples` stores row-wise probe vectors with shape `(..., n, d)`. The
    original MATLAB algorithm is written for a column-wise probe matrix
    $Ω ∈ ℝᵈˣᵐ$, so we transpose into column form internally and mirror the
    MATLAB algebra closely.

    Notes:
        The MATLAB reference takes a matvec budget `m_budget` and internally
        uses `m = floor(m_budget / 2)` probe vectors. This function instead
        follows the local API convention that `samples` already contains the
        probe vectors, so all `n` rows are consumed directly.
    """
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


def _normalized_inverse_h_columns(r_factor: Tensor) -> Tensor:
    r"""Return normalized columns of $(R^{-1})ᴴ$ for a square QR factor."""
    *_, rows, cols = r_factor.shape
    identity = torch.eye(cols, dtype=r_factor.dtype, device=r_factor.device)
    inverse_h = solve_triangular(r_factor.mH, identity, upper=False)
    return inverse_h / vector_norm(inverse_h, dim=-2, keepdim=True)


def btrace_estimator(
    fn: Fn[[Tensor], Tensor],
    adj_fn: Fn[[Tensor], Tensor],
    left_samples: Tensor,
    right_samples: Tensor,
) -> Tensor:
    r"""Experimental efficient two-sided XTrace-style estimator.

    This function implements the same experimental two-sided estimator as
    `xtrace_bilinear_estimator_experimental`, but avoids recomputing leave-one-out
    QR factorizations. Instead, it uses the XTrace rank-one update identity on
    both sides:

    .. math::

       QᵢQᵢᴴ = Q(I - sᵢsᵢᴴ)Qᴴ, \qquad PᵢPᵢᴴ = P(I - tᵢtᵢᴴ)Pᴴ,

    where the columns `sᵢ` and `tᵢ` are obtained from the normalized columns of
    `(R_Q^{-1})ᴴ` and `(R_P^{-1})ᴴ`, respectively.

    The implemented estimator uses the projector form

    .. math::

       \hat tᵢ = \tr(Πᴸᵢ A Πᴿᵢ)
              + uᵢᴴ(I - Πᴸᵢ) A (I - Πᴿᵢ) vᵢ,

    with `Πᴿᵢ = QᵢQᵢᴴ` and `Πᴸᵢ = PᵢPᵢᴴ`.

    Notes:
        This remains an experimental generalization. It is intended as an
        efficient implementation vehicle for further moment-estimation work.
    """
    if left_samples.shape != right_samples.shape:
        raise ValueError("left_samples and right_samples must have matching shapes.")

    *_, num_samples, dim = right_samples.shape

    # Right sketch: V columns are the probe vectors, Y = A V.
    # v_cols, av_cols: (..., d, n)
    v_cols = right_samples.mH
    av_cols = fn(right_samples).mH
    # Q: (..., d, n), R_q: (..., n, n)
    q, r_q = qr(av_cols, mode="reduced")
    # S columns are the normalized null-space update vectors sᵢ.
    # s: (..., n, n)
    s = _normalized_inverse_h_columns(r_q)

    # Left sketch: U columns are the probe vectors, AᴴU drives the left basis.
    # u_cols, ahu_cols: (..., d, n)
    u_cols = left_samples.mH
    ahu_cols = adj_fn(left_samples).mH
    # P: (..., d, n), R_p: (..., n, n)
    p, r_p = qr(ahu_cols, mode="reduced")
    # T columns are the normalized update vectors tᵢ for the left basis.
    # t: (..., n, n)
    t = _normalized_inverse_h_columns(r_p)

    # H = Pᴴ A Q, C = Qᴴ P. Shapes: (..., n, n)
    aq = fn(q.mH).mH
    h = torch.einsum("...dp, ...dq -> ...pq", p.conj(), aq)
    c = torch.einsum("...dq, ...dp -> ...qp", q.conj(), p)

    # Projected trace term:
    # tr(Πᴸᵢ A Πᴿᵢ) = tr((I - tᵢtᵢᴴ) H (I - sᵢsᵢᴴ) C)
    #               = tr(HC) - tᵢᴴHC tᵢ - sᵢᴴCH sᵢ + (tᵢᴴHsᵢ)(sᵢᴴCtᵢ)
    hc = h @ c
    ch = c @ h
    trace_hc = hc.diagonal(dim1=-2, dim2=-1).sum(dim=-1, keepdim=True)
    s_cols = s.mH  # (..., n, n), row i contains sᵢᴴ data as a vector
    t_cols = t.mH  # (..., n, n), row i contains tᵢᴴ data as a vector
    d_t_hc_t = torch.einsum("...ni, ...ij, ...nj -> ...n", t_cols.conj(), hc, t_cols)
    d_s_ch_s = torch.einsum("...ni, ...ij, ...nj -> ...n", s_cols.conj(), ch, s_cols)
    d_t_h_s = torch.einsum("...ni, ...ij, ...nj -> ...n", t_cols.conj(), h, s_cols)
    d_s_c_t = torch.einsum("...ni, ...ij, ...nj -> ...n", s_cols.conj(), c, t_cols)
    projected_trace = trace_hc - d_t_hc_t - d_s_ch_s + d_t_h_s * d_s_c_t

    # W = QᴴV and Z = PᴴU collect probe coordinates in the full left/right bases.
    # Shapes: (..., n, n)
    w = torch.einsum("...dq, ...dn -> ...qn", q.conj(), v_cols)
    z = torch.einsum("...dp, ...dn -> ...pn", p.conj(), u_cols)
    w_rows = w.mH  # (..., n, n), row i is wᵢ
    z_rows = z.mH  # (..., n, n), row i is zᵢ

    # xᵢ = wᵢ - <sᵢ, wᵢ> sᵢ,  yᵢ = zᵢ - <tᵢ, zᵢ> tᵢ
    alpha = vecdot(s_cols, w_rows, dim=-1)
    beta = vecdot(t_cols, z_rows, dim=-1)
    x = w_rows - alpha.unsqueeze(-1) * s_cols
    y = z_rows - beta.unsqueeze(-1) * t_cols

    # Residual vectors:
    # (I - Πᴿᵢ) vᵢ = vᵢ - Q xᵢ,   (I - Πᴸᵢ) uᵢ = uᵢ - P yᵢ
    right_residuals = right_samples - torch.einsum("...dq, ...nq -> ...nd", q, x)
    left_residuals = left_samples - torch.einsum("...dp, ...np -> ...nd", p, y)

    # Residual correction uᵢᴴ(I - Πᴸᵢ) A (I - Πᴿᵢ) vᵢ, all samples at once.
    residual_actions = fn(right_residuals)
    residual_trace = vecdot(left_residuals, residual_actions, dim=-1)

    return (projected_trace + residual_trace).mean(dim=-1)


def btrace_estimator_naive(
    fn: Fn[[Tensor], Tensor],
    adj_fn: Fn[[Tensor], Tensor],
    left_samples: Tensor,
    right_samples: Tensor,
) -> Tensor:
    r"""Experimental two-sided XTrace-style estimator for nonsymmetric operators.

    This estimator uses separate left and right probe families. It is based on
    the leave-one-out construction

    .. math::

       \hat tᵢ = \tr(Pᵢᴴ A Qᵢ) + uᵢᴴ(I - PᵢPᵢᴴ) A (I - QᵢQᵢᴴ) vᵢ,

    where:
        - $Qᵢ$ spans the columns of $A V_{-i}$,
        - $Pᵢ$ spans the columns of $Aᴴ U_{-i}$,
        - $vᵢ$ and $uᵢ$ are the held-out right and left probe vectors.

    The final trace estimate is the average of the leave-one-out estimates.
    This is an experimental implementation intended for moment-estimation
    experiments where left and right probe ladders are available explicitly.

    Args:
        fn: Right action of the operator, $x ↦ A x$, applied row-wise to a
            tensor with shape `(..., n, d)`.
        adj_fn: Left action of the adjoint, $x ↦ Aᴴ x$, applied row-wise to a
            tensor with shape `(..., n, d)`.
        left_samples: Left probe vectors `(..., n, d)`.
        right_samples: Right probe vectors `(..., n, d)`.

    Returns:
        Tensor with shape `(...)` containing the experimental trace estimate.
    """
    if left_samples.shape != right_samples.shape:
        raise ValueError("left_samples and right_samples must have matching shapes.")

    *_, num_samples, _ = right_samples.shape
    if num_samples == 0:
        raise ValueError("xtrace_bilinear_estimator_experimental requires samples.")

    av = fn(right_samples)  # (..., n, d), rows are A vᵢ
    ahu = adj_fn(left_samples)  # (..., n, d), rows are Aᴴ uᵢ
    estimates: list[Tensor] = []

    for i in range(num_samples):
        av_except_i = torch.cat((av[..., :i, :], av[..., i + 1 :, :]), dim=-2)
        ahu_except_i = torch.cat((ahu[..., :i, :], ahu[..., i + 1 :, :]), dim=-2)

        if num_samples == 1:
            q = right_samples.new_zeros(
                *right_samples.shape[:-2], right_samples.shape[-1], 0
            )
            p = left_samples.new_zeros(
                *left_samples.shape[:-2], left_samples.shape[-1], 0
            )
            projected_trace = right_samples.new_zeros(right_samples.shape[:-2])
        else:
            # Qᵢ = orth(A V_{-i}), shape (..., d, n-1)
            q, _ = qr(av_except_i.mH, mode="reduced")
            # Pᵢ = orth(Aᴴ U_{-i}), shape (..., d, n-1)
            p, _ = qr(ahu_except_i.mH, mode="reduced")

            # tr(Pᵢᴴ A Qᵢ), where AQᵢ is obtained by applying A to the basis columns.
            aq = fn(q.mH).mH  # (..., d, n-1)
            projected = torch.einsum("...dp, ...dq -> ...pq", p.conj(), aq)
            projected_trace = projected.diagonal(dim1=-2, dim2=-1).sum(dim=-1)

        # vᵢ, uᵢ: held-out probe vectors, shape (..., d)
        v_i = right_samples[..., i, :]
        u_i = left_samples[..., i, :]

        # Right residual: (I - QᵢQᵢᴴ) vᵢ, shape (..., d)
        right_residual = v_i - _project_onto_columns(q, v_i)
        # Left residual: (I - PᵢPᵢᴴ) uᵢ, shape (..., d)
        left_residual = u_i - _project_onto_columns(p, u_i)

        # Residual correction uᵢᴴ(I - PᵢPᵢᴴ) A (I - QᵢQᵢᴴ) vᵢ
        residual_action = fn(right_residual.unsqueeze(-2)).squeeze(-2)
        residual_trace = vecdot(left_residual, residual_action, dim=-1)
        estimates.append(projected_trace + residual_trace)

    return torch.stack(estimates, dim=-1).mean(dim=-1)


def _project_onto_columns(basis: Tensor, vector: Tensor) -> Tensor:
    r"""Project a batched vector onto the column span of a batched basis.

    Args:
        basis: Batched orthonormal basis with shape `(..., d, k)`.
        vector: Batched vector with shape `(..., d)`.

    Returns:
        Tensor with shape `(..., d)` containing the orthogonal projection of
        `vector` onto `span(basis)`.
    """
    coeffs = torch.einsum("...Nk, ...d -> ...k", basis.conj(), vector)
    return torch.einsum("...Nk, ...k -> ...d", basis, coeffs)


def _balanced_biorthogonal_factors(
    cross: Tensor, rtol: float | None = None
) -> tuple[Tensor, Tensor]:
    r"""Return balanced right/left factors for a small cross-pairing matrix.

    Given the singular value decomposition

    .. math::

       \text{cross} = U \Sigma Vᴴ,

    this returns the factors

    .. math::

       S = V \Sigma^{\dagger/2}, \qquad T = U \Sigma^{\dagger/2},

    where :math:`\Sigma^{\dagger/2}` is the square root of the Moore--Penrose
    pseudoinverse of :math:`\Sigma`. The transformed bases satisfy

    .. math::

       Tᴴ \, \text{cross} \, S = J,

    where :math:`J` is the diagonal projector onto the numerically nondegenerate
    singular subspace. In particular, full-rank directions are normalized to one,
    while singular or numerically tiny directions are mapped to zero instead of
    being inverted.

    Args:
        cross: Batched square cross-pairing matrix of shape ``(..., n, n)``.
        rtol: Relative singular-value cutoff. If omitted, a dtype-based default is
            used.

    Returns:
        A pair ``(S, T)`` with shapes ``(..., n, n)``.
    """
    u_svd, sigma, vh = torch.linalg.svd(cross, full_matrices=False)

    if rtol is None:
        rtol = max(cross.shape[-2], cross.shape[-1]) * torch.finfo(sigma.dtype).eps

    tol = rtol * sigma.amax(dim=-1, keepdim=True)
    keep = sigma > tol
    safe_sigma = torch.where(keep, sigma, torch.ones_like(sigma))
    sigma_inv_sqrt = torch.where(keep, safe_sigma.rsqrt(), torch.zeros_like(sigma))

    d = torch.diag_embed(sigma_inv_sqrt).to(dtype=cross.dtype)
    s = vh.mH @ d
    t = u_svd @ d
    return s, t


def btrace_estimator_new(
    fn: Fn[[Tensor], Tensor],
    adj_fn: Fn[[Tensor], Tensor],
    left_samples: Tensor,
    right_samples: Tensor,
) -> Tensor:
    r"""Experimental efficient two-sided estimator with balanced A-biorthogonal sketches.

    This function builds orthonormal sketch bases from the right and left probe
    actions,

    .. math::

       Y = A V, \qquad Z = Aᴴ U,

    via reduced QR factorizations

    .. math::

       Y = Q R_Q, \qquad Z = P R_P.

    It then forms the small cross matrix

    .. math::

       H = Pᴴ A Q,

    and replaces the separate QR-side normalizations by a balanced
    pseudoinverse-square-root biorthogonalization

    .. math::

       H = U_H \Sigma_H V_Hᴴ, \qquad
       S = V_H \Sigma_H^{\dagger/2}, \qquad
       T = U_H \Sigma_H^{\dagger/2},

    yielding transformed sketch bases

    .. math::

       \widetilde Q = Q S, \qquad \widetilde P = P T,

    such that

    .. math::

       \widetilde Pᴴ A \widetilde Q = J,

    where :math:`J` is the diagonal projector onto the numerically nondegenerate
    paired subspace. If :math:`H` is full rank, then :math:`J = I`. If :math:`H`
    is rank-deficient (including the case :math:`A = 0`), degenerate directions
    are zeroed rather than inverted.

    The estimator uses the leave-one-out oblique projector pair

    .. math::

       \Pi_i^R = \widetilde Q (I - e_i e_iᴴ) \widetilde Pᴴ A,
       \qquad
       \Pi_i^L = \widetilde P (I - e_i e_iᴴ) \widetilde Qᴴ Aᴴ,

    and computes

    .. math::

       \hat t_i
       = \operatorname{tr}(\Pi_i^L A \Pi_i^R)
       + u_iᴴ (I - \Pi_i^L) A (I - \Pi_i^R) v_i.

    Notes:
        This no longer uses the orthogonal XTrace rank-one identities based on the
        normalized columns of ``(R^{-1})ᴴ``. Instead, it uses a balanced
        A-biorthogonal / oblique projector construction derived from the small
        cross-pairing matrix ``Pᴴ A Q``. The construction is defined uniformly even
        when that cross matrix is singular, although exact identity biorthogonality
        is then impossible.
    """
    if left_samples.shape != right_samples.shape:
        raise ValueError("left_samples and right_samples must have matching shapes.")

    # Probe matrices with columns as sample vectors.
    # U, V: (..., d, n)
    u_cols = left_samples.mH
    v_cols = right_samples.mH

    # Right sketch: Y = A V = Q R_Q with Q orthonormal.
    # av_cols, q: (..., d, n)
    av_cols = fn(right_samples).mH
    q, _ = qr(av_cols, mode="reduced")

    # Left sketch: Z = Aᴴ U = P R_P with P orthonormal.
    # ahu_cols, p: (..., d, n)
    ahu_cols = adj_fn(left_samples).mH
    p, _ = qr(ahu_cols, mode="reduced")

    # Small cross-pairing matrix H = Pᴴ A Q.
    # aq, h: (..., d, n), (..., n, n)
    aq = fn(q.mH).mH
    h = p.mH @ aq

    # Balanced A-biorthogonalization on the small cross matrix:
    #   q_b = Q S,  p_b = P T,  with  p_bᴴ A q_b = J.
    s, t = _balanced_biorthogonal_factors(h)
    q_b = q @ s
    p_b = p @ t

    # Reuse A Q = aq to obtain A q_b = (A Q) S.
    aq_b = aq @ s

    # Oblique projected trace term:
    #
    #   tr(Π_i^L A Π_i^R)
    #     = tr((I - E_i) G (I - E_i) F),
    #
    # where
    #   G = q_bᴴ Aᴴ A q_b = (A q_b)ᴴ (A q_b),
    #   F = p_bᴴ A p_b.
    #
    # Expanding with E_i = e_i e_iᴴ gives
    #   tr(GF) - (GF)_{ii} - (FG)_{ii} + G_{ii} F_{ii}.
    g = aq_b.mH @ aq_b

    # Compute F = p_bᴴ A p_b via the untransformed left basis and then apply T.
    ap = fn(p.mH).mH
    f = t.mH @ (p.mH @ ap) @ t

    gf = g @ f
    fg = f @ g
    trace_gf = gf.diagonal(dim1=-2, dim2=-1).sum(dim=-1, keepdim=True)
    diag_gf = gf.diagonal(dim1=-2, dim2=-1)
    diag_fg = fg.diagonal(dim1=-2, dim2=-1)
    diag_g = g.diagonal(dim1=-2, dim2=-1)
    diag_f = f.diagonal(dim1=-2, dim2=-1)
    projected_trace = trace_gf - diag_gf - diag_fg + diag_g * diag_f

    # Coefficients of each probe against the full biorthogonal sketch pair:
    #   w_i = p_bᴴ A v_i,
    #   z_i = q_bᴴ Aᴴ u_i.
    #
    # Leaving out i-th column amounts to zeroing the i-th coefficient.
    w = p_b.mH @ av_cols
    z = q_b.mH @ ahu_cols
    w_rows = w.mH
    z_rows = z.mH
    x = w_rows - torch.diag_embed(w.diagonal(dim1=-2, dim2=-1))
    y = z_rows - torch.diag_embed(z.diagonal(dim1=-2, dim2=-1))

    # Residual vectors:
    #   (I - Π_i^R) v_i = v_i - q_b x_i,
    #   (I - Π_i^L) u_i = u_i - p_b y_i.
    right_residuals = right_samples - torch.einsum("...Nk, ...nk -> ...nd", q_b, x)
    left_residuals = left_samples - torch.einsum("...Nk, ...nk -> ...nd", p_b, y)

    # Residual correction u_iᴴ (I - Π_i^L) A (I - Π_i^R) v_i.
    residual_actions = fn(right_residuals)
    residual_trace = vecdot(left_residuals, residual_actions, dim=-1)

    return (projected_trace + residual_trace).mean(dim=-1)


class LogAbsDetEstimator(nn.Module):
    r"""Estimate log|det(𝕀 + ∂f/∂x)| using the power series expansion and a trace estimator.

    - \log|\det A| = \Re(\tr(\log A)) for any A in the image of the matrix exponential
    - \log|\det A| = ½\tr(\log AᴴA) for any A. (-∞ if A is singular)

    Args:
        method: str in {"exact", "hutch", "xtrace"} specifying the estimation method to use.
        num_samples: Number of random samples to use for the Hutchinson or XTrace estimator.
        num_series_terms: Number of terms to use in the power series expansion.

    Returns:
        y: fn(x)
        logabsdet: Approximation of log|det(𝕀 + ∂f/∂x)|
    """

    num_samples: int | None
    num_series_terms: int | None
    method: Fn[[Fn[[Tensor], Tensor], Tensor], tuple[Tensor, Tensor]]

    def __init__(
        self,
        method: str,
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

        match method:
            case "exact":
                self.method = self.compute_exact
            case "hutch" | "hutchinson":
                if num_samples is None:
                    raise ValueError("num_samples is required for method='hutch'")
                if num_series_terms is None:
                    raise ValueError("num_series_terms is required for method='hutch'")
                self.method = self.compute_hutch
            case "xtrace":
                if num_samples is None:
                    raise ValueError("num_samples is required for method='xtrace'")
                if num_series_terms is None:
                    raise ValueError("num_series_terms is required for method='xtrace'")
                self.method = self.compute_xtrace
            case _:
                raise ValueError(f"Unknown logabsdet estimation method {method!r}")

    @signature("[{(..., d) -> (..., d)}, (..., d)] -> (...)")
    def compute_exact(
        self, fn: Fn[[Tensor], Tensor], x: Tensor
    ) -> tuple[Tensor, Tensor]:
        r"""Compute the exact log-absolute-determinant via the full Jacobian spectrum."""
        y, df = linearize(fn, x)  # (...d), {(...d) -> (...d)}
        batched_df = vmap(df, in_dims=-2, out_dims=-2)  # {(...nd) -> (...nd)}
        dim = y.shape[-1]
        I = torch.eye(dim, dim, device=y.device).expand(*y.shape[:-1], dim, dim)

        # log|det(I+A)| = log|∏(1 + λᵢ)| = ∑log|1 + λᵢ|
        # where λᵢ are the eigenvalues of A. This holds even for non-diagonalizable A.
        jacobian = batched_df(I)
        eigenvalues = torch.linalg.eigvals(jacobian)
        logabsdet = torch.log(torch.abs(1 + eigenvalues)).sum(dim=-1)
        return y, logabsdet

    @signature("[{(..., d) -> (..., d)}, (..., d)] -> (...)")
    def compute_hutch(
        self, fn: Fn[[Tensor], Tensor], x: Tensor
    ) -> tuple[Tensor, Tensor]:
        r"""Estimate the log-absolute-determinant using a power series and Hutchinson."""
        assert self.num_samples is not None
        assert self.num_series_terms is not None

        y, jvp_fn = jvp(fn, x)
        y, vjp_fn = vjp(fn, x)
        batched_jvp_fn = vmap(jvp_fn, in_dims=-2, out_dims=-2)  # (...nd) -> (...nd)
        right_samples = torch.randn(  # (...dn)
            (*x.shape, self.num_samples),
            device=x.device,
            dtype=x.dtype,
        )
        left_samples = right_samples.clone()

        logabsdet = torch.zeros(x.shape[:-1], device=x.device, dtype=x.dtype)
        sign = torch.tensor(-1.0, device=x.device, dtype=x.dtype)
        for k in range(1, self.num_series_terms + 1):
            # log|det(I+A)| = Re(tr(log I+A)) = Re(∑_{k≥1} (-1)ᵏ⁺¹/k tr(Aᵏ))
            sign = sign.neg()

            # hutch impl, using tr(Aᵏ⁺¹) = E[v₀ᵀAvₖ], vₖ=Avₖ₋₁
            right_samples = batched_jvp_fn(right_samples)
            tr_k_power = vecdot(left_samples, right_samples, dim=-1).mean(dim=-1)

            logabsdet = logabsdet + (sign / k) * tr_k_power

        return y, logabsdet

    @signature("[{(..., d) -> (..., d)}, (..., d)] -> (...)")
    def compute_hutch_twosided(
        self, fn: Fn[[Tensor], Tensor], x: Tensor
    ) -> tuple[Tensor, Tensor]:
        r"""Estimate the log-absolute-determinant using a power series and Hutchinson."""
        assert self.num_samples is not None
        assert self.num_series_terms is not None

        y, jvp_fn = jvp(fn, x)
        y, vjp_fn = vjp(fn, x)
        batched_jvp_fn = vmap(jvp_fn, in_dims=-2, out_dims=-2)  # (...nd) -> (...nd)
        batched_vjp_fn = vmap(jvp_fn, in_dims=-2, out_dims=-2)  # (...nd) -> (...nd)
        right_samples = torch.randn(  # (...dn)
            (*x.shape, self.num_samples),
            device=x.device,
            dtype=x.dtype,
        )
        left_samples = right_samples.clone()

        logabsdet = torch.zeros(x.shape[:-1], device=x.device, dtype=x.dtype)
        for k in range(1, self.num_series_terms + 1, 2):
            # log|det(I+A)| = Re(tr(log I+A)) = Re(∑_{k≥1} (-1)ᵏ⁺¹/k tr(Aᵏ))

            # tr(A²ᵏ⁻¹) = E[u₀ᵀA²ᵏ⁻¹v₀] = E[uₖ₋₁ᵀvₖ], vₖ=Aᵏv₀, uₖ=(Aᵀ)ᵏu₀
            right_samples = batched_jvp_fn(right_samples)  # vₖ₊₁ = A vₖ
            tr_odd_power = vecdot(left_samples, right_samples, dim=-1).mean(dim=-1)

            # tr(A²ᵏ) = E[u₀ᵀA²ᵏv₀] = E[uₖᵀvₖ],  vₖ=Aᵏv₀, uₖ=(Aᵀ)ᵏu₀
            left_samples = batched_vjp_fn(left_samples)  # uₖ₊₁ = Aᵀvₖ
            tr_even_power = vecdot(left_samples, right_samples, dim=-1).mean(dim=-1)

            logabsdet = logabsdet + tr_odd_power / k
            logabsdet = logabsdet - tr_even_power / (k + 1)

        return y, logabsdet

    @signature("[{(..., d) -> (..., d)}, (..., d)] -> (...)")
    def compute_xtrace(
        self, fn: Fn[[Tensor], Tensor], x: Tensor
    ) -> tuple[Tensor, Tensor]:
        r"""Estimate the log-absolute-determinant using a power series and XTrace."""
        assert self.num_samples is not None
        assert self.num_series_terms is not None

        y, jvp_fn = linearize(fn, x)  # (...d)  -> (...d)
        batched_jvp_fn = vmap(jvp_fn, in_dims=-1, out_dims=-1)  # (...dn) -> (...dn)
        samples = torch.randn(  # (...dn)
            (*x.shape, self.num_samples),
            device=x.device,
            dtype=x.dtype,
        )
        logabsdet = torch.zeros(x.shape[:-1], device=x.device, dtype=x.dtype)
        sign = torch.tensor(-1.0, device=x.device, dtype=x.dtype)
        for k in range(1, self.num_series_terms + 1):
            # log|det(I+A)| = Re(tr(log I+A)) = Re(∑_{k≥1} (-1)ᵏ⁺¹/k tr(Aᵏ))
            sign = sign.neg()
            samples = batched_jvp_fn(samples)
            tr_k_power = xtrace_estimator(batched_jvp_fn, samples)
            logabsdet = logabsdet + (sign / k) * tr_k_power

        return y, logabsdet

    @signature("[{(..., d) -> (..., d)}, (..., d)] -> (...)")
    def forward(self, fn: Fn[[Tensor], Tensor], x: Tensor) -> tuple[Tensor, Tensor]:
        return self.method(fn, x)
