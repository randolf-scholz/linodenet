r"""ContractiveFlow implementation (iResNet-block)."""

__all__ = [
    "ResidualContraction",
    "ReZeroContraction",
    "ResidualContractionFallback",
    "vector_logabsdet_hutchinson_estimator",
    "vector_logabsdet_xtrace_estimator",
]

import warnings
from collections.abc import Callable
from typing import Final

import torch
from torch import Tensor, nn
from torch.func import linearize, vmap
from torch.linalg import qr, vecdot

from linodenet.mappings.base import TransformBase
from linodenet.mappings.bijections import SmoothSoftsign, TanhMap
from linodenet.nn import ReZero
from linodenet_special import fixpoint_solve
from signatures import signature


@signature("(..., d) -> [(..., d), (...)]")
def vector_logabsdet_hutchinson_estimator(
    fn: Callable[[Tensor], Tensor],
    x: Tensor,
    num_terms: int,
    num_samples: int,
) -> tuple[Tensor, Tensor]:
    r"""Estimate log|det(𝕀 + ∂f/∂x)| using the power series expansion and Hutchinson's trace estimator.

    Args:
        fn: The function for which to compute the Jacobian determinant.
        x: The point at which to evaluate the Jacobian determinant.
           Assumed to be of shape [..., d]
        # event_shape: the shape of the event samples.
        num_terms: The order of the series expansion.
        num_samples: The number of random samples.

    Returns:
        y: fn(x)
        logabsdet: Approximation of log|det(𝕀 + ∂f/∂x)|
    """
    y, jvp_fn = linearize(fn, x)
    # note: or None fixes event_shape=() case.
    batched_jvp_fn = vmap(jvp_fn)  # support num_samples

    v0 = torch.randn(
        num_samples,
        *x.shape,
        device=x.device,
        dtype=x.dtype,
    )
    v = v0.clone()
    logabsdet = torch.zeros(x.shape[:-1], device=x.device, dtype=x.dtype)

    for k in range(1, num_terms + 1):
        v = batched_jvp_fn(v)  # Aᵏv
        coef = 1.0 / k if k % 2 else -1.0 / k
        logabsdet = logabsdet + coef * vecdot(v, v0).mean(dim=0)

    return y, logabsdet


@signature("(..., d) -> [(..., d), (...)]")
def vector_logabsdet_xtrace_estimator(
    fn: Callable[[Tensor], Tensor],
    x: Tensor,
    num_terms: int,
    num_samples: int,
) -> tuple[Tensor, Tensor]:
    r"""Estimate log|det(𝕀 + ∂f/∂x)| using the power series expansion and the XTrace estimator."""
    V = (
        2
        * torch.rand(  # samples ~ Uniform[-1, +1]
            num_samples,
            *x.shape,
            device=x.device,
            dtype=x.dtype,
        )
        - 1
    )
    y, jvp_fn = linearize(fn, x)
    batched_jvp_fn = vmap(jvp_fn)
    logabsdet = torch.zeros(x.shape[:-1], device=x.device, dtype=x.dtype)

    v = V.clone()

    for k in range(1, num_terms + 1):
        v = batched_jvp_fn(v)  # Aᵏv
        coef = 1.0 / k if k % 2 else -1.0 / k
        trace_estimate = torch.zeros_like(logabsdet)

        for i in range(num_samples):
            y_without_i = torch.cat((v[:i], v[i + 1 :]), dim=0)
            if num_samples == 1:
                q_i = v.new_zeros(*x.shape, 0)
            else:
                q_i, _ = qr(y_without_i.movedim(0, -1), mode="reduced")

            if q_i.shape[-1] == 0:
                residual = V[i]
                projected_av = residual
                for _ in range(k):
                    projected_av = jvp_fn(projected_av)
                trace_estimate = trace_estimate + vecdot(residual, projected_av)
                continue

            aq_i = q_i.movedim(-1, 0)
            for _ in range(k):
                aq_i = batched_jvp_fn(aq_i)
            aq_i = aq_i.movedim(0, -1)
            qaq = torch.einsum("...dk,...dk->...", q_i, aq_i)

            coeffs = torch.einsum("...dk,...d->...k", q_i, V[i])
            residual = V[i] - torch.einsum("...dk,...k->...d", q_i, coeffs)
            a_residual = residual
            for _ in range(k):
                a_residual = jvp_fn(a_residual)
            projected_a_residual = a_residual - torch.einsum(
                "...dk,...k->...d",
                q_i,
                torch.einsum("...dk,...d->...k", q_i, a_residual),
            )
            correction = vecdot(residual, projected_a_residual)
            trace_estimate = trace_estimate + qaq + correction

        logabsdet = logabsdet + coef * trace_estimate / num_samples

    return y, logabsdet


class ResidualContraction(TransformBase):
    r"""A residual flow based on a contraction layer.

    Forward: y ← x + g(x)
    Inverse: via fix-point iteration.

    The jacobian determinant of the forward transformation is:

    .. math:: \log\det(∂y/∂x) = \log\det(𝕀 + ∂g/∂x) = \tr\log(𝕀 + ∂g/∂x)

    Using the power series, this is

    .. math:: ∑_k (-1)ᵏ⁺¹ \tr((∂g/∂x)ᵏ)/k

    The trace of A can be estimated with the Hutchinson trace estimator:

    .. math:: \tr(A) = 𝐄[vᵀAv] \qquad 𝐄[v]=0, \Cov[v]=𝕀

    References:
        - https://github.com/jhjacobsen/invertible-resnet
        - | Invertible Residual Networks
          | Jens Behrmann, Will Grathwohl, Ricky T. Q. Chen, David Duvenaud, Jörn-Henrik Jacobsen
          | International Conference on Machine Learning 2019
          | https://proceedings.mlr.press/v97/behrmann19a.html
        - | A stochastic estimator of the trace of the influence matrix for laplacian smoothing splines
          | M.F. Hutchinson
          | Communications in Statistics - Simulation and Computation 1990
          | https://doi.org/10.1080/03610919008812866

    See Also:
        - `ReZeroContraction`: adds a learnable scalar ε and parametrization
            to the contraction layer:. $y ← x + φ(ε)⋅g(x)$ with $φ(ε) ∈ (-1, 1)$.
            ε is initialized to 0, so that the initial transformation is the identity.
        - `ResidualContractionFallback`: Uses a plain python loop rather than `torch.while_loop`.
    """

    maxiter: Final[int]
    atol: Final[float]
    rtol: Final[float]
    num_trace_samples: Final[int]
    num_series_terms: Final[int]
    trace_estimator: Final[str]

    def __init__(
        self,
        contraction: nn.Module,
        maxiter: int = 256,
        atol: float = 1e-6,
        rtol: float = 1e-6,
        *,
        num_trace_samples: int = 1,
        num_series_terms: int = 8,
        trace_estimator: str = "hutchinson",
    ) -> None:
        super().__init__()
        if num_trace_samples < 1:
            raise ValueError("num_trace_samples must be at least 1")
        if num_series_terms < 1:
            raise ValueError("num_series_terms must be at least 1")
        if trace_estimator not in {"hutchinson", "xtrace"}:
            raise ValueError(
                f"trace_estimator must be 'hutchinson' or 'xtrace', got {trace_estimator!r}"
            )
        self.contraction: nn.Module = contraction
        self.maxiter = maxiter
        self.atol = atol
        self.rtol = rtol
        self.num_trace_samples = num_trace_samples
        self.num_series_terms = num_series_terms
        self.trace_estimator = trace_estimator

    def encode(self, x: Tensor) -> Tensor:
        return x + self.contraction(x)

    def encode_and_logabsdet(self, x: Tensor, /) -> tuple[Tensor, Tensor]:
        match self.trace_estimator:
            case "hutchinson":
                fx, logabsdet = vector_logabsdet_hutchinson_estimator(
                    self.contraction,
                    x,
                    self.num_series_terms,
                    self.num_trace_samples,
                )
            case "xtrace":
                fx, logabsdet = vector_logabsdet_xtrace_estimator(
                    self.contraction,
                    x,
                    self.num_series_terms,
                    self.num_trace_samples,
                )
            case _:
                raise ValueError(f"Unknown trace_estimator {self.trace_estimator!r}")
        return x + fx, logabsdet

    def decode(self, y: Tensor) -> Tensor:
        r"""Compute the inverse through fixed point iteration."""
        # note: solve x = y - g(x) = f(x, y)
        return fixpoint_solve(
            lambda x: y - self.contraction(x),  # type: ignore[misc]
            y.clone(),
            maxiter=self.maxiter,
            atol=self.atol,
            rtol=self.rtol,
        )

    def decode_and_logabsdet(self, y: Tensor, /) -> tuple[Tensor, Tensor]:
        raise NotImplementedError


class ReZeroContraction[M: nn.Module](ResidualContraction):
    r"""A residual flow based on a scaled contraction layer.

    .. math:: y ← x + φ(ε)⋅g(x)  \qquad  φ(ε) ∈ (-1, 1), φ(0)=0

    ε is initialized to 0, so that the initial transformation is the identity.

    See Also:
        - `ResidualContraction`
        - `ResidualContractionFallback`
    """

    contraction: ReZero[M]
    scalar: Tensor
    scalar_map: nn.Module

    def __init__(
        self,
        contraction: M,
        *,
        scalar_map: nn.Module | str | None = "smooth-softsign",
        maxiter: int = 256,
        atol: float = 1e-6,
        rtol: float = 1e-6,
        num_trace_samples: int = 1,
        num_series_terms: int = 8,
        trace_estimator: str = "hutchinson",
    ) -> None:
        scalar_map_module: nn.Module
        match scalar_map:
            case None | "smooth-softsign":
                scalar_map_module = SmoothSoftsign()
            case "tanh":
                scalar_map_module = TanhMap()
            case str(other):
                raise ValueError(f"Unknown scalar map {other}")
            case _:
                scalar_map_module = scalar_map
        rezero = ReZero(contraction, scalar_map=scalar_map_module)
        super().__init__(
            rezero,
            maxiter=maxiter,
            atol=atol,
            rtol=rtol,
            num_trace_samples=num_trace_samples,
            num_series_terms=num_series_terms,
            trace_estimator=trace_estimator,
        )
        self.contraction: ReZero[M]  # pyright: ignore[reportIncompatibleVariableOverride]
        self.scalar = self.contraction.scalar
        self.scalar_map = self.contraction.scalar_map


class ResidualContractionFallback(ResidualContraction):
    r"""Fallback implementation of ResidualContraction that uses a plain python loop.

    See Also:
        - `ResidualContraction`
        - `ReZeroContraction`
    """

    def __init__(
        self,
        contraction: nn.Module,
        maxiter: int = 256,
        atol: float = 1e-6,
        rtol: float = 1e-6,
        *,
        num_trace_samples: int = 1,
        num_series_terms: int = 8,
        trace_estimator: str = "hutchinson",
    ) -> None:
        super().__init__(
            contraction=contraction,
            maxiter=maxiter,
            atol=atol,
            rtol=rtol,
            num_trace_samples=num_trace_samples,
            num_series_terms=num_series_terms,
            trace_estimator=trace_estimator,
        )

    def decode(self, y: Tensor) -> Tensor:
        x = y.clone()

        for _ in range(self.maxiter):
            x_prev = x
            x = y - self.contraction(x_prev)
            residual = torch.abs(x - x_prev)
            tolerance = self.rtol * torch.abs(x) + self.atol

            if torch.all(residual <= tolerance):
                return x

        warnings.warn(
            f"No convergence in {self.maxiter} iterations. ",
            stacklevel=2,
        )
        return x

    def decode_and_logabsdet(self, y: Tensor, /) -> tuple[Tensor, Tensor]:
        raise NotImplementedError
