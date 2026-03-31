r"""ContractiveFlow implementation (iResNet-block)."""

__all__ = [
    "ResidualContraction",
    "ReZeroContraction",
    "ResidualContractionFallback",
]

import warnings
from typing import Final

import torch
from torch import Tensor, nn

from linodenet.mappings.bijections import SmoothSoftsign, TanhMap
from linodenet.nn import ReZero
from linodenet_special import fixpoint_solve
from linodenet_special.trace_estimation import LogAbsDetEstimators

from .base import TransformBase


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
        trace_estimator: str = "hutch",
        trace_matvecs: int = 3,
        num_series_terms: int = 8,
        trace_probe_sampler: str = "sphere",
        trace_mode: str = "adjoint",
    ) -> None:
        super().__init__()
        self.contraction: nn.Module = contraction
        self.maxiter = maxiter
        self.atol = atol
        self.rtol = rtol
        self.num_trace_samples = trace_matvecs
        self.num_series_terms = num_series_terms
        self.trace_estimator = trace_estimator
        self.logabsdet_estimator = LogAbsDetEstimators.new(
            trace_estimator,
            num_matvecs=trace_matvecs,
            num_terms=num_series_terms,
            sampler=trace_probe_sampler,
            mode=trace_mode,
        )

    def encode(self, x: Tensor) -> Tensor:
        return x + self.contraction(x)

    def encode_and_logabsdet(self, x: Tensor, /) -> tuple[Tensor, Tensor]:
        return self.logabsdet_estimator(self.contraction, x)

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
        trace_matvecs: int = 1,
        num_series_terms: int = 8,
        trace_estimator: str = "hutch",
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
            trace_matvecs=trace_matvecs,
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
        trace_matvecs: int = 1,
        num_series_terms: int = 8,
        trace_estimator: str = "hutch",
    ) -> None:
        super().__init__(
            contraction=contraction,
            maxiter=maxiter,
            atol=atol,
            rtol=rtol,
            trace_matvecs=trace_matvecs,
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
