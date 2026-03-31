r"""ContractiveFlow implementation (iResNet-block)."""

__all__ = [
    "ResidualContraction",
    "IResNetContraction",
    "ResidualBottleneck",
    "ReZeroBottleneck",
    "IReZeroContraction",
    "ResidualContractionFallback",
]

import warnings
from abc import abstractmethod
from math import sqrt
from typing import Final

import torch
from torch import Tensor, nn
from torch.func import vjp, vmap
from torch.linalg import slogdet
from torch.nn.functional import linear

from linodenet.mappings.bijections import SmoothSoftsign, TanhMap
from linodenet.mappings.nonlinear_contractions import get_nonlinear_contraction
from linodenet.mappings.surjections import OrthogonalHouseholder
from linodenet.nn import ReZero
from linodenet.nn.parametrize import register_parametrization, update_parametrizations
from linodenet_special import fixpoint_solve
from linodenet_special.trace_estimation import LogAbsDetEstimators

from .base import TransformBase


class ResidualContraction(TransformBase):
    r"""Shared base class for residual contractions with implicit inverses.

    Forward: y ← x + g(x)
    Inverse: via fix-point iteration.
    """

    maxiter: Final[int]
    atol: Final[float]
    rtol: Final[float]

    def __init__(
        self,
        *,
        maxiter: int = 256,
        atol: float = 1e-6,
        rtol: float = 1e-6,
    ) -> None:
        super().__init__()
        self.maxiter = maxiter
        self.atol = atol
        self.rtol = rtol

    @abstractmethod
    def contraction(self, x: Tensor, /) -> Tensor: ...

    def encode(self, x: Tensor, /) -> Tensor:
        r"""Compute the forward residual map $y = x + g(x)$."""
        return x + self.contraction(x)

    def encode_and_logabsdet(self, x: Tensor, /) -> tuple[Tensor, Tensor]:
        r"""Compute the forward residual map and its log absolute Jacobian determinant."""
        y = self.encode(x)

        # materialize the full jacobian of y = x + g(x) with vjp
        _, vjp_fn, *_ = vjp(self.encode, x)
        eye_n = torch.eye(x.shape[-1], device=x.device, dtype=x.dtype)
        eye_n = eye_n.expand(*x.shape, x.shape[-1])
        batched_vjp_fn = vmap(lambda w: vjp_fn(w)[0], -1, -1)
        matrix = batched_vjp_fn(eye_n)
        _, logabsdet = slogdet(matrix)
        return y, logabsdet

    def decode(self, y: Tensor, /) -> Tensor:
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
        r"""Compute the inverse residual map and its log absolute Jacobian determinant."""
        x = self.decode(y)
        _, logabsdet = self.encode_and_logabsdet(x)
        return x, -logabsdet


class IResNetContraction[M: nn.Module](ResidualContraction):
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

    num_trace_samples: Final[int]
    num_series_terms: Final[int]
    trace_estimator: Final[str]
    module: M

    def __init__(
        self,
        contraction: M,
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
        super().__init__(maxiter=maxiter, atol=atol, rtol=rtol)
        self.module = contraction
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

    def contraction(self, x: Tensor, /) -> Tensor:
        return self.module(x)

    def encode_and_logabsdet(self, x: Tensor, /) -> tuple[Tensor, Tensor]:
        return self.logabsdet_estimator(self.module, x)


class IReZeroContraction[M: nn.Module](IResNetContraction[ReZero[M]]):
    r"""A residual flow based on a scaled contraction layer.

    .. math:: y ← x + φ(ε)⋅g(x)  \qquad  φ(ε) ∈ (-1, 1), φ(0)=0

    ε is initialized to 0, so that the initial transformation is the identity.

    See Also:
        - `ResidualContraction`
        - `ResidualContractionFallback`
    """

    module: ReZero[M]
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
        self.module: ReZero[M]
        self.scalar = self.module.scalar
        self.scalar_map = self.module.scalar_map


class ResidualContractionFallback(IResNetContraction):
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


class ResidualBottleneck[B: nn.Module](ResidualContraction):
    r"""Residual low-rank contraction with exact low-dimensional logabsdet.

    The transformation is

    .. math:: y = x + ϕᵤ(U f(Vᵀx))

    where $U,V∈ℝ^{n×k}$ have orthonormal columns, $f:ℝᵏ→ℝᵏ$ is an arbitrary
    bottleneck module, and $ϕᵤ$ is an elementwise post-activation.

    Since there is no pre-activation, the inverse can be solved in bottleneck
    coordinates:

    .. math:: z = Vᵀy - Vᵀϕᵤ(U f(z))

    and then lifted back with $x = y - ϕᵤ(U f(z))$.

    The Jacobian determinant is computed exactly with the matrix determinant
    lemma:

    .. math:: \log|det(𝕀ₙ + Dᵤ U Dₛ Vᵀ)| = \log|det(𝕀ₖ + Vᵀ Dᵤ U Dₛ)|.
    """

    input_size: Final[int]
    hidden_size: Final[int]
    use_bias: Final[bool]
    U: nn.Linear
    V: nn.Linear
    bottleneck: B
    activation: nn.Module
    eye: Tensor

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        *,
        bottleneck: B,  # (...k) -> (...k)
        activation: str | nn.Module = "Identity",
        use_bias: bool = True,
        maxiter: int = 256,
        atol: float = 1e-6,
        rtol: float = 1e-6,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__(maxiter=maxiter, atol=atol, rtol=rtol)
        if not 1 <= hidden_size <= input_size:
            raise ValueError("hidden_size must be between 1 and input_size")

        match activation:
            case str(name):
                activation_module = get_nonlinear_contraction(name)
            case _:
                activation_module = activation

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.use_bias = use_bias
        self.bottleneck = bottleneck
        self.activation = activation_module
        self.U = nn.Linear(
            hidden_size, input_size, bias=use_bias, device=device, dtype=dtype
        )
        self.V = nn.Linear(
            input_size, hidden_size, bias=use_bias, device=device, dtype=dtype
        )
        self.register_buffer(
            "eye",
            torch.eye(hidden_size, device=device, dtype=dtype),
            persistent=True,
        )

        self.reset_parameters()
        register_parametrization(self.U, "weight", OrthogonalHouseholder())
        register_parametrization(self.V, "weight", OrthogonalHouseholder(mode="rows"))

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.U.weight, a=sqrt(5))
        nn.init.kaiming_uniform_(self.V.weight, a=sqrt(5))
        if self.U.bias is not None:
            bound = 1 / sqrt(self.hidden_size) if self.hidden_size > 0 else 0.0
            nn.init.uniform_(self.U.bias, -bound, bound)
        if self.V.bias is not None:
            bound = 1 / sqrt(self.input_size) if self.input_size > 0 else 0.0
            nn.init.uniform_(self.V.bias, -bound, bound)
        update_parametrizations(self)

    def lift(self, z: Tensor, /) -> Tensor:
        r"""Compute $ϕᵤ(Uϕₛ(z))$."""
        return self.activation(self.U(self.bottleneck(z)))

    def contraction(self, x: Tensor, /) -> Tensor:
        return self.lift(self.V(x))

    def latent_map(self, z: Tensor, /) -> Tensor:
        r"""Compute $z + Vᵀϕᵤ(Uϕₛ(z))$ in bottleneck space."""
        return z + linear(self.lift(z), self.V.weight)

    def encode_and_logabsdet(self, x: Tensor, /) -> tuple[Tensor, Tensor]:
        z = self.V(x)
        u = self.lift(z)

        # log|det(𝕀ₙ + 𝐃(Vᵀ lift)(z))| = log|det 𝐃(z + Vᵀ lift(z))|
        # materialize the low-rank jacobian of the latent map with vjp
        _, vjp_fn, *_ = vjp(self.latent_map, z)
        batched_vjp_fn = vmap(lambda w: vjp_fn(w)[0], -1, -1)
        matrix = batched_vjp_fn(self.eye.expand(*z.shape, self.hidden_size))
        _, logabsdet = slogdet(matrix)
        return x + u, logabsdet

    def decode(self, y: Tensor, /) -> Tensor:
        # solve z = Vy + b - Vϕᵤ(Uϕₛ(z)),  z = Vx + b
        vty = self.V(y)
        z_star = fixpoint_solve(
            lambda z: vty - (self.latent_map(z) - z),  # type: ignore[misc]
            vty.clone(),
            maxiter=self.maxiter,
            atol=self.atol,
            rtol=self.rtol,
        )
        # x⁎ = y - ϕᵤ(Uϕₛ(z⁎))
        return y - self.lift(z_star)


class ReZeroBottleneck[B: nn.Module](ResidualBottleneck[ReZero[B]]):
    r"""Residual bottleneck with a learnable ReZero-scaled bottleneck module.

    .. math:: y = x + ϕᵤ(U (φ(ε)⋅f)(Vx))

    where $φ(ε) ∈ (-1, 1)$ and $φ(0)=0$, so the bottleneck branch is
    initialized at zero.
    """

    bottleneck: ReZero[B]
    scalar: Tensor
    scalar_map: nn.Module

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        *,
        bottleneck: B,
        scalar_map: nn.Module | str | None = "smooth-softsign",
        activation: str | nn.Module = "Identity",
        use_bias: bool = True,
        maxiter: int = 256,
        atol: float = 1e-6,
        rtol: float = 1e-6,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
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

        super().__init__(
            input_size=input_size,
            hidden_size=hidden_size,
            bottleneck=ReZero(bottleneck, scalar_map=scalar_map_module),
            activation=activation,
            use_bias=use_bias,
            maxiter=maxiter,
            atol=atol,
            rtol=rtol,
            device=device,
            dtype=dtype,
        )
        self.bottleneck: ReZero[B]
        self.scalar = self.bottleneck.scalar
        self.scalar_map = self.bottleneck.scalar_map
