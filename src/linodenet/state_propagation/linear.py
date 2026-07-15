r"""Linear ODE module, to be used analogously to `scipy.integrate.odeint`."""

__all__ = [
    "LinearFlow",
    "LinearGaussianFlow",
    "linear_gaussian_flow",
    "linear_flow",
]

from typing import Final, Optional

import torch
from torch import Tensor, nn

from linodenet.initializations import resolve_kernel_initialization
from linodenet.mappings import PositiveDefinite
from linodenet.nn import ReZero
from linodenet.nn.parametrize import register_parametrization
from linodenet.parametrizations import resolve_matrix_parametrization
from signatures import signature

from .base import ContinuousFlow


@signature("[(..., $n), (..., d), (d, d), (d,)?] -> (..., $n, d)")
def linear_flow(
    timedeltas: Tensor,  # (..., $n)
    x0: Tensor,  # (..., d)
    kernel: Tensor,  # (d, d)
    bias: Optional[Tensor] = None,
    /,
) -> Tensor:  # (..., $n, d)
    r"""Linear ODE.

    .. math:: dxₜ/dt = Axₜ + b

    Given x₀, then $xₜ = eᴬᵗx₀ + φ₁(At)bt$. Here, $φₖ(z) = ∑ₙ₌₀^∞ zⁿ/(n+k)!$ are the phi-functions,
    which can be computed from the matrix exponential of an augmented block matrix.
    In particular, φ₀(A) = eᴬ and φ₁(A) = (eᴬ - I)/A.
    """
    if bias is None:
        Adt = torch.einsum("..., kl -> ...kl", timedeltas, kernel)
        expAdt = torch.linalg.matrix_exp(Adt)  # (*bs, n)
        return torch.einsum("...nkl, ...l -> ...nk", expAdt, x0)

    # use augmented block matrix [[A, b], [0, 0]]
    n = bias.shape[-1]
    M = torch.cat(
        [
            torch.cat([kernel, bias.unsqueeze(-1)], dim=-1),
            torch.zeros((1, n + 1), device=kernel.device, dtype=kernel.dtype),
        ],
        dim=0,
    )
    Mdt = torch.einsum("..., kl -> ...kl", timedeltas, M)
    expMdt = torch.linalg.matrix_exp(Mdt)
    expAdt = expMdt[..., :n, :n]
    phi1bt = expMdt[..., :n, -1]
    return torch.einsum("...nkl, ...l -> ...nk", expAdt, x0) + phi1bt


class LinearFlow(nn.Module, ContinuousFlow):
    r"""Linear Flow, solves $ẋ = Ax$, i.e. $x_{t+∆t} = e^{A{∆t}}xₜ$.

    This is augmented by 2 techniques:

    1. parametrization of the kernel, e.g. restricting it to some subset of matrices,
       such as skew-symmetric matrices, which leads to stable dynamics.
    2. an optional ReZero gate applied to the kernel, which can be used to improve
       the learning dynamics.

    .. math:: e^{ρ(π(A))∆t}x
    """

    # Constants
    input_size: Final[int]
    r"""CONST: The dimensionality of inputs."""
    output_size: Final[int]
    r"""CONST: The dimensionality of the outputs."""
    use_rezero: Final[bool]
    r"""CONST: Whether the kernel is wrapped in ``ReZero``."""
    use_bias: Final[bool]
    r"""CONST: Whether the flow has a learnable affine bias."""

    # Parameters
    weight: Tensor
    r"""PARAM: The learnable weight-matrix of the linear ODE component."""
    bias: Tensor | None
    r"""PARAM: Optional learnable bias of the linear ODE component."""
    rezero: nn.Module
    r"""MODULE: Optional ReZero gate applied to the kernel."""
    # Buffers
    kernel: Tensor
    r"""BUFFER: The system matrix of the linear ODE component."""
    kernel_initialization: nn.Module
    r"""MODULE: Optional Initialization of the kernel."""
    kernel_parametrization: nn.Module | None
    r"""MODULE: Optional parametrization of the kernel."""

    @property
    def config(self) -> dict:
        return {
            "input_size": self.input_size,
            "kernel_initialization": self.kernel_initialization,
            "kernel_parametrization": self.kernel_parametrization,
            "use_rezero": self.use_rezero,
            "use_bias": self.use_bias,
        }

    def __init__(
        self,
        input_size: int,
        *,
        kernel_initialization: str | Tensor | nn.Module = "skew-symmetric",
        kernel_parametrization: Optional[str | nn.Module] = None,
        use_rezero: bool = True,
        use_bias: bool = False,
    ) -> None:
        r"""Initialize the Linear ODE Cell."""
        super().__init__()
        ContinuousFlow.__init__(self, input_shape=(input_size,))

        # initialize constants
        self.input_size = input_size
        self.output_size = input_size
        self.use_rezero = use_rezero
        self.use_bias = use_bias
        self.kernel_initialization = resolve_kernel_initialization(
            input_size, kernel_initialization
        )

        # initialize parameters
        self.weight = nn.Parameter(self.kernel_initialization())
        self.register_parameter(
            "bias",
            nn.Parameter(torch.zeros(input_size)) if self.use_bias else None,
        )

        # apply parametrization if given
        self.kernel_parametrization = resolve_matrix_parametrization(
            kernel_parametrization
        )
        register_parametrization(self, "weight", self.kernel_parametrization)

        # initialize buffers
        self.rezero = ReZero() if self.use_rezero else nn.Identity()
        self.register_buffer("kernel", self.rezero(self.weight))

    @signature("[(...), (..., d)] -> (..., d)")
    def step(self, timedeltas: Tensor, x0: Tensor) -> Tensor:
        r"""Propagate the linear ODE for a single time-delta.

        .. math:: step(∆t, x) = e^{ρ(π(A))∆t}x
        """
        return self.forward(timedeltas.unsqueeze(-1), x0).squeeze(-2)

    @signature("[(..., $n), (..., d)] -> (..., $n, d)")
    def forward(self, timedeltas: Tensor, x0: Tensor) -> Tensor:
        r"""Propagate the linear ODE for a sequence of time-deltas.

        .. math:: step(∆tₙ, x) = e^{ρ(π(A))∆tₙ}x
        """
        # update buffer
        self.kernel = self.rezero(self.weight)
        return linear_flow(timedeltas, x0, self.kernel, self.bias)

    @signature("[(..., $n), (..., d)] -> (..., $n, d)")
    def forecast(
        self,
        timestamps: Tensor,
        x0: Tensor,
        *,
        t0: Tensor | float,
    ) -> Tensor:
        r"""Propagate the linear ODE for a sequence of timestamps.

        .. math::
            ∆tₙ &= tₙ - t₀ \\
            step(∆tₙ, x) &= e^{ρ(π(A))∆tₙ}x
        """
        return self(timestamps - t0, x0)


def linear_gaussian_flow(
    delta_t: Tensor,  # (..., $n)
    z0: tuple[Tensor, Tensor],  # (..., d), (..., d, d)
    A: Tensor,  # (d, d)
    Q: Tensor,  # (d, d)
    b: Optional[Tensor] = None,  # (d,) | None
    /,
) -> tuple[Tensor, Tensor]:  # (...., $n, d), (..., $n, d, d)
    r"""Propagate a linear-gaussian system.

    .. math:: dZₜ = AZₜdt + bdt + C dWₜ

    Given Z₀∼𝓝(μ₀, Σ₀), then Zₜ∼𝓝(μₜ, Σₜ) for all $t$, with

    .. math::
        μₜ &= eᴬᵗμ₀ + φ₁(At)bt \\
        Σₜ &= eᴬᵗΣ₀eᴬᵀᵗ + ??

    Args:
        delta_t: The time-delta(s) to propagate for, of shape (..., $n).
        z0: initial state given as a pair of mean and covariance matrix,
            of shapes (..., d) and (..., d, d) respectively.
        A: The system matrix of the linear ODE component, of shape (d, d).
        Q: The diffusion matrix of the linear SDE component, of shape (d, d). Must be symmetric positive definite.
        b: Optional affine bias of the linear ODE component, of shape (d,). If None, then no bias is applied.
    """
    mu_0, sigma_0 = z0
    n = A.shape[-1]

    if b is None:
        # [[A, Q], [0, -Aᵀ]]
        # -> [[F, C], [0, F⁻ᵀ]]
        M = torch.cat(
            [
                torch.cat([A, Q], dim=-1),
                torch.cat([torch.zeros_like(A), -A.mT], dim=-1),
            ],
            dim=0,
        )

    else:
        # use augmented block matrix
        # [[A, Q, b], [0, -Aᵀ, 0], [0, 0, 0]]
        # -> [[F, C, r], [0, F⁻ᵀ, 0], [0, 0, 1]]
        b = b.unsqueeze(-1)
        M = torch.cat(
            [
                torch.cat([A, Q, b], dim=-1),
                torch.cat([torch.zeros_like(A), -A.mT, torch.zeros_like(b)], dim=-1),
                torch.zeros((1, 2 * n + 1), dtype=A.dtype, device=A.device),
            ],
            dim=0,
        )

    # exp(M∆t) is a block matrix
    Mdt = torch.einsum("..., kl -> ...kl", delta_t, M)
    P = torch.linalg.matrix_exp(Mdt)
    F = P[..., :n, :n]  # top left block
    C = P[..., :n, n : 2 * n]  # top center block
    r = P[..., :n, -1] if b is not None else 0.0  # top right block
    mu_t = torch.einsum("...nkl, ...l -> ...nk", F, mu_0) + r  # eᴬᵗμ₀ + φ₁(At)bt
    sigma_t = F @ sigma_0.unsqueeze(-3) @ F.mT + C @ F.mT  # eᴬᵗΣ₀eᴬᵀᵗ + CFᵀ

    return mu_t, sigma_t


class LinearGaussianFlow(nn.Module, ContinuousFlow):
    r"""Implements the propagation of a Normal distribution under linear ODE/SDE.

    That is, $z₀∼𝓝(μ₀, Σ₀)$ is propagated under the linear ODE/SDE

    .. math:: dZₜ = AZₜdt + bdt + C dWₜ

    Then, at time $t$ the solution is $zₜ∼𝓝(μₜ, Σₜ)$, where

    .. math::
        μₜ &= eᴬᵗμ₀ + φ₁(At)bt  \\
        Σₜ &= eᴬᵗΣ₀e^{Aᵀt} + ∫₀ᵗ eᴬ⁽ᵗ⁻ˢ⁾ Q e^{Aᵀ(t-s)} ds

    The last integral can be computed in closed form using a block matrix exponential.

    References:
        - | Computing integrals involving the matrix exponential
          | Van Loan
          | IEEE Transactions on Automatic Control, 1978
    """

    A: Tensor
    b: Tensor | None
    Q: Tensor  # C = √Q
    kernel_initialization: nn.Module
    r"""MODULE: Optional Initialization of the drift kernel."""
    kernel_parametrization: nn.Module | None
    r"""MODULE: Optional parametrization of the drift kernel."""

    @property
    def config(self) -> dict:
        return {
            "input_size": self.input_size,
            "kernel_initialization": self.kernel_initialization,
            "kernel_parametrization": self.kernel_parametrization,
        }

    def __init__(
        self,
        input_size: int,
        *,
        kernel_initialization: str | Tensor | nn.Module = "skew-symmetric",
        kernel_parametrization: Optional[str | nn.Module] = None,
    ) -> None:
        super().__init__()
        ContinuousFlow.__init__(self, input_shape=(input_size,))

        self.input_size = input_size
        self.kernel_initialization = resolve_kernel_initialization(
            input_size, kernel_initialization
        )

        self.A = nn.Parameter(self.kernel_initialization())
        self.kernel_parametrization = resolve_matrix_parametrization(
            kernel_parametrization
        )
        register_parametrization(self, "A", self.kernel_parametrization)
        self.Q = nn.Parameter(torch.randn(input_size, input_size))
        self.b = nn.Parameter(torch.randn(input_size))
        register_parametrization(self, "Q", PositiveDefinite())

    def forward(
        self, delta_t: Tensor, z_0: tuple[Tensor, Tensor]
    ) -> tuple[Tensor, Tensor]:
        r"""Propagate the linear-Gaussian system for one or more time-deltas."""
        return linear_gaussian_flow(delta_t, z_0, self.A, self.Q, self.b)
