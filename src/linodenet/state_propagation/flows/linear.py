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

from linodenet.initializations import INITIALIZATIONS
from linodenet.initializations.modules import Constant as ConstantInitialization
from linodenet.mappings import MATRIX_PROJECTIONS
from linodenet.nn import ReZero
from linodenet.nn.parametrize import register_parametrization
from linodenet.utils import resolve_name
from signatures import signature

from .base import ContinuousFlow, ContinuousFlowBase


class LinearFlow(ContinuousFlowBase):
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

    # Parameters
    weight: Tensor
    r"""PARAM: The learnable weight-matrix of the linear ODE component."""
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
        }

    def __init__(
        self,
        input_size: int,
        *,
        kernel_initialization: str | Tensor | nn.Module = "skew-symmetric",
        kernel_parametrization: Optional[str | nn.Module] = None,
        use_rezero: bool = True,
    ) -> None:
        r"""Initialize the Linear ODE Cell."""
        super().__init__(input_shape=(input_size,))

        # initialize constants
        self.input_size = input_size
        self.output_size = input_size
        self.use_rezero = use_rezero
        match kernel_initialization:
            case nn.Module() as initialization:
                self.kernel_initialization = initialization

            case Tensor() as tensor:
                if tensor.shape != (input_size, input_size):
                    raise ValueError(
                        f"Kernel has bad shape! {tensor.shape} but should be"
                        f" {(input_size, input_size)}"
                    )
                self.kernel_initialization = ConstantInitialization(tensor)

            case str(key):
                initialization_cls = resolve_name(INITIALIZATIONS, key)

                try:
                    initialization = initialization_cls(input_size)  # type: ignore[arg-count]
                except Exception as exc:
                    exc.add_note(
                        f"failed to initialize kernel_initialization {initialization_cls}"
                    )
                    raise

                assert isinstance(initialization, nn.Module)
                self.kernel_initialization = initialization

            case _:
                raise TypeError(
                    "kernel_initialization must be a string, tensor, or nn.Module, "
                    f"got {type(kernel_initialization)!r}."
                )

        # initialize parameters
        self.weight = nn.Parameter(self.kernel_initialization())

        # apply parametrization if given
        match kernel_parametrization:
            case None:
                self.kernel_parametrization = None

            case nn.Module() as parametrization:
                register_parametrization(self, "weight", parametrization)
                self.kernel_parametrization = parametrization

            case str(key):
                parametrization_cls = resolve_name(MATRIX_PROJECTIONS, key)

                try:
                    parametrization = parametrization_cls()
                except Exception as exc:
                    exc.add_note(
                        f"failed to initialize parametrization {parametrization_cls}"
                    )
                    raise

                assert isinstance(parametrization, nn.Module)
                register_parametrization(self, "weight", parametrization)
                self.kernel_parametrization = parametrization

            case _:
                raise TypeError(
                    "kernel_parametrization must be a string, nn.Module, or None, "
                    f"got {type(kernel_parametrization)!r}."
                )

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
        self.kernel = self.rezero(self.weight)
        Adt = torch.einsum("..., kl -> ...kl", timedeltas, self.kernel)
        expAdt = torch.linalg.matrix_exp(Adt)  # (*bs, n)
        return torch.einsum("...nkl, ...l -> ...nk", expAdt, x0)

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


@signature("[(d, d), (..., $n), (..., d)] -> (..., $n, d)")
def linear_flow(
    timedeltas: Tensor,
    x0: Tensor,
    kernel: Tensor,
    bias: Tensor | None = None,
) -> Tensor:
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
    else:
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
        phi1bt = expAdt[..., :n, -1]
        return torch.einsum("...nkl, ...l -> ...nk", expAdt, x0) + phi1bt


class LinearGaussianFlow(ContinuousFlow):
    r"""Implements the propagation of a Normal distribution under linear ODE/SDE.

    That is, $z₀∼𝓝(μ₀, Σ₀)$ is propagated under the linear ODE/SDE

    .. math:: dZₜ = AZₜdt + bdt + C dWₜ

    Then, at time $t$ the solution is $zₜ∼𝓝(μₜ, Σₜ)$, where

    .. math::
        μₜ &= eᴬᵗμ₀ + φ₁(At)bt
        Σₜ &= eᴬᵗΣ₀eᴬᵀᵗ + φ₂(At)CCᵀt
    """

    A: Tensor
    b: Tensor | None
    Q: Tensor  # C = √Q

    def forward(
        self, delta_t: Tensor, z_0: tuple[Tensor, Tensor]
    ) -> tuple[Tensor, Tensor]:
        mu_0, sigma_0 = z_0
        A = self.A
        b = self.b
        Q = self.Q
        zero = torch.zeros_like(A)
        n = A.shape[-1]

        if b is None:
            # [[A, Q], [0, -Aᵀ]]
            # -> [[F, C], [0, F⁻ᵀ]]
            M = torch.cat(
                [
                    torch.cat([A, Q], dim=-1),
                    torch.cat([zero, -A.mT], dim=-1),
                ],
                dim=0,
            )

        else:
            # use augmented block matrix
            # [[A, b, Q], [0, 0, 0], [0, 0, -Aᵀ]]
            # -> [[F, r, C], [0, 1, 0], [0, 0, F⁻ᵀ]]
            b = b.unsqueeze(-1)
            zero_vec = torch.zeros_like(b)
            M = torch.cat(
                [
                    torch.cat([A, b, Q], dim=-1),
                    torch.zeros((1, 2 * n + 1), dtype=A.dtype, device=A.device),
                    torch.cat([zero, zero_vec, -A.mT], dim=-1),
                ],
                dim=0,
            )

        # exp(M∆t) is a block matrix
        P = torch.linalg.matrix_exp(M * delta_t)
        F = P[..., :n, :n]
        C = P[..., :n, -n:]
        r = P[..., :n, n] if b is not None else zero_vec

        mu_t = F @ mu_0 + r
        sigma_t = F @ sigma_0 @ F.mT + C @ F.mT

        return mu_t, sigma_t


def linear_gaussian_flow(
    A: Tensor,
    b: Tensor | None,
    Q: Tensor,
    delta_t: Tensor,
    z_0: tuple[Tensor, Tensor],
) -> tuple[Tensor, Tensor]:
    r"""Propagate a linear-gaussian system.

    .. math:: dZₜ = AZₜdt + bdt + C dWₜ

    Given Z₀∼𝓝(μ₀, Σ₀), then Zₜ∼𝓝(μₜ, Σₜ) for all $t$.
    """
    mu_0, sigma_0 = z_0
    zero = torch.zeros_like(A)
    n = A.shape[-1]

    if b is None:
        # [[A, Q], [0, -Aᵀ]]
        # -> [[F, C], [0, F⁻ᵀ]]
        M = torch.cat(
            [
                torch.cat([A, Q], dim=-1),
                torch.cat([zero, -A.mT], dim=-1),
            ],
            dim=0,
        )

    else:
        # use augmented block matrix
        # [[A, b, Q], [0, 0, 0], [0, 0, -Aᵀ]]
        # -> [[F, r, C], [0, 1, 0], [0, 0, F⁻ᵀ]]
        b = b.unsqueeze(-1)
        zero_vec = torch.zeros_like(b)
        M = torch.cat(
            [
                torch.cat([A, b, Q], dim=-1),
                torch.zeros((1, 2 * n + 1), dtype=A.dtype, device=A.device),
                torch.cat([zero, zero_vec, -A.mT], dim=-1),
            ],
            dim=0,
        )

    # exp(M∆t) is a block matrix
    P = torch.linalg.matrix_exp(M * delta_t)
    F = P[..., :n, :n]
    C = P[..., :n, -n:]
    r = P[..., :n, n] if b is not None else zero_vec

    mu_t = F @ mu_0 + r
    sigma_t = F @ sigma_0 @ F.mT + C @ F.mT

    return mu_t, sigma_t
