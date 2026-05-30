r"""Linear ODE module, to be used analogously to `scipy.integrate.odeint`."""

__all__ = ["LinearFlow"]

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

from .base import ContinuousFlowBase


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
