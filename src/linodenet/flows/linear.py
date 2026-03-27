r"""Linear ODE module, to be used analogously to `scipy.integrate.odeint`."""

__all__ = ["LinearFlow"]

from collections.abc import Callable
from typing import Final, Optional

import torch
from torch import Tensor, nn

from linodenet.initializations import INITIALIZATION_FNS, Initialization
from linodenet.mappings import MATRIX_PROJECTION_FNS
from linodenet.mappings.functional import identity as identity_map
from linodenet.types import SelfMap
from signatures import signature

from .continuous import ContinuousFlowBase


class LinearFlow(ContinuousFlowBase):
    r"""Linear Flow, solves $ẋ = Ax$, i.e. $x_{t+∆t} = e^{A{∆t}}xₜ$.

    This is augmented by 2 techniques:

    1. parametrization of the kernel, e.g. restricting it to some subset of matrices,
       such as skew-symmetric matrices, which leads to stable dynamics.
    2. a learnable scalar applied to the kernel, which can be used to improve
       the learning dynamics.

    .. math:: e^{ε⋅π(A)∆t}x
    """

    # Constants
    input_size: Final[int]
    r"""CONST: The dimensionality of inputs."""
    output_size: Final[int]
    r"""CONST: The dimensionality of the outputs."""
    scalar_learnable: Final[bool]
    r"""CONST: Whether the scalar is learnable or not."""
    input_shape: Final[tuple[int]]
    r"""CONST: The shape of the input state."""

    # Parameters
    scalar: Tensor
    r"""PARAM: the scalar applied to the kernel."""
    weight: Tensor
    r"""PARAM: The learnable weight-matrix of the linear ODE component."""
    # Buffers
    kernel: Tensor
    r"""BUFFER: The system matrix of the linear ODE component."""

    @property
    def config(self) -> dict:
        return {
            "input_size": self.input_size,
            "kernel_initialization": self.kernel_initialization_spec,
            "kernel_parametrization": self.kernel_parametrization_spec,
            "scalar": self.scalar_init,
            "scalar_learnable": self.scalar_learnable,
        }

    def __init__(
        self,
        input_size: int,
        *,
        kernel_initialization: str | Tensor | Initialization = "skew-symmetric",
        kernel_parametrization: Optional[str | SelfMap[Tensor] | nn.Module] = None,
        scalar: float = 0.0,
        scalar_learnable: bool = True,
    ) -> None:
        r"""Initialize the Linear ODE Cell."""
        super().__init__()
        self.kernel_initialization_spec = kernel_initialization
        self.kernel_parametrization_spec = kernel_parametrization
        self.scalar_init = scalar

        def kernel_initialization_dispatch() -> Callable[[], Tensor]:
            r"""Dispatch the kernel initialization."""
            match kernel_initialization:
                case str(key):
                    init = INITIALIZATION_FNS[key.replace("-", "_")]
                    return lambda: init(input_size)
                case Callable() as func:
                    tensor = Tensor(func(input_size))
                    if tensor.shape != (input_size, input_size):
                        raise ValueError(
                            f"Kernel has bad shape! {tensor.shape} but should be"
                            f" {(input_size, input_size)}"
                        )
                    return lambda: Tensor(func(input_size))
                case other:
                    try:
                        tensor = torch.as_tensor(other)
                    except Exception as e:
                        raise TypeError(
                            f"Cannot convert {other} to a tensor for kernel initialization!"
                        ) from e
                    if tensor.shape != (input_size, input_size):
                        raise ValueError(
                            f"Kernel has bad shape! {tensor.shape} but should be"
                            f" {(input_size, input_size)}"
                        )
                    return lambda: tensor

        # this looks funny, but it needs to be written that way to be compatible with torchscript
        def kernel_parametrization_dispatch() -> SelfMap[Tensor]:
            r"""Dispatch the kernel parametrization."""
            match kernel_parametrization:
                case None:
                    return identity_map
                case str(key):
                    return MATRIX_PROJECTION_FNS[key]
                case Callable() as func:
                    return func
                case _:
                    raise TypeError(f"{type(kernel_parametrization)=} not supported!")

        # initialize constants
        self.input_size = input_size
        self.output_size = input_size
        self.scalar_learnable = scalar_learnable
        self.input_shape = (input_size,)
        self._kernel_initialization = kernel_initialization_dispatch()
        self._kernel_parametrization = kernel_parametrization_dispatch()

        # initialize parameters
        self.scalar = nn.Parameter(
            torch.tensor(scalar),
            requires_grad=self.scalar_learnable,
        )
        self.weight = nn.Parameter(self._kernel_initialization())

        # initialize buffers
        # NOTE: do we need persistent=False?
        self.register_buffer("kernel", self.kernel_parametrization(self.weight))

    def kernel_initialization(self) -> Tensor:
        r"""Draw an initial kernel matrix (random or static)."""
        return self._kernel_initialization()

    def kernel_parametrization(self, w: Tensor) -> Tensor:
        r"""Parametrize the Kernel, e.g. by projecting onto skew-symmetric matrices."""
        return self._kernel_parametrization(w)

    @signature("[(...), (..., d)] -> (..., d)")
    def step(self, timedeltas: Tensor, x0: Tensor) -> Tensor:
        r"""Propagate the linear ODE for a single time-delta.

        .. math:: step(∆t, x) = e^{ε⋅π(A)∆t}x
        """
        return self.forward(timedeltas.unsqueeze(-1), x0).squeeze(-2)

    @signature("[(..., $n), (..., d)] -> (..., $n, d)")
    def forward(self, timedeltas: Tensor, x0: Tensor) -> Tensor:
        r"""Propagate the linear ODE for a sequence of time-deltas.

        .. math:: step(∆tₙ, x) = e^{ε⋅π(A)∆tₙ}x
        """
        self.kernel = self.scalar * self.kernel_parametrization(self.weight)
        Adt = torch.einsum("...n, kl -> ...nkl", timedeltas, self.kernel)
        expAdt = torch.linalg.matrix_exp(Adt)  # (*bs, n)
        xhat = torch.einsum("...nkl, ...l -> ...nk", expAdt, x0)
        return xhat

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
            step(∆tₙ, x) &= e^{ε⋅π(A)∆tₙ}x
        """
        return self(timestamps - t0, x0)
