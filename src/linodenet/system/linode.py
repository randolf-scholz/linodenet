r"""Linear ODE module, to be used analogously to `scipy.integrate.odeint`."""

__all__ = [
    # Classes
    "LinODECell",
    "LinODE",
]

from collections.abc import Callable
from typing import Final, Optional

import torch
from torch import Tensor, jit, nn

from blueprint import ModelBlueprint, initialize
from linodenet.initializations import INITIALIZATIONS, Initialization
from linodenet.projections import FUNCTIONAL_PROJECTIONS, Projection
from linodenet.signatures import signature
from linodenet.types import SelfMap


class LinODECell(nn.Module):
    r"""Linear System module, solves $ẋ = Ax$, i.e. $x_{t+∆t} = e^{A{∆t}}x_t$.

    By default, the Cell is parametrized by

    .. math:: e^{γ⋅A⋅∆t}x
    """

    # TODO: Use proper parametrization

    # Constants
    input_size: Final[int]
    r"""CONST: The dimensionality of inputs."""
    output_size: Final[int]
    r"""CONST: The dimensionality of the outputs."""
    scalar_learnable: Final[bool]
    r"""CONST: Whether the scalar is learnable or not."""

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
                    init = INITIALIZATIONS[key]
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
                    return FUNCTIONAL_PROJECTIONS["identity"]
                case str(key):
                    return FUNCTIONAL_PROJECTIONS[key]
                case Callable() as func:
                    return func
                case _:
                    raise TypeError(f"{type(kernel_parametrization)=} not supported!")

        # initialize constants
        self.input_size = input_size
        self.output_size = input_size
        self._kernel_initialization = kernel_initialization_dispatch()
        self._kernel_parametrization = kernel_parametrization_dispatch()
        self.scalar_learnable = scalar_learnable

        # initialize parameters
        self.scalar = nn.Parameter(
            torch.tensor(scalar), requires_grad=self.scalar_learnable
        )
        self.weight = nn.Parameter(self._kernel_initialization())

        # initialize buffers
        # NOTE: do we need persistent=False?
        self.register_buffer("kernel", self.kernel_parametrization(self.weight))

    def kernel_initialization(self) -> Tensor:
        r"""Draw an initial kernel matrix (random or static)."""
        return self._kernel_initialization()

    @jit.export
    def kernel_parametrization(self, w: Tensor) -> Tensor:
        r"""Parametrize the Kernel, e.g. by projecting onto skew-symmetric matrices."""
        return self._kernel_parametrization(w)

    @jit.export
    @signature("[(...,), (..., d)] -> (..., d)")
    def forward(self, dt: Tensor, x0: Tensor) -> Tensor:
        r"""Propagate the linear ODE from time t₀ to t₁ = t₀ + ∆t.

        Args:
            dt: The time difference t₁ - t₀ between x₀ and x̂.
            x0: Time observed value at t₀.

        Returns:
            xhat: The predicted value at t₁
        """
        self.kernel = self.scalar * self.kernel_parametrization(self.weight)
        Adt = torch.einsum("..., kl -> ...kl", dt, self.kernel)
        expAdt = torch.linalg.matrix_exp(Adt)
        xhat = torch.einsum("...kl, ...l -> ...k", expAdt, x0)
        return xhat


class LinODE(nn.Module):
    r"""Linear ODE module, to be used analogously to `scipy.integrate.odeint`."""

    # Constants
    input_size: Final[int]
    r"""CONST: The dimensionality of inputs."""
    output_size: Final[int]
    r"""CONST: The dimensionality of the outputs."""

    # Buffers
    xhat: Tensor
    r"""BUFFER: The forward prediction."""

    # Parameters
    kernel: Tensor
    r"""PARAM: The system matrix of the linear ODE component."""

    # Functions
    kernel_initialization: Initialization
    r"""FUNC: Parameter-less function that draws a initial system matrix."""
    kernel_projection: Projection
    r"""FUNC: Regularization function for the kernel."""

    @property
    def config(self) -> dict:
        return {
            "input_size": self.input_size,
            "cell": self.cell,
        }

    _DEFAULT_CELL_BLUEPRINT = {
        "__name__": LinODECell.__name__,
        "__module__": LinODECell.__module__,
        "input_size": None,
        "kernel_initialization": None,
        "kernel_parametrization": None,
        "scalar": 0.0,
        "scalar_learnable": True,
    }

    def __init__(
        self,
        input_size: int,
        *,
        cell: nn.Module | ModelBlueprint = _DEFAULT_CELL_BLUEPRINT,
    ) -> None:
        super().__init__()
        if isinstance(cell, nn.Module):
            self.cell = cell
        else:
            cell["input_size"] = input_size
            self.cell = initialize(cell)

        self.input_size = input_size
        self.output_size = input_size

        # Buffers
        kernel = getattr(self.cell, "kernel", None)
        if not isinstance(kernel, Tensor):
            raise TypeError("The cell must have a kernel attribute!")

        self.register_buffer("kernel", kernel, persistent=False)
        self.register_buffer("xhat", torch.tensor(()), persistent=False)

    @jit.export
    @signature("[(..., $n), (..., d)] -> (..., $n, d)")
    def forward(self, T: Tensor, x0: Tensor) -> Tensor:
        r"""Returns the estimated true state of the system at the times $t∈T$."""
        DT = torch.moveaxis(torch.diff(T), -1, 0)
        X: list[Tensor] = [x0]

        # iterate over LEN, this works even when no BATCH dim present.
        for dt in DT:
            X.append(self.cell(dt, X[-1]))

        # shape: [LEN, ..., DIM]
        Xhat = torch.stack(X, dim=0)
        # shape: [..., LEN, DIM]
        self.xhat = torch.moveaxis(Xhat, 0, -2)

        return self.xhat
