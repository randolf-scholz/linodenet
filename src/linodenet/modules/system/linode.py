r"""Linear ODE module, to be used analogously to `scipy.integrate.odeint`."""

__all__ = [
    # Classes
    "LinODECell",
    "LinODE",
]

from collections.abc import Callable, Iterable
from typing import Any, Final

import torch
from torch import Tensor, jit, nn

from linodenet.initializations import INITIALIZATIONS, Initialization, gaussian
from linodenet.projections import FUNCTIONAL_PROJECTIONS, Projection
from linodenet.types import SelfMap
from linodenet.utils import deep_dict_update, initialize_from_dict


class LinODECell(nn.Module):
    r"""Linear System module, solves $ẋ = Ax$, i.e. $x_{t+∆t} = e^{A{∆t}}x_t$.

    .. Signature:: ``[∆t=(...,), x=(..., d)] -> (..., d)]``.

    By default, the Cell is parametrized by

    .. math:: e^{γ⋅A⋅∆t}x
    """

    # TODO: Use proper parametrization

    HP = {
        "__name__": __qualname__,
        "__module__": __name__,
        "input_size": None,
        "kernel_initialization": None,
        "kernel_parametrization": None,
        "scalar": 0.0,
        "scalar_learnable": True,
    }

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

    def __init__(
        self,
        input_size: int,
        *,
        kernel_initialization: None | str | Tensor | Initialization = "skew-symmetric",
        kernel_parametrization: None | str | SelfMap[Tensor] | nn.Module = None,
        scalar: float = 0.0,
        scalar_learnable: bool = True,
    ) -> None:
        r"""Initialize the Linear ODE Cell."""
        super().__init__()
        config = deep_dict_update(
            self.HP,
            {
                "input_size": input_size,
                "kernel_initialization": kernel_initialization,
                "kernel_parametrization": kernel_parametrization,
                "scalar": scalar,
                "scalar_learnable": scalar_learnable,
            },
        )

        kernel_initialization = config["kernel_initialization"]
        kernel_parametrization = config["kernel_parametrization"]
        scalar = config["scalar"]
        scalar_learnable = config["scalar_learnable"]
        del config

        def kernel_initialization_dispatch() -> Callable[[], Tensor]:
            r"""Dispatch the kernel initialization."""
            match kernel_initialization:
                case None:
                    return lambda: gaussian(input_size)
                case str(key):
                    _init = INITIALIZATIONS[key]
                    return lambda: _init(input_size)
                case Callable() as func:  # type: ignore[misc]
                    tensor = Tensor(func(input_size))
                    if tensor.shape != (input_size, input_size):
                        raise ValueError(
                            f"Kernel has bad shape! {tensor.shape} but should be"
                            f" {(input_size, input_size)}"
                        )
                    return lambda: Tensor(func(input_size))
                case Tensor() as tensor:
                    if tensor.shape != (input_size, input_size):
                        raise ValueError(
                            f"Kernel has bad shape! {tensor.shape} but should be"
                            f" {(input_size, input_size)}"
                        )
                    return lambda: tensor
                case Iterable() as iterable:
                    tensor = Tensor(iterable)
                    if tensor.shape != (input_size, input_size):
                        raise ValueError(
                            f"Kernel has bad shape! {tensor.shape} but should be"
                            f" {(input_size, input_size)}"
                        )
                    return lambda: tensor
                case _:
                    raise TypeError(f"{type(kernel_initialization)=} not supported!")

        # this looks funny, but it needs to be written that way to be compatible with torchscript
        def kernel_parametrization_dispatch() -> SelfMap[Tensor]:
            r"""Dispatch the kernel parametrization."""
            match kernel_parametrization:
                case None:
                    return FUNCTIONAL_PROJECTIONS["identity"]
                case str(key):
                    return FUNCTIONAL_PROJECTIONS[key]
                case Callable() as func:  # type: ignore[misc]
                    return func  # type: ignore[unreachable]
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
    def forward(self, dt: Tensor, x0: Tensor) -> Tensor:
        r"""Signature: ``[(...,), (..., d)] -> (..., d)``.

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

    HP = {
        "__name__": __qualname__,
        "__module__": __name__,
        "cell": LinODECell.HP,
        "kernel_initialization": None,
        "kernel_projection": None,
    }
    r"""Dictionary of hyperparameters."""

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

    def __init__(self, input_size: int, **cfg: Any) -> None:
        super().__init__()
        config = deep_dict_update(self.HP, cfg)

        config["cell"]["input_size"] = input_size

        self.input_size = input_size
        self.output_size = input_size
        self.cell: nn.Module = initialize_from_dict(config["cell"])

        # Buffers
        kernel = getattr(self.cell, "kernel", None)
        if not isinstance(kernel, Tensor):
            raise TypeError("The cell must have a kernel attribute!")
        self.register_buffer("kernel", self.cell.kernel, persistent=False)
        self.register_buffer("xhat", torch.tensor(()), persistent=False)

    @jit.export
    def forward(self, T: Tensor, x0: Tensor) -> Tensor:
        r""".. Signature:: ``[(..., N), (..., d)] -> (..., N, d)``.

        Returns the estimated true state of the system at the times $t∈T$.
        """
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
