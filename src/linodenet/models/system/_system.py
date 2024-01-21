r"""Models for the latent dynamical system."""

__all__ = [
    # Classes
    "LinODECell",
    "System",
    "SystemABC",
]

from abc import abstractmethod
from collections.abc import Callable, Iterable
from typing import Final, Protocol, runtime_checkable

import torch
from torch import Tensor, jit, nn

from linodenet.initializations import INITIALIZATIONS, Initialization
from linodenet.initializations.functional import gaussian
from linodenet.projections import FUNCTIONAL_PROJECTIONS
from linodenet.types import SelfMap
from linodenet.utils import deep_dict_update


@runtime_checkable
class System(Protocol):
    """Protocol for System Components."""

    def __call__(self, dt: Tensor, z: Tensor, /) -> Tensor:
        """Forward pass of the system.

        .. Signature: ``[∆t=(...,), x=(..., d)] -> (..., d)]``.
        """
        ...


class SystemABC(nn.Module):
    """Abstract Base Class for System components."""

    @abstractmethod
    def forward(self, dt: Tensor, z: Tensor, /) -> Tensor:
        r"""Forward pass of the system.

        Args:
            dt: The time-step to advance the system.
            z: The state estimate at time t.

        Returns:
            z': The updated state of the system at time t + ∆t.
        """


class LinODECell(nn.Module):
    r"""Linear System module, solves $ẋ = Ax$, i.e. $x_{t+∆t} = e^{A{∆t}}x_t$.

    .. Signature:: ``[∆t=(...,), x=(..., d)] -> (..., d)]``.

    By default, the Cell is parametrized by

    .. math:: e^{γ⋅A⋅∆t}x
    """

    # TODO: Use proper parametrization

    HP = {
        "__name__": __qualname__,
        "__module__": __module__,
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
    """CONST: Whether the scalar is learnable or not."""

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
        """Initialize the Linear ODE Cell."""
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

        def kernel_initialization_dispatch():
            r"""Dispatch the kernel initialization."""
            match kernel_initialization:
                case None:
                    return lambda: gaussian(input_size)
                case str() as key:
                    assert key in INITIALIZATIONS, "Unknown initialization!"
                    _init = INITIALIZATIONS[key]
                    return lambda: _init(input_size)
                case Callable() as func:  # type: ignore[misc]
                    assert Tensor(func(input_size)).shape == (input_size, input_size)  # type: ignore[unreachable]
                    return lambda: Tensor(func(input_size))
                case Tensor() as tensor:
                    assert tensor.shape == (input_size, input_size), (
                        f"Kernel has bad shape! {tensor.shape} but should be"
                        f" {(input_size, input_size)}"
                    )
                    return lambda: tensor
                case Iterable() as iterable:
                    tensor = Tensor(iterable)
                    assert tensor.shape == (input_size, input_size), (
                        f"Kernel has bad shape! {tensor.shape} but should be"
                        f" {(input_size, input_size)}"
                    )
                    return lambda: tensor
                case _:
                    raise TypeError(f"{type(kernel_initialization)=} not supported!")

        # this looks funny, but it needs to be written that way to be compatible with torchscript
        def kernel_parametrization_dispatch():
            r"""Dispatch the kernel parametrization."""
            match kernel_parametrization:
                case None:
                    return FUNCTIONAL_PROJECTIONS["identity"]
                case str() as key:
                    assert key in FUNCTIONAL_PROJECTIONS
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
