r"""Models for the latent dynamical system."""

__all__ = [
    # Classes
    "LinODECell",
]

import logging
from typing import Final, Optional, Union

import torch
from torch import Tensor, jit, nn

from linodenet.initializations import (
    FunctionalInitialization,
    FunctionalInitializations,
)
from linodenet.initializations.functional import gaussian
from linodenet.projections import PROJECTIONS, Projection
from linodenet.util import autojit

__logger__ = logging.getLogger(__name__)


@autojit
class LinODECell(nn.Module):
    r"""Linear System module, solves `ẋ = Ax`, i.e. `x̂ = e^{A\Delta t}x`.

    Parameters
    ----------
    input_size: int
    kernel_initialization: Union[Tensor, Callable[int, Tensor]]

    Attributes
    ----------
    input_size:  int
        The dimensionality of the input space.
    output_size: int
        The dimensionality of the output space.
    kernel: Tensor
        The system matrix
    kernel_initialization: Callable[[], Tensor]
        Parameter-less function that draws a initial system matrix
    kernel_parametrization: Callable[[Tensor], Tensor]
        Parametrization for the kernel
    """

    # Constants
    input_size: Final[int]
    r"""CONST: The dimensionality of inputs."""
    output_size: Final[int]
    r"""CONST: The dimensionality of the outputs."""

    # Parameters
    kernel: Tensor
    r"""PARAM: The system matrix of the linear ODE component."""

    # Buffers
    scale: Tensor
    r"""BUFFER: static scaling applied to the kernel."""

    def __init__(
        self,
        input_size: int,
        *,
        kernel_initialization: Optional[
            Union[str, Tensor, FunctionalInitialization]
        ] = None,
        kernel_parametrization: Optional[Union[str, Projection]] = None,
        scale: float = 1.0,
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = input_size

        def kernel_initialization_dispatch():
            if kernel_initialization is None:
                return lambda: gaussian(input_size)
            if kernel_initialization in FunctionalInitializations:
                _init = FunctionalInitializations[kernel_initialization]
                return lambda: _init(input_size)
            if callable(kernel_initialization):
                assert Tensor(kernel_initialization(input_size)).shape == (
                    input_size,
                    input_size,
                )
                return lambda: Tensor(kernel_initialization(input_size))
            if isinstance(kernel_initialization, Tensor):
                assert kernel_initialization.shape == (input_size, input_size)
                return lambda: kernel_initialization
            assert Tensor(kernel_initialization).shape == (input_size, input_size)
            return lambda: Tensor(kernel_initialization)

        # this looks funny, but it needs to be written that way to be compatible with torchscript
        def kernel_parametrization_dispatch():
            if kernel_parametrization is None:
                _kernel_regularization = PROJECTIONS["identity"]
            elif kernel_parametrization in PROJECTIONS:
                _kernel_regularization = PROJECTIONS[kernel_parametrization]
            elif callable(kernel_parametrization):
                _kernel_regularization = kernel_parametrization
            else:
                raise NotImplementedError(f"{kernel_parametrization=} unknown")
            return _kernel_regularization

        self.register_buffer("scale", torch.tensor(scale), persistent=False)
        self._kernel_initialization = kernel_initialization_dispatch()
        self._kernel_parametrization = kernel_parametrization_dispatch()
        self.kernel = nn.Parameter(self._kernel_initialization() * self.scale)

    def kernel_initialization(self) -> Tensor:
        r"""Draw an initial kernel matrix (random or static)."""
        return self._kernel_initialization()

    @jit.export
    def kernel_parametrization(self, w: Tensor) -> Tensor:
        r"""Parametrize the Kernel, e.g. by projecting onto skew-symmetric matrices."""
        return self._kernel_parametrization(w)

    @jit.export
    def forward(self, dt: Tensor, x0: Tensor) -> Tensor:
        r"""Signature: `[...,]×[...,d] ⟶ [...,d]`.

        Parameters
        ----------
        dt: Tensor, shape=(...,)
            The time difference `t_1 - t_0` between `x_0` and `x̂`.
        x0:  Tensor, shape=(...,DIM)
            Time observed value at `t_0`

        Returns
        -------
        xhat:  Tensor, shape=(...,DIM)
            The predicted value at `t_1`
        """
        A = self.kernel_parametrization(self.kernel)
        Adt = torch.einsum("kl, ... -> ...kl", A, dt)
        expAdt = torch.matrix_exp(Adt)
        xhat = torch.einsum("...kl, ...l -> ...k", expAdt, x0)
        return xhat
