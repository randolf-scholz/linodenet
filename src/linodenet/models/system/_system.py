r"""Models for the latent dynamical system."""

__all__ = [
    # Classes
    "LinODECell",
]

from typing import Any, Final

import torch
from torch import Tensor, jit, nn

from linodenet.initializations import FunctionalInitializations
from linodenet.initializations.functional import gaussian
from linodenet.projections import PROJECTIONS
from linodenet.util import ReZeroCell, deep_dict_update


class LinODECell(nn.Module):
    r"""Linear System module, solves $ẋ = Ax$, i.e. $x̂ = e^{A{∆t}}x$.

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

    HP = {
        "__name__": __qualname__,  # type: ignore[name-defined]
        "__doc__": __doc__,
        "__module__": __module__,  # type: ignore[name-defined]
        "input_size": int,
        "kernel_initialization": None,
        "kernel_parametrization": None,
        "scale": 1.0,
        "rezero": True,
    }

    # Constants
    input_size: Final[int]
    r"""CONST: The dimensionality of inputs."""
    output_size: Final[int]
    r"""CONST: The dimensionality of the outputs."""
    use_rezero: Final[bool]
    r"""CONST: Whether to use rezero."""

    # Parameters
    raw_kernel: Tensor
    r"""PARAM: The system matrix of the linear ODE component."""

    # Buffers
    scale: Tensor
    r"""BUFFER: static scaling applied to the kernel."""

    kernel: Tensor
    r"""BUFFER: The parametrized system matrix of the linear ODE component."""

    def __init__(
        self,
        input_size: int,
        # kernel_initialization: Optional[
        #     Union[str, Tensor, FunctionalInitialization]
        # ] = None,
        # kernel_parametrization: Optional[Union[str, Projection]] = None,
        **HP: Any,
    ):
        super().__init__()
        self.CFG = HP = deep_dict_update(self.HP, HP)

        self.input_size = input_size
        self.output_size = input_size
        kernel_init = HP["kernel_initialization"]
        kernel_parametrization = HP["kernel_parametrization"]

        def kernel_initialization_dispatch():
            r"""Dispatch the kernel initialization."""
            if kernel_init is None:
                return lambda: gaussian(input_size)
            if isinstance(kernel_init, str):
                assert (
                    kernel_init in FunctionalInitializations
                ), "Unknown initialization!"
                _init = FunctionalInitializations[kernel_init]
                return lambda: _init(input_size)
            if callable(kernel_init):
                assert Tensor(kernel_init(input_size)).shape == (
                    input_size,
                    input_size,
                )
                return lambda: Tensor(kernel_init(input_size))
            if isinstance(kernel_init, Tensor):
                tensor = kernel_init
                assert tensor.shape == (
                    input_size,
                    input_size,
                ), f"Kernel has bad shape! {tensor.shape} but should be {(input_size, input_size)}"
                return lambda: tensor

            tensor = Tensor(kernel_init)
            assert tensor.shape == (
                input_size,
                input_size,
            ), f"Kernel has bad shape! {tensor.shape} but should be {(input_size, input_size)}"
            return lambda: tensor

        # this looks funny, but it needs to be written that way to be compatible with torchscript
        def kernel_parametrization_dispatch():
            r"""Dispatch the kernel parametrization."""
            if kernel_parametrization is None:
                _kernel_regularization = PROJECTIONS["identity"]
            elif kernel_parametrization in PROJECTIONS:
                _kernel_regularization = PROJECTIONS[kernel_parametrization]
            elif callable(kernel_parametrization):
                _kernel_regularization = kernel_parametrization
            else:
                raise NotImplementedError(f"{kernel_parametrization=} unknown")
            return _kernel_regularization

        self.register_buffer("scale", torch.tensor(HP["scale"]), persistent=False)
        self._kernel_initialization = kernel_initialization_dispatch()
        self._kernel_parametrization = kernel_parametrization_dispatch()
        self.kernel = nn.Parameter(self._kernel_initialization() * self.scale)

        self.use_rezero = HP["rezero"]

        if self.use_rezero:
            self.rezero = ReZeroCell()

    def kernel_initialization(self) -> Tensor:
        r"""Draw an initial kernel matrix (random or static)."""
        return self._kernel_initialization()

    @jit.export
    def kernel_parametrization(self, w: Tensor) -> Tensor:
        r"""Parametrize the Kernel, e.g. by projecting onto skew-symmetric matrices."""
        return self._kernel_parametrization(w)

    @jit.export
    def forward(self, dt: Tensor, x0: Tensor) -> Tensor:
        r"""Signature: ``[(...,), (..., d)] -> (..., d)]``.

        Parameters
        ----------
        dt: Tensor, shape=(...,)
            The time difference $t_1 - t_0$ between $x_0$ and $x̂$.
        x0:  Tensor, shape=(...,DIM)
            Time observed value at $t_0$.

        Returns
        -------
        xhat:  Tensor, shape=(...,DIM)
            The predicted value at $t_1$
        """
        A = self.kernel_parametrization(self.kernel)
        if self.use_rezero:
            A = self.rezero(A)
        Adt = torch.einsum("kl, ... -> ...kl", A, dt)
        expAdt = torch.matrix_exp(Adt)
        xhat = torch.einsum("...kl, ...l -> ...k", expAdt, x0)
        return xhat
