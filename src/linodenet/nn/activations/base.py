r"""Base classes for activations."""

__all__ = [
    # ABCs & Protocols
    "Activation",
    "GenericActivation",
    "ActivationBase",
    # functions
    "get_activation",
]

from abc import abstractmethod
from collections.abc import Callable
from typing import Protocol, overload, runtime_checkable

from torch import Tensor, nn

from blueprint import Makes, initialize
from signatures import signature

from .crelu import CReLU, crelu
from .imported import (
    BimodalToGaussian,
    GaussianToBimodal,
    GaussianToMixture,
    MixtureToGaussian,
    bimodal_to_gaussian,
    gaussian_to_bimodal,
    hard_bend,
)

type GenericActivation = Callable[..., Tensor | tuple[Tensor, ...]]
r"""Type alias for generic activation functions (may require additional args!)."""


@runtime_checkable
class Activation(Protocol):
    r"""Protocol for activation functions.

    We define (element-wise) activations as callables that take a single tensor input
    and returns a tensor of the same shape.
    """

    @abstractmethod
    @signature("(..., *xs) -> (..., *xs)")
    def __call__(self, x: Tensor, /) -> Tensor: ...


class ActivationBase(nn.Module):
    r"""Abstract Base Class for Activation components."""

    @abstractmethod
    @signature("(..., *xs) -> (..., *xs)")
    def forward(self, x: Tensor, /) -> Tensor:
        r"""Forward pass of the activation.

        Args:
            x: The input tensor to be activated.

        Returns:
            y: The activated tensor.
        """
        ...


@overload
def get_activation[T: nn.Module](arg: Makes[T], /, **cfg: object) -> T: ...
@overload
def get_activation(arg: str | dict, /, **cfg: object) -> nn.Module: ...
def get_activation(arg: object, /, **cfg: object) -> nn.Module:
    r"""Get an activation function by name.

    Args:
        arg: The activation to retrieve. Can be one of the following:
            - A string name of an activation function or class.
            - A dictionary (Blueprint) with instructions for initializing an activation function or class.
            - An instance of an activation function or class.
            - A class of an activation function or class.
        **cfg: Additional keyword arguments to pass to the activation function or class when initializing.
    """
    match arg:
        # if a name, look up in the dictionary
        case str(name):
            # avoid circular import

            match ACTIVATIONS.get(name):
                case None:
                    pass
                case cls:
                    return get_activation(cls, **cfg)

            raise LookupError(f"Unknown activation function: {name!r}")

        # if a class, try to instantiate it with the given configuration
        case type() as cls:
            try:
                return cls(**cfg)
            except TypeError as exc:
                exc.add_note(f"Failed to instantiate {cls} with arguments {cfg!r}")
                raise

        # if a config, use the blueprint system to initialize it
        case dict(spec):
            result = initialize(spec)
            assert isinstance(result, nn.Module)
            return result

        # if an instance, return as-is
        case nn.Module() as instance:
            if cfg:
                raise ValueError(f"Cannot pass arguments to an instance: {cfg!r}")
            return instance

        case _:
            raise TypeError(f"Invalid argument: {arg!r}")


ACTIVATIONS: dict[str, type[nn.Module]] = {
    "CReLU": CReLU,
    "GaussianToBimodal": GaussianToBimodal,
    "BimodalToGaussian": BimodalToGaussian,
    "GaussianToMixture": GaussianToMixture,
    "MixtureToGaussian": MixtureToGaussian,
    # torch imports
    "CELU"        : nn.CELU,
    "ELU"         : nn.ELU,
    "GELU"        : nn.GELU,
    "GLU"         : nn.GLU,
    "Hardshrink"  : nn.Hardshrink,
    "Hardsigmoid" : nn.Hardsigmoid,
    "Hardswish"   : nn.Hardswish,
    "Hardtanh"    : nn.Hardtanh,
    "Identity"    : nn.Identity,
    "LeakyReLU"   : nn.LeakyReLU,
    "LogSigmoid"  : nn.LogSigmoid,
    "Mish"        : nn.Mish,
    "PReLU"       : nn.PReLU,
    "RReLU"       : nn.RReLU,
    "ReLU"        : nn.ReLU,
    "ReLU6"       : nn.ReLU6,
    "SELU"        : nn.SELU,
    "SiLU"        : nn.SiLU,
    "Sigmoid"     : nn.Sigmoid,
    "Softplus"    : nn.Softplus,
    "Softshrink"  : nn.Softshrink,
    "Softsign"    : nn.Softsign,
    "Tanh"        : nn.Tanh,
    "Tanhshrink"  : nn.Tanhshrink,
}  # fmt: skip
r"""Dictionary containing all available activation classes."""


ACTIVATION_FNS: dict[str, Activation] = {
    "hard_bend": hard_bend,
    "gaussian_to_bimodal": gaussian_to_bimodal,
    "bimodal_to_gaussian": bimodal_to_gaussian,
    # torch imports
    "celu"        : nn.functional.celu,
    "elu"         : nn.functional.elu,
    "gelu"        : nn.functional.gelu,
    "hardshrink"  : nn.functional.hardshrink,
    "hardsigmoid" : nn.functional.hardsigmoid,
    "hardswish"   : nn.functional.hardswish,
    "hardtanh"    : nn.functional.hardtanh,
    "leaky_relu"  : nn.functional.leaky_relu,
    "log_sigmoid" : nn.functional.logsigmoid,
    "mish"        : nn.functional.mish,
    "relu"        : nn.functional.relu,
    "relu6"       : nn.functional.relu6,
    "rrelu"       : nn.functional.rrelu,
    "selu"        : nn.functional.selu,
    "sigmoid"     : nn.functional.sigmoid,
    "silu"        : nn.functional.silu,
    "softplus"    : nn.functional.softplus,
    "softshrink"  : nn.functional.softshrink,
    "softsign"    : nn.functional.softsign,
    "tanh"        : nn.functional.tanh,
    "tanhshrink"  : nn.functional.tanhshrink,
}  # fmt: skip
r"""Dictionary containing all available activation functions."""

ACTIVATION_FNS_WITH_ARGS: dict[str, GenericActivation] = {
    "crelu": crelu,
}  # fmt: skip
r"""Activations that do not match the usual signature of activations."""
