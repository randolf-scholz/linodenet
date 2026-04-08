r"""Base classes for activations."""

__all__ = [
    "ACTIVATIONS",
    "ACTIVATION_FNS",
    "ACTIVATION_FNS_WITH_ARGS",
    # ABCs & Protocols
    "Activation",
    "Activations",
    "GenericActivation",
    "ActivationBase",
]

import re
from abc import abstractmethod
from collections.abc import Callable
from enum import StrEnum
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
    gaussian_to_mixture,
    hard_bend,
    mixture_to_gaussian,
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


def _to_kebab_case(value: str, /) -> str:
    r"""Normalize a name to lowercase kebab-case."""
    normalized = re.sub(r"_", "-", value.strip())
    normalized = re.sub(r"([a-z0-9])([A-Z][a-z])", r"\1-\2", normalized)
    return normalized.lower()


class Activations(StrEnum):
    r"""Enum of the provided activation modules."""

    CRELU = "crelu"
    GAUSSIAN_TO_BIMODAL = "gaussian-to-bimodal"
    BIMODAL_TO_GAUSSIAN = "bimodal-to-gaussian"
    GAUSSIAN_TO_MIXTURE = "gaussian-to-mixture"
    MIXTURE_TO_GAUSSIAN = "mixture-to-gaussian"
    CELU = "celu"
    ELU = "elu"
    GELU = "gelu"
    GLU = "glu"
    HARDSHRINK = "hardshrink"
    HARDSIGMOID = "hardsigmoid"
    HARDSWISH = "hardswish"
    HARDTANH = "hardtanh"
    IDENTITY = "identity"
    LEAKY_RELU = "leaky-relu"
    LOG_SIGMOID = "log-sigmoid"
    MISH = "mish"
    PRELU = "prelu"
    RRELU = "rrelu"
    RELU = "relu"
    RELU6 = "relu6"
    SELU = "selu"
    SILU = "silu"
    SIGMOID = "sigmoid"
    SOFTPLUS = "softplus"
    SOFTSHRINK = "softshrink"
    SOFTSIGN = "softsign"
    TANH = "tanh"
    TANHSHRINK = "tanhshrink"

    @classmethod
    def _missing_(cls, value: object) -> Activations | None:
        if not isinstance(value, str):
            return None
        normalized = _to_kebab_case(value)

        if activation := cls.__members__.get(normalized):
            return activation

        for member in cls:
            if _to_kebab_case(ACTIVATIONS[member].__name__) == normalized:
                return member
        return None

    @overload
    @classmethod
    def new[T: nn.Module](cls, arg: Makes[T], /, **cfg: object) -> T: ...
    @overload
    @classmethod
    def new(cls, arg: str, /, **cfg: object) -> nn.Module: ...
    @classmethod
    def new(cls, arg: object, /, **cfg: object) -> nn.Module:
        r"""Instantiate an activation module from a name, class, config, or instance."""
        match arg:
            case Activations() as member:
                return cls.new(ACTIVATIONS[member], **cfg)

            case str(name):
                try:
                    activation = cls(name)
                except ValueError as exc:
                    normalized = _to_kebab_case(name)
                    raise LookupError(
                        f"Unknown activation function: {name!r} (normalized: {normalized!r})"
                    ) from exc
                return cls.new(activation, **cfg)

            case type() as module_cls:
                try:
                    return module_cls(**cfg)
                except TypeError as exc:
                    exc.add_note(
                        f"Failed to instantiate {module_cls} with arguments {cfg!r}"
                    )
                    raise

            case dict(spec):
                result = initialize(spec)
                assert isinstance(result, nn.Module)
                return result

            case nn.Module() as instance:
                if cfg:
                    raise ValueError(f"Cannot pass arguments to an instance: {cfg!r}")
                return instance

            case _:
                raise TypeError(f"Invalid argument: {arg!r}")


ACTIVATIONS: dict[Activations, type[nn.Module]] = {
    Activations.CRELU: CReLU,
    Activations.GAUSSIAN_TO_BIMODAL: GaussianToBimodal,
    Activations.BIMODAL_TO_GAUSSIAN: BimodalToGaussian,
    Activations.GAUSSIAN_TO_MIXTURE: GaussianToMixture,
    Activations.MIXTURE_TO_GAUSSIAN: MixtureToGaussian,
    # torch imports
    Activations.CELU: nn.CELU,
    Activations.ELU: nn.ELU,
    Activations.GELU: nn.GELU,
    Activations.GLU: nn.GLU,
    Activations.HARDSHRINK: nn.Hardshrink,
    Activations.HARDSIGMOID: nn.Hardsigmoid,
    Activations.HARDSWISH: nn.Hardswish,
    Activations.HARDTANH: nn.Hardtanh,
    Activations.IDENTITY: nn.Identity,
    Activations.LEAKY_RELU: nn.LeakyReLU,
    Activations.LOG_SIGMOID: nn.LogSigmoid,
    Activations.MISH: nn.Mish,
    Activations.PRELU: nn.PReLU,
    Activations.RRELU: nn.RReLU,
    Activations.RELU: nn.ReLU,
    Activations.RELU6: nn.ReLU6,
    Activations.SELU: nn.SELU,
    Activations.SILU: nn.SiLU,
    Activations.SIGMOID: nn.Sigmoid,
    Activations.SOFTPLUS: nn.Softplus,
    Activations.SOFTSHRINK: nn.Softshrink,
    Activations.SOFTSIGN: nn.Softsign,
    Activations.TANH: nn.Tanh,
    Activations.TANHSHRINK: nn.Tanhshrink,
}  # fmt: skip
r"""Dictionary mapping activation enum values to module classes."""


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
    "mixture_to_gaussian": mixture_to_gaussian,
    "gaussian_to_mixture": gaussian_to_mixture,
}  # fmt: skip
r"""Activations that do not match the usual signature of activations."""
