r"""Base classes for activations."""

__all__ = [
    # ABCs & Protocols
    "Activation",
    "GenericActivation",
    "ActivationRequiresDim",
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


class ActivationRequiresDim(Protocol):
    r"""Protocol for activation functions that require a dimension argument."""

    @abstractmethod
    @signature("[(..., *xs), dim] -> (..., *xs)")
    def __call__(self, x: Tensor, /, *, dim: int | tuple[int, ...]) -> Tensor: ...


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
def get_activation[T: Activation](arg: Makes[T], /, **cfg: object) -> T: ...
@overload
def get_activation(arg: str | dict, /, **cfg: object) -> Activation: ...
def get_activation(arg: object, /, **cfg: object) -> Activation:
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
            from linodenet.nn.activations import ALL_ACTIVATIONS  # noqa: PLC0415

            try:
                obj = ALL_ACTIVATIONS[name]
            except KeyError as exc:
                exc.add_note(
                    f"Activation {name!r} not found in {list(ALL_ACTIVATIONS)=}"
                )
                raise
            return get_activation(obj, **cfg)

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
            assert isinstance(result, Activation)
            return result

        # if an instance, return as-is
        case Activation() as instance:
            if cfg:
                raise ValueError(f"Cannot pass arguments to an instance: {cfg!r}")
            return instance

        case _:
            raise TypeError(f"Invalid argument: {arg!r}")
