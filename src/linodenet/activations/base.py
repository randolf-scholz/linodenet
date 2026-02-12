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
from typing import Concatenate, Protocol, overload, runtime_checkable

from torch import Tensor, nn

from blueprint import Makes, initialize
from linodenet.signatures import signature

type GenericActivation = Callable[Concatenate[Tensor, ...], Tensor]
r"""Type alias for generic activation functions (may require additional args!)."""


@runtime_checkable
class Activation(Protocol):
    r"""Protocol for Activation Components."""

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
            from linodenet.activations import ALL_ACTIVATIONS  # noqa: PLC0415

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
