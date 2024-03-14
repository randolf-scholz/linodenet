r"""Different Filter models to be used in conjunction with LinodeNet.

A Filter takes two positional inputs:
    - An input tensor x: the current estimation of the state of the system
    - An input tensor y: the current measurement of the system
    - An optional input tensor mask: a mask to be applied to the input tensor
"""

__all__ = [
    # Classes
    "Encoder",
    "EncoderABC",
    "Identity",
]

from abc import abstractmethod

from torch import Tensor, nn
from typing_extensions import Protocol, runtime_checkable


@runtime_checkable
class Encoder(Protocol):
    """Protocol for Encoder Components."""

    def __call__(self, x: Tensor, /) -> Tensor:
        """Forward pass of the encoder.

        .. Signature: ``(..., d) -> (..., d)``.
        """
        ...


class EncoderABC(nn.Module):
    """Abstract Base Class for Encoder components."""

    @abstractmethod
    def forward(self, x: Tensor, /) -> Tensor:
        r"""Forward pass of the encoder.

        Args:
            x: The input tensor to be encoded.

        Returns:
            z: The encoded tensor.
        """
        ...


class Identity(nn.Module):
    r"""Identity with HP attribute."""

    HP = {
        "__name__": __qualname__,
        "__module__": __name__,
    }
    r"""Hyperparameters of the component."""

    def forward(self, x: Tensor) -> Tensor:
        r""".. Signature: ``... -> ...``."""
        return x