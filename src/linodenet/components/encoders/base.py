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
]

from abc import abstractmethod
from typing import Protocol, runtime_checkable

from torch import Tensor, nn


@runtime_checkable
class Encoder(Protocol):
    r"""Protocol for Encoder Components."""

    def __call__(self, x: Tensor, /) -> Tensor:
        r"""Forward pass of the encoder.

        .. Signature: ``(..., d) -> (..., d)``.
        """
        ...


class EncoderABC(nn.Module):
    r"""Abstract Base Class for Encoder components."""

    @abstractmethod
    def forward(self, x: Tensor, /) -> Tensor:
        r"""Forward pass of the encoder.

        Args:
            x: The input tensor to be encoded.

        Returns:
            z: The encoded tensor.
        """
        ...
