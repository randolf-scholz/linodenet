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

from linodenet.signatures import signature


@runtime_checkable
class Encoder(Protocol):
    r"""Protocol for Encoder Components."""

    @signature("(..., *xs) -> (..., *xs)")
    def __call__(self, x: Tensor, /) -> Tensor: ...


class EncoderABC(nn.Module):
    r"""Abstract Base Class for Encoder components."""

    @abstractmethod
    @signature("(..., *xs) -> (..., *xs)")
    def forward(self, x: Tensor, /) -> Tensor:
        r"""Forward pass of the encoder.

        Args:
            x: The input tensor to be encoded.

        Returns:
            z: The encoded tensor.
        """
        ...
