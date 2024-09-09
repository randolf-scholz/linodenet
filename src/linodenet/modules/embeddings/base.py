r"""#TODO add module summary line.

#TODO add module description.
"""

__all__ = [
    # ABCs & Protocols
    "Embedding",
    "EmbeddingABC",
    # Classes
    "ConcatEmbedding",
    "ConcatProjection",
    "LinearEmbedding",
]

from abc import abstractmethod
from typing import Final, Optional, Protocol, Self, runtime_checkable

import torch
from torch import Tensor, jit, nn
from torch.nn import functional


@runtime_checkable
class Embedding(Protocol):
    r"""Protocol for Embedding Components."""

    def __call__(self, x: Tensor, /) -> Tensor:
        r"""Forward pass of the embedding.

        .. Signature: ``... -> ...``.
        """
        ...


class EmbeddingABC(nn.Module):
    r"""Abstract Base Class for Embedding components."""

    @abstractmethod
    def forward(self, x: Tensor, /) -> Tensor:
        r"""Forward pass of the embedding.

        .. Signature: ``... -> ...``.

        Args:
            x: The input tensor to be embedded.

        Returns:
            z: The embedded input tensor.
        """


class ConcatEmbedding(nn.Module):
    r"""Maps $x ⟼ [x,w]$."""

    HP = {
        "__name__": __qualname__,
        "__module__": __name__,
        "input_size": int,
        "output_size": int,
    }
    r"""Dictionary of hyperparameters."""

    # Constants
    input_size: Final[int]
    r"""CONST: The dimensionality of the inputs."""
    output_size: Final[int]
    r"""CONST: The dimensionality of the outputs."""
    padding_size: Final[int]
    r"""CONST: The size of the padding."""

    # BUFFERS
    scale: Tensor
    r"""BUFFER: The scaling scalar."""

    # Parameters
    padding: Tensor
    r"""PARAM: The padding vector."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        inverse: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        if not (input_size <= output_size):
            raise ValueError(
                f"{input_size=} must be smaller or equal to {output_size=}!"
            )
        self.input_size = input_size
        self.output_size = output_size
        self.padding_size = output_size - input_size
        self.padding = nn.Parameter(torch.randn(self.padding_size))

        if inverse is None:
            self.inverse = ConcatProjection(
                input_size=output_size, output_size=input_size, inverse=self
            )
            self.inverse.padding = self.padding

    @jit.export
    def forward(self, x: Tensor) -> Tensor:
        r"""Concatenate the input with the padding.

        .. Signature:: ``(..., d) -> (..., d+e)``.
        """
        shape = x.shape[:-1] + (self.padding_size,)
        return torch.cat([x, self.padding.expand(shape)], dim=-1)

    @jit.export
    def encode(self, x: Tensor) -> Tensor:
        r"""Concatenate the input with the padding.

        .. Signature:: ``(..., d) -> (..., d+e)``.
        """
        shape = x.shape[:-1] + (self.padding_size,)
        return torch.cat([x, self.padding.expand(shape)], dim=-1)

    @jit.export
    def decode(self, y: Tensor) -> Tensor:
        r"""Remove the padded state.

        .. Signature: ``(..., d+e) -> (..., d)``.
        """
        return y[..., : self.input_size]


class ConcatProjection(nn.Module):
    r"""Maps $z = [x,w] ⟼ x$."""

    HP = {
        "__name__": __qualname__,
        "__module__": __name__,
        "input_size": int,
        "output_size": int,
    }
    r"""Dictionary of hyperparameters."""

    # Constants
    input_size: Final[int]
    r"""CONST: The dimensionality of the inputs."""
    output_size: Final[int]
    r"""CONST: The dimensionality of the outputs."""
    padding_size: Final[int]
    r"""CONST: The size of the padding."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        inverse: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        if not (input_size >= output_size):
            raise ValueError(
                f"{input_size=} must be greater or equal to {output_size=}!"
            )
        self.input_size = input_size
        self.output_size = output_size
        self.padding_size = input_size - output_size
        self.padding = nn.Parameter(torch.randn(self.padding_size))

        if inverse is None:
            self.inverse = ConcatEmbedding(
                input_size=output_size, output_size=input_size, inverse=self
            )
            self.inverse.padding = self.padding

    @jit.export
    def forward(self, x: Tensor) -> Tensor:
        r"""Remove the padded state.

        .. Signature: ``(..., d+e) -> (..., d)``.
        """
        return self.encode(x)

    @jit.export
    def encode(self, x: Tensor) -> Tensor:
        r"""Remove the padded state.

        .. Signature: ``(..., d+e) -> (..., d)``.
        """
        return x[..., : self.output_size]

    @jit.export
    def decode(self, y: Tensor) -> Tensor:
        r"""Concatenate the input with the padding.

        .. Signature:: ``(..., d) -> (..., d+e)``.
        """
        shape = y.shape[:-1] + (self.padding_size,)
        return torch.cat([y, self.padding.expand(shape)], dim=-1)


class LinearEmbedding(nn.Module):
    r"""Maps $x ↦ Ax + b$ and $y ↦ A⁺(y-b)$."""

    HP = {
        "__name__": __qualname__,
        "__module__": __name__,
        "input_size": int,
        "output_size": int,
    }
    r"""Dictionary of hyperparameters."""

    # Constants
    input_size: Final[int]
    r"""CONST: The dimensionality of the inputs."""
    output_size: Final[int]
    r"""CONST: The dimensionality of the outputs."""
    is_inverse: Final[bool]
    r"""CONST: Whether this is the inverse of another module."""
    with_bias: Final[bool]
    r"""CONST: Whether this module has a bias."""

    # PARAMS
    weight: Tensor
    r"""PARAM: The weight matriz."""
    bias: Tensor
    r"""PARAM: The bias vector."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        bias: bool = True,
        inverse: Optional[Self] = None,
    ) -> None:
        super().__init__()
        if not (input_size <= output_size):
            raise ValueError(
                f"{input_size=} must be smaller or equal to {output_size=}!"
            )
        self.input_size = input_size
        self.output_size = output_size
        self.with_bias = bias

        self.weight = nn.Parameter(torch.empty((output_size, input_size)))
        self.register_parameter(
            "bias", nn.Parameter(torch.empty(output_size)) if bias else None
        )
        self.reset_parameters()

        self.is_inverse = inverse is not None
        if not self.is_inverse:
            self.inverse = LinearEmbedding(
                input_size=output_size, output_size=input_size, inverse=self
            )
            self.inverse.weight = self.weight
            self.inverse.bias = self.bias

    @jit.export
    def reset_parameters(self) -> None:
        r"""Reset both weight matrix and bias vector."""
        with torch.no_grad():
            bound = float(torch.rsqrt(torch.tensor(self.input_size)))
            self.weight.uniform_(-bound, bound)
            if self.bias is not None:
                self.bias.uniform_(-bound, bound)

    @jit.export
    def forward(self, x: Tensor) -> Tensor:
        r"""Concatenate the input with the padding.

        .. Signature:: ``(..., d) -> (..., e)``.
        """
        if self.is_inverse:
            return self._decode(x)
        return self._encode(x)

    @jit.export
    def encode(self, x: Tensor) -> Tensor:
        r"""Concatenate the input with the padding.

        .. Signature:: ``(..., d) -> (..., e)``.
        """
        if self.is_inverse:
            return self._decode(x)
        return self._encode(x)

    @jit.export
    def decode(self, y: Tensor) -> Tensor:
        r"""Remove the padded state.

        .. Signature: ``(..., d+e) -> (..., d)``.
        """
        if self.is_inverse:
            return self._encode(y)
        return self._decode(y)

    @jit.export
    def _encode(self, x: Tensor) -> Tensor:
        return functional.linear(x, self.weight, self.bias)

    @jit.export
    def _decode(self, y: Tensor) -> Tensor:
        if self.with_bias:
            y = y - self.bias
        return torch.linalg.lstsq(self.weight, y)[0]
