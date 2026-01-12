r"""#TODO add module summary line.

#TODO add module description.
"""

__all__ = [
    # ABCs & Protocols
    "Embedding",
    "EmbeddingBase",
    # Classes
    "ConcatEmbedding",
    "LinearEmbedding",
]

from abc import abstractmethod
from typing import Final, Protocol, runtime_checkable

import torch
from torch import Tensor, jit, nn
from torch.nn import functional

from linodenet.signatures import signature


@runtime_checkable
class Embedding[X, Y](Protocol):
    r"""Protocol for Embedding Components."""

    @abstractmethod
    @signature("(...) -> (...)")
    def __call__(self, x: X, /) -> Y:
        r"""Forward pass of the embedding."""
        ...

    @abstractmethod
    @signature("(...) -> (...)")
    def left_inverse(self, y: Y, /) -> X:
        r"""Left-inverse pass of the embedding."""
        ...


class EmbeddingBase(nn.Module, Embedding[Tensor, Tensor]):
    r"""Abstract Base Class for Embedding components."""

    @abstractmethod
    def forward(self, x: Tensor, /) -> Tensor: ...
    @abstractmethod
    def left_inverse(self, y: Tensor, /) -> Tensor: ...

    @jit.export
    def encode(self, x: Tensor) -> Tensor:
        r"""Alias for `forward` method."""
        return self.forward(x)

    @jit.export
    def decode(self, y: Tensor) -> Tensor:
        r"""Alias for `left_inverse` method."""
        return self.left_inverse(y)


class ConcatEmbedding(EmbeddingBase):
    r"""Maps $x ⟼ [x,w]$.

    This map is left-invertible via $[x,w] ⟼ x$.

    See Also:
        - `linodenet.projections.ConcatProjection`
    """

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

    @jit.export
    @signature("(..., d) -> (..., d+e)")
    def forward(self, x: Tensor) -> Tensor:
        r"""Concatenate the input with the padding."""
        shape = x.shape[:-1] + (self.padding_size,)
        return torch.cat([x, self.padding.expand(shape)], dim=-1)

    @jit.export
    @signature("(..., d+e) -> (..., d)")
    def left_inverse(self, y: Tensor) -> Tensor:
        r"""Remove the padded state."""
        return y[..., : self.input_size]


class LinearEmbedding(EmbeddingBase):
    r"""Maps $x ↦ Ax + b$ and $y ↦ A⁺(y-b)$.

    Note:
        x ↦ Ax + b is surjective if A has full row rank (input_size ≤ output_size).
        x ↦ Ax + b is injective if A has full column rank (input_size ≥ output_size).
        In the former case, the map is right-invertible, in the latter left-invertible.
    """

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

    @jit.export
    def reset_parameters(self) -> None:
        r"""Reset both weight matrix and bias vector."""
        with torch.no_grad():
            bound = float(torch.rsqrt(torch.tensor(self.input_size)))
            self.weight.uniform_(-bound, bound)
            if self.bias is not None:
                self.bias.uniform_(-bound, bound)

    @jit.export
    @signature("(..., d) -> (..., e)")
    def forward(self, x: Tensor) -> Tensor:
        r"""Concatenate the input with the padding."""
        return functional.linear(x, self.weight, self.bias)

    @jit.export
    @signature("(..., d) -> (..., e)")
    def left_inverse(self, y: Tensor) -> Tensor:
        r"""Remove the padded state."""
        if self.with_bias:
            y = y - self.bias
        return torch.linalg.lstsq(self.weight, y)[0]
