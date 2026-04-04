r"""Embedding components.

An embedding is an injective mapping $f:X → Y$, that is:

1. It is left-invertible, i.e. there exists a mapping $g:Y → X$ such that
   $g(f(x)) = x$ for all $x ∈ X$, but not necessarily $f(g(y)) = y$ for all $y ∈ Y$.
2. The output dimensionality is (generally) larger than the input dimensionality.
3. we require both an `forward` and a `left_inverse` method, aliased to `encode` and `decode`.
"""

__all__ = [
    # Classes
    "ConcatEmbedding",
    "LinearEmbedding",
]

from typing import Final

import torch
from torch import Tensor, nn
from torch.nn import functional

from linodenet.domains import VectorDomains
from signatures import signature

from .base import EmbeddingBase


class ConcatEmbedding(EmbeddingBase):
    r"""Maps $x ⟼ [x,w]$.

    This map is left-invertible via $[x,w] ⟼ x$.

    See Also:
        - `linodenet.projections.ConcatProjection`
    """

    DOMAIN: Final[VectorDomains] = VectorDomains.REAL
    CODOMAIN: Final[VectorDomains] = VectorDomains.REAL

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

    @property
    def config(self) -> dict:
        return {
            "input_size": self.input_size,
            "output_size": self.output_size,
        }

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

    @signature("(..., d) -> (..., d+e)")
    def forward(self, x: Tensor, /) -> Tensor:
        r"""Concatenate the input with the padding."""
        shape = x.shape[:-1] + (self.padding_size,)
        return torch.cat([x, self.padding.expand(shape)], dim=-1)

    @signature("(..., d+e) -> (..., d)")
    def left_inverse(self, y: Tensor, /) -> Tensor:
        r"""Remove the padded state."""
        return y[..., : self.input_size]


class LinearEmbedding(EmbeddingBase):
    r"""Maps $x ↦ Ax + b$ and $y ↦ A⁺(y-b)$.

    Note:
        x ↦ Ax + b is surjective if A has full row rank (input_size ≤ output_size).
        x ↦ Ax + b is injective if A has full column rank (input_size ≥ output_size).
        In the former case, the map is right-invertible, in the latter left-invertible.
    """

    DOMAIN: Final[VectorDomains] = VectorDomains.REAL
    CODOMAIN: Final[VectorDomains] = VectorDomains.REAL

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

    @property
    def config(self) -> dict:
        return {
            "input_size": self.input_size,
            "output_size": self.output_size,
            "bias": self.with_bias,
        }

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

    def reset_parameters(self) -> None:
        r"""Reset both weight matrix and bias vector."""
        with torch.no_grad():
            bound = float(torch.rsqrt(torch.tensor(self.input_size)))
            self.weight.uniform_(-bound, bound)
            if self.bias is not None:
                self.bias.uniform_(-bound, bound)

    @signature("(..., d) -> (..., e)")
    def forward(self, x: Tensor, /) -> Tensor:
        r"""Concatenate the input with the padding."""
        return functional.linear(x, self.weight, self.bias)

    @signature("(..., d) -> (..., e)")
    def left_inverse(self, y: Tensor, /) -> Tensor:
        r"""Remove the padded state."""
        if self.with_bias:
            y = y - self.bias
        return torch.linalg.lstsq(self.weight, y)[0]
