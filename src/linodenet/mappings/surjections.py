r"""Surjections are a weaker form of projections."""

__all__ = [
    # Classes
    "ConcatProjection",
    "CayleyMap",
    "GramMatrix",
]

from typing import Final

import torch
from torch import Tensor, jit, nn

from linodenet.domains import MatrixDomains, VectorDomains
from linodenet.mappings.base import SurjectionBase
from signatures import signature


class GramMatrix(SurjectionBase):
    r"""Parametrize a matrix via gram matrix ($XᵀX$)."""

    DOMAIN: Final[MatrixDomains] = MatrixDomains.GENERAL
    CODOMAIN: Final[MatrixDomains] = MatrixDomains.POSITIVE_SEMIDEFINITE

    @jit.export
    @signature("(..., n, n) -> (..., n, n)")
    def forward(self, x: Tensor) -> Tensor:
        return torch.einsum("...kn, ...mk -> ...mn", x, x)

    @jit.export
    @signature("(..., n, n) -> (..., n, n)")
    def right_inverse(self, y: Tensor) -> Tensor:
        r"""This requires the matrix square root, which is not implemented in PyTorch.

        See: https://github.com/pytorch/pytorch/issues/9983
        """
        raise NotImplementedError


class CayleyMap(SurjectionBase):
    r"""Parametrize a matrix to be orthogonal via Cayley-Map.

    References:
        - https://pytorch.org/tutorials/intermediate/parametrizations.html
        - https://en.wikipedia.org/wiki/Cayley_transform#Matrix_map
    """

    DOMAIN: Final[MatrixDomains] = MatrixDomains.SKEW_SYMMETRIC
    CODOMAIN: Final[MatrixDomains] = MatrixDomains.SPECIAL_ORTHOGONAL

    @jit.export
    @signature("(..., n, n) -> (..., n, n)")
    def forward(self, x: Tensor) -> Tensor:
        I = torch.eye(x.shape[-1], dtype=x.dtype, device=x.device)
        return torch.linalg.lstsq(I + x, I - x).solution

    @jit.export
    @signature("(..., n, n) -> (..., n, n)")
    def right_inverse(self, y: Tensor) -> Tensor:
        I = torch.eye(y.shape[-1], dtype=y.dtype, device=y.device)
        return torch.linalg.lstsq(I - y, I + y).solution


class ConcatProjection(SurjectionBase):
    r"""Maps $[x,y] ⟼ x$.

    See Also:
        - `linodenet.embeddings.ConcatEmbedding`
    """

    DOMAIN: Final[VectorDomains] = VectorDomains.GENERAL
    CODOMAIN: Final[VectorDomains] = VectorDomains.GENERAL

    # Constants
    input_size: Final[int]
    r"""CONST: The dimensionality of the inputs."""
    output_size: Final[int]
    r"""CONST: The dimensionality of the outputs."""
    padding_size: Final[int]
    r"""CONST: The size of the padding."""

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
        if not (input_size >= output_size):
            raise ValueError(
                f"{input_size=} must be greater or equal to {output_size=}!"
            )
        self.input_size = input_size
        self.output_size = output_size
        self.padding_size = input_size - output_size
        self.padding = nn.Parameter(torch.randn(self.padding_size))

    @jit.export
    @signature("(..., d+e) -> (..., d)")
    def forward(self, x: Tensor) -> Tensor:
        r"""Remove the padded state."""
        return x[..., : self.output_size]

    @jit.export
    @signature("(..., d) -> (..., d+e)")
    def right_inverse(self, y: Tensor) -> Tensor:
        r"""Concatenate the input with the padding."""
        shape = y.shape[:-1] + (self.padding_size,)
        return torch.cat([y, self.padding.expand(shape)], dim=-1)
