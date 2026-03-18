r"""Surjections are a weaker form of projections."""

__all__ = [
    # Classes
    "ConcatProjection",
    "GramMatrix",
    "OrthogonalCayley",
    "OrthogonalMatExp",
    "PositiveVector",
    "StochasticVector",
]

from typing import Final

import torch
from torch import Tensor, nn

from linodenet.domains import MatrixDomains, VectorDomains
from linodenet.mappings.base import SurjectionBase
from linodenet.mappings.bijections import CayleyMap
from linodenet.mappings.functional import skew_symmetric
from signatures import signature


class GramMatrix(SurjectionBase):
    r"""Parametrize a matrix via gram matrix ($XᵀX$)."""

    DOMAIN: Final[MatrixDomains] = MatrixDomains.RECTANGULAR
    CODOMAIN: Final[MatrixDomains] = MatrixDomains.POSITIVE_SEMIDEFINITE

    @signature("(..., m, n) -> (..., n, n)")
    def forward(self, x: Tensor) -> Tensor:
        return torch.einsum("...km, ...kn -> ...mn", x, x)

    @signature("(..., n, n) -> (..., n, n)")
    def right_inverse(self, y: Tensor) -> Tensor:
        r"""This requires the matrix square root, which is not implemented in PyTorch.

        See: https://github.com/pytorch/pytorch/issues/9983
        """
        raise NotImplementedError


class ConcatProjection(SurjectionBase):
    r"""Maps $[x,y] ⟼ x$.

    See Also:
        - `linodenet.embeddings.ConcatEmbedding`
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

    @signature("(..., d+e) -> (..., d)")
    def forward(self, x: Tensor) -> Tensor:
        r"""Remove the padded state."""
        return x[..., : self.output_size]

    @signature("(..., d) -> (..., d+e)")
    def right_inverse(self, y: Tensor) -> Tensor:
        r"""Concatenate the input with the padding."""
        shape = y.shape[:-1] + (self.padding_size,)
        return torch.cat([y, self.padding.expand(shape)], dim=-1)


class PositiveVector(SurjectionBase):
    r"""Map vectors to the positive cone componentwise."""

    DOMAIN: Final[VectorDomains] = VectorDomains.REAL
    CODOMAIN: Final[VectorDomains] = VectorDomains.POSITIVE

    @signature("(..., n) -> (..., n)")
    def forward(self, x: Tensor) -> Tensor:
        return torch.nn.functional.softplus(x)

    @signature("(..., n) -> (..., n)")
    def right_inverse(self, y: Tensor) -> Tensor:
        return torch.where(y > 20, y, y + torch.log(-torch.expm1(-y)))


class StochasticVector(SurjectionBase):
    r"""Map vectors to the probability simplex."""

    DOMAIN: Final[VectorDomains] = VectorDomains.REAL
    CODOMAIN: Final[VectorDomains] = VectorDomains.STOCHASTIC

    @signature("(..., n) -> (..., n)")
    def forward(self, x: Tensor) -> Tensor:
        return x.softmax(dim=-1)

    @signature("(..., n) -> (..., n)")
    def right_inverse(self, y: Tensor) -> Tensor:
        logits = y.log()
        return logits - logits.mean(dim=-1, keepdim=True)


class OrthogonalMatExp(SurjectionBase):
    r"""Map square matrices to orthogonal matrices via $X ↦ \exp(½(X-Xᵀ))$.

    Note:
        Over the reals, this construction lands in the determinant-$1$ component of
        the orthogonal group.
    """

    DOMAIN: Final[MatrixDomains] = MatrixDomains.SQUARE
    CODOMAIN: Final[MatrixDomains] = MatrixDomains.SPECIAL_ORTHOGONAL

    @signature("(..., n, n) -> (..., n, n)")
    def forward(self, x: Tensor) -> Tensor:
        return torch.matrix_exp(skew_symmetric(x))

    @signature("(..., n, n) -> (..., n, n)")
    def right_inverse(self, y: Tensor) -> Tensor:
        r"""Compute a matrix-log fallback by diagonalizing the orthogonal matrix."""
        # FIXME: https://github.com/pytorch/pytorch/issues/9983 (matrix_log)
        eigenvalues, eigenvectors = torch.linalg.eig(y)
        log_diagonal = torch.diag_embed(eigenvalues.log())
        log_matrix = eigenvectors @ log_diagonal @ torch.linalg.inv(eigenvectors)
        return skew_symmetric(log_matrix.real)


class OrthogonalCayley(SurjectionBase):
    r"""Map square matrices to orthogonal matrices via skew-symmetrization and Cayley.

    Note:
        This construction is the composition $X ↦ ½(X-Xᵀ) ↦ (𝕀-A)(𝕀+A)^{-1}$.
        Its image is `MatrixDomains.CAYLEY_ORTHOGONAL`, i.e. the orthogonal matrices
        without eigenvalue $-1$.
    """

    DOMAIN: Final[MatrixDomains] = MatrixDomains.SQUARE
    CODOMAIN: Final[MatrixDomains] = MatrixDomains.CAYLEY_ORTHOGONAL

    def __init__(self) -> None:
        super().__init__()
        self.cayley = CayleyMap()

    @signature("(..., n, n) -> (..., n, n)")
    def forward(self, x: Tensor) -> Tensor:
        return self.cayley(skew_symmetric(x))

    @signature("(..., n, n) -> (..., n, n)")
    def right_inverse(self, y: Tensor) -> Tensor:
        return self.cayley.inverse(y)
