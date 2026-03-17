r"""Parametrizations for matrices (rank-2 tensors)."""

__all__ = [
    # Parametrizations
    "CayleyMap",
    "GramMatrix",
    "MatrixExponential",
    # inherited from linodenet.projections
    "Banded",
    "Diagonal",
    "SpectralNorm",
    "Hamiltonian",
    "Identity",
    "LowRank",
    "LowerTriangular",
    "Masked",
    "Normal",
    "OrthogonalProjection",
    "RankOne",
    "SkewSymmetric",
    "Symmetric",
    "Symplectic",
    "Traceless",
    "Tridiagonal",
    "UpperTriangular",
]

from typing import Final

import torch
from torch import Tensor, jit

from linodenet.domains import MatrixDomains
from linodenet.mappings import bijections, projections, surjections
from linodenet.parametrize.base import ParametrizationBase
from signatures import signature

# Fixed projection modules are wrapped lazily by `register_parametrization`.
MatrixExponential = bijections.MatrixExponential()
Diagonal = projections.Diagonal()
Hamiltonian = projections.Hamiltonian()
Identity = projections.Identity()
LowerTriangular = projections.LowerTriangular()
Normal = projections.Normal()
OrthogonalProjection = projections.Orthogonal()
RankOne = projections.RankOne()
SkewSymmetric = projections.SkewSymmetric()
Symmetric = projections.Symmetric()
Symplectic = projections.Symplectic()
Traceless = projections.Traceless()
Tridiagonal = projections.Tridiagonal()
UpperTriangular = projections.UpperTriangular()
GramMatrix = surjections.GramMatrix()


Banded = projections.Banded
Masked = projections.Masked
LowRank = projections.LowRank
SpectralNorm = projections.LipschitzBounded


class CayleyMap(ParametrizationBase):
    r"""Parametrize a matrix to be orthogonal via Cayley-Map.

    References:
        - https://pytorch.org/tutorials/intermediate/parametrizations.html
        - https://en.wikipedia.org/wiki/Cayley_transform#Matrix_map
    """

    DOMAIN: Final[MatrixDomains] = MatrixDomains.SKEW_SYMMETRIC
    CODOMAIN: Final[MatrixDomains] = MatrixDomains.SPECIAL_ORTHOGONAL

    Id: Tensor
    r"""BUFFER: The identity matrix."""

    def __init__(self, tensor: Tensor) -> None:
        m, n = tensor.shape
        if m != n:
            raise ValueError(f"Expected a square matrix, got {m} x {n}")
        super().__init__(tensor, unsafe=False)
        self.register_buffer("Id", torch.eye(n))

    @jit.export
    @signature("(..., n, n) -> (..., n, n)")
    def forward(self, x: Tensor) -> Tensor:
        return torch.linalg.lstsq(self.Id + x, self.Id - x).solution

    @jit.export
    @signature("(..., n, n) -> (..., n, n)")
    def right_inverse(self, y: Tensor) -> Tensor:
        return torch.linalg.lstsq(self.Id - y, self.Id + y).solution
