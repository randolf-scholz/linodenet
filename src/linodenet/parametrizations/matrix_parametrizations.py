r"""Parametrizations for matrices (rank-2 tensors)."""

__all__ = [
    "Banded",
    "CayleyMap",
    "Contraction",
    "Diagonal",
    "GramMatrix",
    "Hamiltonian",
    "Identity",
    "LipschitzBounded",
    "LowRank",
    "LowerTriangular",
    "Masked",
    "MatrixExponential",
    "Normal",
    "OrthogonalProjection",
    "RankOne",
    "SkewSymmetric",
    "SpectralNormalized",
    "Symmetric",
    "Symplectic",
    "Traceless",
    "Tridiagonal",
    "UpperTriangular",
]

from linodenet.mappings import bijections, projections, surjections

MatrixExponential = bijections.MatrixExponential
CayleyMap = bijections.CayleyMap

GramMatrix = surjections.GramMatrix

Banded = projections.Banded
Contraction = projections.Contraction
Diagonal = projections.Diagonal
Hamiltonian = projections.Hamiltonian
Identity = projections.Identity
LipschitzBounded = projections.LipschitzBounded
LowRank = projections.LowRank
LowerTriangular = projections.LowerTriangular
Masked = projections.Masked
Normal = projections.Normal
OrthogonalProjection = projections.Orthogonal
RankOne = projections.RankOne
SkewSymmetric = projections.SkewSymmetric
SpectralNormalized = projections.SpectralNormalized
Symmetric = projections.Symmetric
Symplectic = projections.Symplectic
Traceless = projections.Traceless
Tridiagonal = projections.Tridiagonal
UpperTriangular = projections.UpperTriangular
