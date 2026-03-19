r"""Parametrizations for matrices (rank-2 tensors)."""

__all__ = [
    "Banded",
    "CayleyMap",
    "Contraction",
    "Diagonal",
    "GramMatrix",
    "Hamiltonian",
    "LipschitzBounded",
    "LowRank",
    "LowerTriangular",
    "Masked",
    "MatrixExponential",
    "Normal",
    "OrthogonalCayley",
    "OrthogonalHouseholder",
    "OrthogonalMatExp",
    "OrthogonalProjection",
    "RankOne",
    "SkewSymmetric",
    "SpectralNormalized",
    "Symmetric",
    "DiagonallyDominant",
    "Symplectic",
    "Traceless",
    "Tridiagonal",
    "UpperTriangular",
]

from linodenet.mappings import bijections, projections, surjections

MatrixExponential = bijections.MatrixExponential
CayleyMap = bijections.CayleyMap

GramMatrix = surjections.GramMatrix
OrthogonalCayley = surjections.OrthogonalCayley
OrthogonalHouseholder = surjections.OrthogonalHouseholder
OrthogonalMatExp = surjections.OrthogonalMatExp

Banded = projections.Banded
Contraction = projections.Contraction
Diagonal = projections.Diagonal
DiagonallyDominant = projections.DiagonallyDominant
Hamiltonian = projections.Hamiltonian
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
