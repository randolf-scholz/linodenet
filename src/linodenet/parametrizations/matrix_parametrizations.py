r"""Parametrizations for matrices (rank-2 tensors)."""

__all__ = [
    "Banded",
    "NegativeDefinite",
    "PositiveDefinite",
    "Contraction",
    "Diagonal",
    "PositiveSemiDefinite",
    "Hamiltonian",
    "LipschitzBounded",
    "LowRank",
    "LowerTriangular",
    "Masked",
    "Normal",
    "OrthogonalCayley",
    "OrthogonalHouseholder",
    "SpecialOrthogonal",
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

from linodenet.mappings import projections, surjections

PositiveDefinite = surjections.PositiveDefinite
NegativeDefinite = surjections.NegativeDefinite
PositiveSemiDefinite = surjections.PositiveSemiDefinite

OrthogonalCayley = surjections.OrthogonalCayley
OrthogonalHouseholder = surjections.OrthogonalHouseholder
SpecialOrthogonal = surjections.SpecialOrthogonal

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
RankOne = projections.RankOne
SkewSymmetric = projections.SkewSymmetric
SpectralNormalized = projections.SpectralNormalized
Symmetric = projections.Symmetric
Symplectic = projections.Symplectic
Traceless = projections.Traceless
Tridiagonal = projections.Tridiagonal
UpperTriangular = projections.UpperTriangular
