r"""Parametrizations for torch."""
# ruff: noqa: F403, F405

__all__ = [
    # submodules
    "parametrize",
    # Constants
    "PARAMETRIZATIONS",
    "MATRIX_PARAMETRIZATIONS",
    "VECTOR_PARAMETRIZATIONS",
]

from linodenet.nn import parametrize
from linodenet.nn.parametrize import *
from linodenet.parametrizations import (
    matrix_parametrizations,
    tensor_parametrizations,
    vector_parametrizations,
)
from linodenet.parametrizations.matrix_parametrizations import *
from linodenet.parametrizations.tensor_parametrizations import *
from linodenet.parametrizations.vector_parametrizations import *

__all__ += parametrize.__all__
__all__ += vector_parametrizations.__all__
__all__ += matrix_parametrizations.__all__
__all__ += tensor_parametrizations.__all__


PARAMETRIZATIONS: dict[str, type[Parametrization]] = {
    "ReZero"                 : tensor_parametrizations.ReZero,
}  # fmt: skip
r"""Dictionary of all available parametrizations."""


VECTOR_PARAMETRIZATIONS: dict[str, type[Surjection]] = {
    "UnitVector"           : vector_parametrizations.UnitVector,
    "StochasticVector"     : vector_parametrizations.StochasticVector,
    "PositiveVector"       : vector_parametrizations.PositiveVector,
}  # fmt: skip

MATRIX_PARAMETRIZATIONS: dict[str, type[Surjection]] = {
    "Banded"               : matrix_parametrizations.Banded,
    "Contraction"          : matrix_parametrizations.Contraction,
    "Diagonal"             : matrix_parametrizations.Diagonal,
    "DiagonallyDominant"   : matrix_parametrizations.DiagonallyDominant,
    "Hamiltonian"          : matrix_parametrizations.Hamiltonian,
    "LipschitzBounded"     : matrix_parametrizations.LipschitzBounded,
    "LowRank"              : matrix_parametrizations.LowRank,
    "LowerTriangular"      : matrix_parametrizations.LowerTriangular,
    "Masked"               : matrix_parametrizations.Masked,
    "NegativeDefinite"     : matrix_parametrizations.NegativeDefinite,
    "Normal"               : matrix_parametrizations.Normal,
    "OrthogonalCayley"     : matrix_parametrizations.OrthogonalCayley,
    "OrthogonalHouseholder": matrix_parametrizations.OrthogonalHouseholder,
    "PositiveDefinite"     : matrix_parametrizations.PositiveDefinite,
    "PositiveSemiDefinite" : matrix_parametrizations.PositiveSemiDefinite,
    "RankOne"              : matrix_parametrizations.RankOne,
    "SkewSymmetric"        : matrix_parametrizations.SkewSymmetric,
    "SpecialOrthogonal"    : matrix_parametrizations.SpecialOrthogonal,
    "SpectralNormalized"   : matrix_parametrizations.SpectralNormalized,
    "Symmetric"            : matrix_parametrizations.Symmetric,
    "Symplectic"           : matrix_parametrizations.Symplectic,
    "Traceless"            : matrix_parametrizations.Traceless,
    "Tridiagonal"          : matrix_parametrizations.Tridiagonal,
    "UpperTriangular"      : matrix_parametrizations.UpperTriangular,
}  # fmt: skip
r"""Dictionary of all available matrix parametrizations."""
