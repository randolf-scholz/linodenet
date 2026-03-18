r"""Parametrizations for torch."""
# ruff: noqa: F403, F405

__all__ = [
    # submodules
    "parametrize",
    # Constants
    "PARAMETRIZATIONS",
    "MATRIX_PARAMETRIZATIONS",
    "VECTOR_PARAMETRIZATIONS",
    "SPECIAL_PARAMETRIZATIONS",
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


PARAMETRIZATIONS: dict[str, Surjection | type[Parametrization]] = {
    "CayleyMap"              : matrix_parametrizations.CayleyMap,
    # Learnable parametrizations
    "ReZero"                 : tensor_parametrizations.ReZero,
    "WrappedParametrization" : parametrize.WrappedParametrization

}  # fmt: skip
r"""Dictionary of all available parametrizations."""

MATRIX_PARAMETRIZATIONS: dict[str, Surjection] = {
    "Hamiltonian"          : matrix_parametrizations.Hamiltonian,
    "Identity"             : matrix_parametrizations.Identity,
    "Normal"               : matrix_parametrizations.Normal,
    "OrthogonalProjection" : matrix_parametrizations.OrthogonalProjection,
    "RankOne"              : matrix_parametrizations.RankOne,
    "SkewSymmetric"        : matrix_parametrizations.SkewSymmetric,
    "Symmetric"            : matrix_parametrizations.Symmetric,
    "Symplectic"           : matrix_parametrizations.Symplectic,
    "Traceless"            : matrix_parametrizations.Traceless,
    "Diagonal"             : matrix_parametrizations.Diagonal,
    "LowerTriangular"      : matrix_parametrizations.LowerTriangular,
    "Tridiagonal"          : matrix_parametrizations.Tridiagonal,
    "UpperTriangular"      : matrix_parametrizations.UpperTriangular,
    "GramMatrix"           : matrix_parametrizations.GramMatrix,
    "MatrixExponential"    : matrix_parametrizations.MatrixExponential,
}  # fmt: skip

VECTOR_PARAMETRIZATIONS: dict[str, Surjection] = {
    "UnitVector"           : vector_parametrizations.UnitVector,
    "StochasticVector"     : vector_parametrizations.StochasticVector,
    "PositiveVector"       : vector_parametrizations.PositiveVector,
}  # fmt: skip

SPECIAL_PARAMETRIZATIONS: dict[str, type[Surjection]] = {
    "SpectralNorm" : matrix_parametrizations.SpectralNorm,
    "Banded"       : matrix_parametrizations.Banded,
    "Masked"       : matrix_parametrizations.Masked,
    "LowRank"      : matrix_parametrizations.LowRank,
}  # fmt: skip
