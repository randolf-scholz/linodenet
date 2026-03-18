r"""Parametrizations for torch."""
# ruff: noqa: F403, F405

__all__ = [
    # submodules
    "parametrize",
    # Constants
    "STATIC_PARAMETRIZATIONS",
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


STATIC_PARAMETRIZATIONS: dict[str, Surjection | type[Parametrization]] = {
    # Wrappers
    # "WrappedParametrization": WrappedParametrization,
    # Parametrizations
    "CayleyMap"             : CayleyMap,
    "GramMatrix"            : GramMatrix,
    "MatrixExponential"     : MatrixExponential,
    # Learnable parametrizations
    "ReZero"                : ReZero,
    # linodenet.projections
    "Hamiltonian"           : Hamiltonian,
    "Identity"              : Identity,
    "Normal"                : Normal,
    "OrthogonalProjection"  : OrthogonalProjection,
    "RankOne"               : RankOne,
    "SkewSymmetric"         : SkewSymmetric,
    "Symmetric"             : Symmetric,
    "Symplectic"            : Symplectic,
    "Traceless"             : Traceless,
    # linodenet.projections masked
    "Diagonal"              : Diagonal,
    "LowerTriangular"       : LowerTriangular,
    "Tridiagonal"           : Tridiagonal,
    "UpperTriangular"       : UpperTriangular,
}  # fmt: skip
r"""Dictionary of all available parametrizations."""

SPECIAL_PARAMETRIZATIONS = {
    "SpectralNorm": SpectralNorm,
    "Banded": Banded,
    "Masked": Masked,
    "LowRank": LowRank,
}
