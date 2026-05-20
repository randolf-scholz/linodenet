r"""Parametrizations for torch."""
# ruff: noqa: F403, F405

__all__ = [
    # submodules
    "exponential_maps",
    "parametrize",
    # Constants
    "RIEMANN_MANIFOLDS",
    "SCALAR_PARAMETRIZATIONS",
    "MATRIX_PARAMETRIZATIONS",
    "VECTOR_PARAMETRIZATIONS",
    "TENSOR_PARAMETRIZATIONS",
]

from linodenet.nn import parametrize
from linodenet.nn.parametrize import *

from . import (
    exponential_maps,
    matrix_parametrizations,
    scalar_parametrizations,
    tensor_parametrizations,
    vector_parametrizations,
)
from .exponential_maps import *
from .matrix_parametrizations import *
from .tensor_parametrizations import *
from .vector_parametrizations import *

__all__ += parametrize.__all__
__all__ += exponential_maps.__all__
__all__ += matrix_parametrizations.__all__
__all__ += scalar_parametrizations.__all__
__all__ += tensor_parametrizations.__all__
__all__ += vector_parametrizations.__all__


RIEMANN_MANIFOLDS: dict[str, type[ManifoldBase]] = {
    "PositiveDefiniteManifold"  : exponential_maps.PositiveDefiniteManifold,
    "SpecialOrthogonalManifold" : exponential_maps.SpecialOrthogonalManifold,
    "SphereManifold"            : exponential_maps.SphereManifold,
}  # fmt: skip
r"""Dictionary of all available Riemannian-manifold modules."""

SCALAR_PARAMETRIZATIONS: dict[str, type[Surjection]] = {}
r"""Dictionary of all builtin scalar parametrizations."""

VECTOR_PARAMETRIZATIONS: dict[str, type[Surjection]] = {
    "UnitVector"           : vector_parametrizations.UnitVector,
    "StochasticVector"     : vector_parametrizations.StochasticVector,
    "PositiveVector"       : vector_parametrizations.PositiveVector,
}  # fmt: skip
r"""Dictionary of all builtin vector parametrizations."""

MATRIX_PARAMETRIZATIONS: dict[str, type[Surjection]] = {
    "Banded"               : matrix_parametrizations.Banded,
    "CholeskyFactor"       : matrix_parametrizations.CholeskyFactor,
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
r"""Dictionary of all builtin matrix parametrizations."""

TENSOR_PARAMETRIZATIONS: dict[str, type[Surjection]] = {
    "ReZero": tensor_parametrizations.ReZero,
}  # fmt: skip
r"""Dictionary of all builtin tensor parametrizations."""
