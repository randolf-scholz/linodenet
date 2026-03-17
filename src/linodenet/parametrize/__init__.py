r"""Parametrizations for torch."""
# ruff: noqa: F403, F405

__all__ = [
    # Constants
    "PARAMETRIZATIONS",
    # ABCs & Protocols
    "Parametrization",
    "WrappedParametrization",
    "ParametrizationBase",
    "ParametrizationList",
    # torch.nn.utils.parametrize replacements
    "cached",
    "is_parametrized",
    "parametrize",
    "register_parametrization",
    # Functions
    "deepcopy_with_parametrizations",
    "get_parametrizations",
    "iter_parametrizations",
    "register_optimizer_hook",
    "is_parametrization",
    # "update_caches",
    # "update_originals",
    # "detach_caches",
    "update_parametrizations",
]

from linodenet.mappings.projections import SpectralNorm
from linodenet.parametrize import matrix_parametrizations, tensor_parametrizations
from linodenet.parametrize.base import (
    Parametrization,
    ParametrizationBase,
    ParametrizationList,
    WrappedParametrization,
    cached,
    deepcopy_with_parametrizations,
    # detach_caches,
    get_parametrizations,
    is_parametrization,
    is_parametrized,
    iter_parametrizations,
    parametrize,
    register_optimizer_hook,
    register_parametrization,
    # update_caches,
    # update_originals,
    update_parametrizations,
)
from linodenet.parametrize.matrix_parametrizations import *
from linodenet.parametrize.tensor_parametrizations import *

__all__ += matrix_parametrizations.__all__
__all__ += tensor_parametrizations.__all__


PARAMETRIZATIONS = {
    # Wrappers
    # "WrappedParametrization": WrappedParametrization,
    # Parametrizations
    "CayleyMap"             : CayleyMap,
    "GramMatrix"            : GramMatrix,
    "MatrixExponential"     : MatrixExponential,
    "SpectralNorm"          : SpectralNorm,
    # Learnable parametrizations
    "ReZero"                : ReZero,
    # linodenet.projections
    "Hamiltonian"           : Hamiltonian,
    "Identity"              : Identity,
    "LowRank"               : LowRank,
    "Normal"                : Normal,
    "OrthogonalProjection"  : OrthogonalProjection,
    "RankOne"               : RankOne,
    "SkewSymmetric"         : SkewSymmetric,
    "Symmetric"             : Symmetric,
    "Symplectic"            : Symplectic,
    "Traceless"             : Traceless,
    # linodenet.projections masked
    "Banded"                : Banded,
    "Diagonal"              : Diagonal,
    "LowerTriangular"       : LowerTriangular,
    "Masked"                : Masked,
    "Tridiagonal"           : Tridiagonal,
    "UpperTriangular"       : UpperTriangular,
}  # fmt: skip
r"""Dictionary of all available parametrizations."""
