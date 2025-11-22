r"""Parametrizations for torch."""

__all__ = [
    # Constants
    "PARAMETRIZATIONS",
    # ABCs & Protocols
    "Parametrization",
    "WrappedParametrization",
    "ParametrizationBase",
    "ParametrizationList",
    # Parametrizations
    "Banded",
    "CayleyMap",
    "Diagonal",
    "GramMatrix",
    "Hamiltonian",
    "Identity",
    "LowRank",
    "LowerTriangular",
    "Masked",
    "MatrixExponential",
    "Normal",
    "OrthogonalProjection",
    "ReZero",
    "SkewSymmetric",
    "SpectralNormalization",
    "Symmetric",
    "Symplectic",
    "Traceless",
    "UpperTriangular",
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
from linodenet.parametrize.matrix_parametrizations import (
    Banded,
    CayleyMap,
    Diagonal,
    GramMatrix,
    Hamiltonian,
    Identity,
    LowerTriangular,
    LowRank,
    Masked,
    MatrixExponential,
    Normal,
    OrthogonalProjection,
    SkewSymmetric,
    SpectralNormalization,
    Symmetric,
    Symplectic,
    Traceless,
    UpperTriangular,
)
from linodenet.parametrize.tensor_parametrizations import ReZero

PARAMETRIZATIONS: dict[str, type[ParametrizationBase]] = {
    # Wrappers
    # "WrappedParametrization": WrappedParametrization,
    # Parametrizations
    "CayleyMap"             : CayleyMap,
    "GramMatrix"            : GramMatrix,
    "MatrixExponential"     : MatrixExponential,
    "SpectralNormalization" : SpectralNormalization,
    # Learnable parametrizations
    "ReZero"                : ReZero,
    # linodenet.projections
    "Hamiltonian"           : Hamiltonian,
    "Identity"              : Identity,
    "LowRank"               : LowRank,
    "Normal"                : Normal,
    "OrthogonalProjection"  : OrthogonalProjection,
    "SkewSymmetric"         : SkewSymmetric,
    "Symmetric"             : Symmetric,
    "Symplectic"            : Symplectic,
    "Traceless"             : Traceless,
    # linodenet.projections masked
    "Banded"                : Banded,
    "Diagonal"              : Diagonal,
    "LowerTriangular"       : LowerTriangular,
    "Masked"                : Masked,
    "UpperTriangular"       : UpperTriangular,
}  # fmt: skip
r"""Dictionary of all available parametrizations."""
