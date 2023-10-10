"""Parametrizations for Torch."""

__all__ = [
    # Protocol
    "Parametrization",
    # Classes
    "ParametrizationBase",
    "ParametrizationDict",
    "parametrize",
    # functions
    "register_parametrization",
    "get_parametrizations",
    "reset_all_caches",
    # Parametrizations
    "CayleyMap",
    "GramMatrix",
    "MatrixExponential",
    "ReZero",
    # linodenet.projections
    "Diagonal",
    "SkewSymmetric",
    "SpectralNormalization",
    "Symmetric",
    "Traceless",
    # Constants
    "PARAMETRIZATIONS",
]

from linodenet.parametrize.base import (
    Parametrization,
    ParametrizationBase,
    ParametrizationDict,
    get_parametrizations,
    parametrize,
    register_parametrization,
    reset_all_caches,
)
from linodenet.parametrize.parametrizations import (
    CayleyMap,
    Diagonal,
    GramMatrix,
    MatrixExponential,
    ReZero,
    SkewSymmetric,
    SpectralNormalization,
    Symmetric,
    Traceless,
)

PARAMETRIZATIONS: dict[str, type[Parametrization]] = {
    "CayleyMap": CayleyMap,
    "GramMatrix": GramMatrix,
    "MatrixExponential": MatrixExponential,
    "ReZero": ReZero,
    # linodenet.projections
    "Diagonal": Diagonal,
    "SkewSymmetric": SkewSymmetric,
    "SpectralNormalization": SpectralNormalization,
    "Symmetric": Symmetric,
    "Traceless": Traceless,
}
