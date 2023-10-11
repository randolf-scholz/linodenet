"""Parametrizations for Torch."""

__all__ = [
    # Protocol
    "Parametrization",
    # Constants
    "PARAMETRIZATIONS",
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
    "SpectralNormalization",
    # Learnable parametrizations
    "ReZero",
    # linodenet.projections
    "Hamiltonian",
    "Normal",
    "Orthogonal",
    "SkewSymmetric",
    "Symmetric",
    "Symplectic",
    "Traceless",
    # linodenet.projections masked
    "Banded",
    "Diagonal",
    "LowerTriangular",
    "Masked",
    "UpperTriangular",
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
    Banded,
    CayleyMap,
    Diagonal,
    GramMatrix,
    Hamiltonian,
    LowerTriangular,
    Masked,
    MatrixExponential,
    Normal,
    Orthogonal,
    ReZero,
    SkewSymmetric,
    SpectralNormalization,
    Symmetric,
    Symplectic,
    Traceless,
    UpperTriangular,
)

PARAMETRIZATIONS: dict[str, type[Parametrization]] = {
    # Parametrizations
    "CayleyMap": CayleyMap,
    "GramMatrix": GramMatrix,
    "MatrixExponential": MatrixExponential,
    "SpectralNormalization": SpectralNormalization,
    # Learnable parametrizations
    "ReZero": ReZero,
    # linodenet.projections
    "Hamiltonian": Hamiltonian,
    "Normal": Normal,
    "Orthogonal": Orthogonal,
    "SkewSymmetric": SkewSymmetric,
    "Symmetric": Symmetric,
    "Symplectic": Symplectic,
    "Traceless": Traceless,
    # linodenet.projections masked
    "Banded": Banded,
    "Diagonal": Diagonal,
    "LowerTriangular": LowerTriangular,
    "Masked": Masked,
    "UpperTriangular": UpperTriangular,
}
