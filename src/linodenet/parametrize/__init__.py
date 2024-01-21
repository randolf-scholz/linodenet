"""Parametrizations for Torch."""

__all__ = [
    # Constants
    "PARAMETRIZATIONS",
    # ABCs & Protocols
    "GeneralParametrization",
    "Parametrization",
    # Classes
    "ParametrizationBase",
    "ParametrizationDict",
    "ParametrizationMulticache",
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
    "Orthogonal",
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
    "detach_caches",
    "get_parametrizations",
    "register_optimizer_hook",
    "update_caches",
    "update_originals",
    "update_parametrizations",
]

from linodenet.parametrize.base import (
    GeneralParametrization,
    Parametrization,
    ParametrizationBase,
    ParametrizationDict,
    ParametrizationMulticache,
    cached,
    deepcopy_with_parametrizations,
    detach_caches,
    get_parametrizations,
    is_parametrized,
    parametrize,
    register_optimizer_hook,
    register_parametrization,
    update_caches,
    update_originals,
    update_parametrizations,
)
from linodenet.parametrize.parametrizations import (
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
    "Identity": Identity,
    "LowRank": LowRank,
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
