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
    # torch.nn.utils.parametrize replacements
    "parametrize",
    "is_parametrized",
    "register_parametrization",
    "cached",
    # Functions
    "deepcopy_with_parametrizations",
    "detach_caches",
    "get_parametrizations",
    "register_optimizer_hook",
    "update_caches",
    "update_originals",
    "update_parametrizations",
    # Parametrizations
    "CayleyMap",
    "GramMatrix",
    "MatrixExponential",
    "SpectralNormalization",
    # Learnable parametrizations
    "ReZero",
    # linodenet.projections
    "Hamiltonian",
    "Identity",
    "LowRank",
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
