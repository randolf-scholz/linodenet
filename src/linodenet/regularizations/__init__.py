r"""Regularizations for the Linear ODE Networks.

Notes:
    Contains regularizations in both modular and functional form.
    - See `~linodenet.regularizations.functional` for functional implementations.
    - See `~linodenet.regularizations..modular` for modular implementations.
"""

__all__ = [
    # Sub-Modules
    "functional",
    "modular",
    # Constants
    "REGULARIZATIONS",
    "FUNCTIONAL_REGULARIZATIONS",
    "MODULAR_REGULARIZATIONS",
    # ABCs & Protocols
    "Regularization",
    "RegularizationABC",
    # Classes
    "Banded",
    "Contraction",
    "Diagonal",
    "Hamiltonian",
    "Identity",
    "LogDetExp",
    "LowRank",
    "LowerTriangular",
    "Masked",
    "MatrixNorm",
    "Normal",
    "Orthogonal",
    "SkewSymmetric",
    "Symmetric",
    "Symplectic",
    "Traceless",
    "UpperTriangular",
    # Functions
    "banded",
    "contraction",
    "diagonal",
    "hamiltonian",
    "identity",
    "logdetexp",
    "low_rank",
    "lower_triangular",
    "masked",
    "matrix_norm",
    "normal",
    "orthogonal",
    "skew_symmetric",
    "symmetric",
    "symplectic",
    "traceless",
    "upper_triangular",
]

from linodenet.regularizations import functional, modular
from linodenet.regularizations.functional import (
    Regularization,
    banded,
    contraction,
    diagonal,
    hamiltonian,
    identity,
    logdetexp,
    low_rank,
    lower_triangular,
    masked,
    matrix_norm,
    normal,
    orthogonal,
    skew_symmetric,
    symmetric,
    symplectic,
    traceless,
    upper_triangular,
)
from linodenet.regularizations.modular import (
    Banded,
    Contraction,
    Diagonal,
    Hamiltonian,
    Identity,
    LogDetExp,
    LowerTriangular,
    LowRank,
    Masked,
    MatrixNorm,
    Normal,
    Orthogonal,
    RegularizationABC,
    SkewSymmetric,
    Symmetric,
    Symplectic,
    Traceless,
    UpperTriangular,
)

FUNCTIONAL_REGULARIZATIONS: dict[str, Regularization] = {
    "banded"           : banded,
    "contraction"      : contraction,
    "diagonal"         : diagonal,
    "hamiltonian"      : hamiltonian,
    "identity"         : identity,
    "logdetexp"        : logdetexp,
    "low_rank"         : low_rank,
    "lower_triangular" : lower_triangular,
    "masked"           : masked,
    "matrix_norm"      : matrix_norm,
    "normal"           : normal,
    "orthogonal"       : orthogonal,
    "skew_symmetric"   : skew_symmetric,
    "symmetric"        : symmetric,
    "symplectic"       : symplectic,
    "traceless"        : traceless,
    "upper_triangular" : upper_triangular,
}  # fmt: skip
r"""Dictionary of all available modular metrics."""

MODULAR_REGULARIZATIONS: dict[str, type[Regularization]] = {
    "Banded"          : Banded,
    "Contraction"     : Contraction,
    "Diagonal"        : Diagonal,
    "Hamiltonian"     : Hamiltonian,
    "Identity"        : Identity,
    "LogDetExp"       : LogDetExp,
    "LowRank"         : LowRank,
    "LowerTriangular" : LowerTriangular,
    "Masked"          : Masked,
    "MatrixNorm"      : MatrixNorm,
    "Normal"          : Normal,
    "Orthogonal"      : Orthogonal,
    "SkewSymmetric"   : SkewSymmetric,
    "Symmetric"       : Symmetric,
    "Symplectic"      : Symplectic,
    "Traceless"       : Traceless,
    "UpperTriangular" : UpperTriangular,
}  # fmt: skip
r"""Dictionary of all available modular metrics."""

REGULARIZATIONS: dict[str, Regularization | type[Regularization]] = {
    **FUNCTIONAL_REGULARIZATIONS,
    **MODULAR_REGULARIZATIONS,
}
r"""Dictionary containing all available projections."""
