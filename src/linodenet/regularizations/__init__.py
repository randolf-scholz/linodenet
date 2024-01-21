r"""Regularizations for the Linear ODE Networks.

Notes:
    Contains regularizations in both modular and functional form.
    - See `~linodenet.regularizations.functional` for functional implementations.
    - See `~linodenet.regularizations..modular` for modular implementations.
"""

__all__ = [
    # Sub-Modules
    "functional",
    # Constants
    "REGULARIZATIONS",
    "FUNCTIONAL_REGULARIZATIONS",
    "MODULAR_REGULARIZATIONS",
    # ABCs & Protocols
    "Regularization",
    "RegularizationABC",
    # Classes
    "Banded",
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

from linodenet.regularizations import functional
from linodenet.regularizations._regularizations import (
    Banded,
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
from linodenet.regularizations.functional import (
    Regularization,
    banded,
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

FUNCTIONAL_REGULARIZATIONS: dict[str, Regularization] = {
    "logdetexp": logdetexp,
    "matrix_norm": matrix_norm,
    # linodenet.projections (matrix groups)
    "hamiltonian": hamiltonian,
    "identity": identity,
    "low_rank": low_rank,
    "normal": normal,
    "orthogonal": orthogonal,
    "skew_symmetric": skew_symmetric,
    "symmetric": symmetric,
    "symplectic": symplectic,
    "traceless": traceless,
    # linodenet.projections (masked)
    "banded": banded,
    "diagonal": diagonal,
    "lower_triangular": lower_triangular,
    "masked": masked,
    "upper_triangular": upper_triangular,
}
r"""Dictionary of all available modular metrics."""

MODULAR_REGULARIZATIONS: dict[str, type[Regularization]] = {
    "LogDetExp": LogDetExp,
    "MatrixNorm": MatrixNorm,
    # matrix groups
    "Hamiltonian": Hamiltonian,
    "Identity": Identity,
    "LowRank": LowRank,
    "Normal": Normal,
    "Orthogonal": Orthogonal,
    "SkewSymmetric": SkewSymmetric,
    "Symmetric": Symmetric,
    "Symplectic": Symplectic,
    "Traceless": Traceless,
    # masked
    "Banded": Banded,
    "Diagonal": Diagonal,
    "LowerTriangular": LowerTriangular,
    "Masked": Masked,
    "UpperTriangular": UpperTriangular,
}
r"""Dictionary of all available modular metrics."""

REGULARIZATIONS: dict[str, Regularization | type[Regularization]] = {
    **FUNCTIONAL_REGULARIZATIONS,
    **MODULAR_REGULARIZATIONS,
}
r"""Dictionary containing all available projections."""
