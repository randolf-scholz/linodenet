r"""Regularizations for the Linear ODE Networks.

Notes
-----
Contains regularizations in both modular and functional form.
  - See `~linodenet.regularizations.functional` for functional implementations.
  - See `~linodenet.regularizations..modular` for modular implementations.
"""

__all__ = [
    # Protocols
    "Regularization",
    # Base Classes
    "RegularizationABC",
    # Constants
    "REGULARIZATIONS",
    "FUNCTIONAL_REGULARIZATIONS",
    "MODULAR_REGULARIZATIONS",
    # Types
    # Sub-Modules
    "functional",
    "modular",
    # Functions
    "logdetexp",
    "matrix_norm",
    # linodenet.projections (matrix groups)
    "hamiltonian",
    "identity",
    "low_rank",
    "normal",
    "orthogonal",
    "skew_symmetric",
    "symmetric",
    "symplectic",
    "traceless",
    # linodenet.projections (masked)
    "banded",
    "diagonal",
    "lower_triangular",
    "masked",
    "upper_triangular",
    # Classes
    "LogDetExp",
    "MatrixNorm",
    # matrix groups
    "Hamiltonian",
    "Identity",
    "LowRank",
    "Normal",
    "Orthogonal",
    "SkewSymmetric",
    "Symmetric",
    "Symplectic",
    "Traceless",
    # masked
    "Banded",
    "Diagonal",
    "Masked",
    "LowerTriangular",
    "UpperTriangular",
]

from linodenet.regularizations import functional, modular
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
from linodenet.regularizations.modular import (
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
    "Masked": Masked,
    "LowerTriangular": LowerTriangular,
    "UpperTriangular": UpperTriangular,
}
r"""Dictionary of all available modular metrics."""

REGULARIZATIONS: dict[str, Regularization | type[Regularization]] = {
    **FUNCTIONAL_REGULARIZATIONS,
    **MODULAR_REGULARIZATIONS,
}
r"""Dictionary containing all available projections."""
