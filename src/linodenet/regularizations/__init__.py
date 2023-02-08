r"""Regularizations for the Linear ODE Networks.

Notes
-----
Contains regularizations in both modular and functional form.
  - See `~linodenet.regularizations.functional` for functional implementations.
  - See `~linodenet.regularizations..modular` for modular implementations.
"""

__all__ = [
    # Constants
    "REGULARIZATIONS",
    "FUNCTIONAL_REGULARIZATIONS",
    "MODULAR_REGULARIZATIONS",
    # Types
    "Regularization",
    "RegularizationABC",
    # Sub-Modules
    "functional",
    "modular",
    # Functions
    "banded",
    "diagonal",
    "identity",
    "logdetexp",
    "masked",
    "matrix_norm",
    "normal",
    "orthogonal",
    "skew_symmetric",
    "symmetric",
    # Classes
    "Banded",
    "Diagonal",
    "Identity",
    "LogDetExp",
    "Masked",
    "Normal",
    "Orthogonal",
    "SkewSymmetric",
    "Symmetric",
]

from linodenet.regularizations import functional, modular
from linodenet.regularizations._regularizations import Regularization, RegularizationABC
from linodenet.regularizations.functional import (
    banded,
    diagonal,
    identity,
    logdetexp,
    masked,
    matrix_norm,
    normal,
    orthogonal,
    skew_symmetric,
    symmetric,
)
from linodenet.regularizations.modular import (
    Banded,
    Diagonal,
    Identity,
    LogDetExp,
    Masked,
    MatrixNorm,
    Normal,
    Orthogonal,
    SkewSymmetric,
    Symmetric,
)

FUNCTIONAL_REGULARIZATIONS: dict[str, Regularization] = {
    "banded": banded,
    "diagonal": diagonal,
    "identity": identity,
    "logdetexp": logdetexp,
    "masked": masked,
    "matrix_norm": matrix_norm,
    "normal": normal,
    "orthogonal": orthogonal,
    "skew_symmetric": skew_symmetric,
    "symmetric": symmetric,
}
r"""Dictionary of all available modular metrics."""

MODULAR_REGULARIZATIONS: dict[str, type[Regularization]] = {
    "Banded": Banded,
    "Diagonal": Diagonal,
    "Identity": Identity,
    "LogDetExp": LogDetExp,
    "Masked": Masked,
    "MatrixNorm": MatrixNorm,
    "Normal": Normal,
    "Orthogonal": Orthogonal,
    "SkewSymmetric": SkewSymmetric,
    "Symmetric": Symmetric,
}
r"""Dictionary of all available modular metrics."""

REGULARIZATIONS: dict[str, Regularization | type[Regularization]] = {
    **FUNCTIONAL_REGULARIZATIONS,
    **MODULAR_REGULARIZATIONS,
}
r"""Dictionary containing all available projections."""
