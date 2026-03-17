r"""Regularizations for the Linear ODE Networks.

Notes:
    - See `linodenet.regularizations.functional` for functional implementations.
    - See `linodenet.regularizations.modules` for module-based implementations.
"""

__all__ = [
    # Sub-Modules
    "functional",
    "modules",
    # Constants
    "REGULARIZATIONS",
    "FUNCTIONAL_REGULARIZATIONS",
    "MODULAR_REGULARIZATIONS",
]

from linodenet.regularizations import functional, modules
from linodenet.regularizations.functional import *
from linodenet.regularizations.modules import *

__all__ += functional.__all__
__all__ += modules.__all__

FUNCTIONAL_REGULARIZATIONS: dict[str, Regularization] = {
    "banded"           : functional.banded,
    "contraction"      : functional.contraction,
    "diagonal"         : functional.diagonal,
    "hamiltonian"      : functional.hamiltonian,
    "identity"         : functional.identity,
    "log_det_exp"      : functional.log_det_exp,
    "low_rank"         : functional.low_rank,
    "lower_triangular" : functional.lower_triangular,
    "masked"           : functional.masked,
    "matrix_norm"      : functional.matrix_norm,
    "normal"           : functional.normal,
    "orthogonal"       : functional.orthogonal,
    "skew_symmetric"   : functional.skew_symmetric,
    "symmetric"        : functional.symmetric,
    "symplectic"       : functional.symplectic,
    "traceless"        : functional.traceless,
    "upper_triangular" : functional.upper_triangular,
}  # fmt: skip
r"""Dictionary of all available modular metrics."""

MODULAR_REGULARIZATIONS: dict[str, type[Regularization]] = {
    "Banded"          : modules.Banded,
    "Contraction"     : modules.Contraction,
    "Diagonal"        : modules.Diagonal,
    "Hamiltonian"     : modules.Hamiltonian,
    "Identity"        : modules.Identity,
    "LogDetExp"       : modules.LogDetExp,
    "LowRank"         : modules.LowRank,
    "LowerTriangular" : modules.LowerTriangular,
    "Masked"          : modules.Masked,
    "MatrixNorm"      : modules.MatrixNorm,
    "Normal"          : modules.Normal,
    "Orthogonal"      : modules.Orthogonal,
    "SkewSymmetric"   : modules.SkewSymmetric,
    "Symmetric"       : modules.Symmetric,
    "Symplectic"      : modules.Symplectic,
    "Traceless"       : modules.Traceless,
    "UpperTriangular" : modules.UpperTriangular,
}  # fmt: skip
r"""Dictionary of all available modular metrics."""

REGULARIZATIONS: dict[str, Regularization | type[Regularization]] = {
    **FUNCTIONAL_REGULARIZATIONS,
    **MODULAR_REGULARIZATIONS,
}
r"""Dictionary containing all available projections."""
