r"""Regularizations for the Linear ODE Networks.

Notes:
    - See `linodenet.regularizations.functional` for functional implementations.
    - See `linodenet.regularizations.modules` for module-based implementations.
"""
# ruff: noqa: F403, F405

__all__ = [
    # Sub-Modules
    "functional",
    "modules",
    # Constants
    "REGULARIZATION_FNS_WITH_ARGS",
    "REGULARIZATION_FNS_WITHOUT_ARGS",
    "REGULARIZATION_FNS",
    "REGULARIZATIONS",
]

from linodenet.regularizations import functional, modules
from linodenet.regularizations.functional import *
from linodenet.regularizations.modules import *

__all__ += functional.__all__
__all__ += modules.__all__

REGULARIZATION_FNS_WITHOUT_ARGS: dict[str, Regularization] = {
    "diagonal"              : functional.diagonal,
    "diagonally_dominant"   : functional.diagonally_dominant,
    "hamiltonian"           : functional.hamiltonian,
    "identity"              : functional.identity,
    "log_det_exp"           : functional.log_det_exp,
    "lower_triangular"      : functional.lower_triangular,
    "matrix_norm"           : functional.matrix_norm,
    "normal"                : functional.normal,
    "orthogonal"            : functional.orthogonal,
    "rank_one"              : functional.rank_one,
    "skew_symmetric"        : functional.skew_symmetric,
    "spectral_normalized"   : functional.spectral_normalized,
    "symmetric"             : functional.symmetric,
    "symplectic"            : functional.symplectic,
    "traceless"             : functional.traceless,
    "tridiagonal"           : functional.tridiagonal,
    "unit_vector"           : functional.unit_vector,
    "upper_triangular"      : functional.upper_triangular,
    "vector_norm"           : functional.vector_norm,
}  # fmt: skip
r"""Dictionary of all available regularizations (function)."""

REGULARIZATION_FNS_WITH_ARGS: dict[str, RegularizationWithArgs] = {
    "banded"           : functional.banded,
    "contraction"      : functional.contraction,
    "lipschitz_bounded": functional.lipschitz_bounded,
    "low_rank"         : functional.low_rank,
    "masked"           : functional.masked,
}  # fmt: skip
r"""Dictionary of all available regularizations (function)."""

REGULARIZATION_FNS: dict[str, Regularization | RegularizationWithArgs] = {
    **REGULARIZATION_FNS_WITHOUT_ARGS,
    **REGULARIZATION_FNS_WITH_ARGS,
}
r"""Dictionary of all available regularizations (function)."""

REGULARIZATIONS: dict[str, type[Regularization]] = {
    "Banded"               : modules.Banded,
    "Contraction"          : modules.Contraction,
    "Diagonal"             : modules.Diagonal,
    "DiagonallyDominant"   : modules.DiagonallyDominant,
    "Hamiltonian"          : modules.Hamiltonian,
    "Identity"             : modules.Identity,
    "LogDetExp"            : modules.LogDetExp,
    "LipschitzBounded"     : modules.LipschitzBounded,
    "LowRank"              : modules.LowRank,
    "LowerTriangular"      : modules.LowerTriangular,
    "Masked"               : modules.Masked,
    "MatrixNorm"           : modules.MatrixNorm,
    "Normal"               : modules.Normal,
    "Orthogonal"           : modules.Orthogonal,
    "RankOne"              : modules.RankOne,
    "SkewSymmetric"        : modules.SkewSymmetric,
    "SpectralNormalized"   : modules.SpectralNormalized,
    "Symmetric"            : modules.Symmetric,
    "Symplectic"           : modules.Symplectic,
    "Traceless"            : modules.Traceless,
    "Tridiagonal"          : modules.Tridiagonal,
    "UnitVector"           : modules.UnitVector,
    "UpperTriangular"      : modules.UpperTriangular,
}  # fmt: skip
r"""Dictionary of all available regularizations (nn.Module)."""
