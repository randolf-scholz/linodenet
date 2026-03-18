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
    "REGULARIZATIONS",
    "REGULARIZATION_FNS",
    "REGULARIZATION_MODULES",
    "SPECIAL_REGULARIZATIONS",
]

from linodenet.regularizations import functional, modules
from linodenet.regularizations.functional import *
from linodenet.regularizations.modules import *

__all__ += functional.__all__
__all__ += modules.__all__

REGULARIZATION_FNS: dict[str, Regularization] = {
    "diagonal"         : functional.diagonal,
    "hamiltonian"      : functional.hamiltonian,
    "identity"         : functional.identity,
    "log_det_exp"      : functional.log_det_exp,
    "lower_triangular" : functional.lower_triangular,
    "matrix_norm"      : functional.matrix_norm,
    "normal"           : functional.normal,
    "orthogonal"       : functional.orthogonal,
    "rank_one"         : functional.rank_one,
    "skew_symmetric"   : functional.skew_symmetric,
    "spectral_normalized": functional.spectral_normalized,
    "symmetric"        : functional.symmetric,
    "symplectic"       : functional.symplectic,
    "traceless"        : functional.traceless,
    "tridiagonal"      : functional.tridiagonal,
    "upper_triangular" : functional.upper_triangular,
}  # fmt: skip
r"""Dictionary of all available modular metrics."""


SPECIAL_REGULARIZATIONS = {
    "banded"           : functional.banded,
    "contraction"      : functional.contraction,
    "lipschitz_bounded": functional.lipschitz_bounded,
    "low_rank"         : functional.low_rank,
    "masked"           : functional.masked,
}  # fmt: skip
r"""Regularizations that require additional arguments."""


REGULARIZATION_MODULES: dict[str, type[Regularization]] = {
    "Banded"          : modules.Banded,
    "Contraction"     : modules.Contraction,
    "Diagonal"        : modules.Diagonal,
    "Hamiltonian"     : modules.Hamiltonian,
    "Identity"        : modules.Identity,
    "LogDetExp"       : modules.LogDetExp,
    "LipschitzBounded": modules.LipschitzBounded,
    "LowRank"         : modules.LowRank,
    "LowerTriangular" : modules.LowerTriangular,
    "Masked"          : modules.Masked,
    "MatrixNorm"      : modules.MatrixNorm,
    "Normal"          : modules.Normal,
    "Orthogonal"      : modules.Orthogonal,
    "RankOne"         : modules.RankOne,
    "SkewSymmetric"   : modules.SkewSymmetric,
    "SpectralNormalized": modules.SpectralNormalized,
    "Symmetric"       : modules.Symmetric,
    "Symplectic"      : modules.Symplectic,
    "Traceless"       : modules.Traceless,
    "Tridiagonal"     : modules.Tridiagonal,
    "UpperTriangular" : modules.UpperTriangular,
}  # fmt: skip
r"""Dictionary of all available modular metrics."""

REGULARIZATIONS = {
    **REGULARIZATION_FNS,
    **SPECIAL_REGULARIZATIONS,
    **REGULARIZATION_MODULES,
}
r"""Dictionary containing all available projections."""
