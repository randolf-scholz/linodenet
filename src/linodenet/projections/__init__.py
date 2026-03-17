r"""Projections for the Linear ODE Networks.

Notes:
    - See `linodenet.projections.functional` for functional implementations.
    - See `linodenet.projections.modules` for module-based implementations.
"""

__all__ = [
    # Sub-Modules
    "functional",
    "modules",
    "surjections",
    # Constants
    "FUNCTIONAL_PROJECTIONS",
    "MODULAR_PROJECTIONS",
    "PROJECTIONS",
]

from linodenet.projections import functional, modules, surjections
from linodenet.projections.functional import *
from linodenet.projections.modules import *
from linodenet.projections.surjections import *

__all__ += functional.__all__
__all__ += modules.__all__
__all__ += surjections.__all__


FUNCTIONAL_PROJECTIONS: dict[str, FunctionalProjection] = {
    "banded"              : functional.banded,
    "contraction"         : functional.contraction,
    "diagonal"            : functional.diagonal,
    "diagonally_dominant" : functional.diagonally_dominant,
    "hamiltonian"         : functional.hamiltonian,
    "identity"            : functional.identity,
    "low_rank"            : functional.low_rank,
    "lower_triangular"    : functional.lower_triangular,
    "masked"              : functional.masked,
    "normal"              : functional.normal,
    "orthogonal"          : functional.orthogonal,
    "rank_one"            : functional.rank_one,
    "skew_symmetric"      : functional.skew_symmetric,
    "symmetric"           : functional.symmetric,
    "symplectic"          : functional.symplectic,
    "traceless"           : functional.traceless,
    "tridiagonal"         : functional.tridiagonal,
    "upper_triangular"    : functional.upper_triangular,
}  # fmt: skip
r"""Dictionary of all available modular metrics."""

MODULAR_PROJECTIONS: dict[str, type[ProjectionBase]] = {
    "Banded"             : modules.Banded,
    "Contraction"        : modules.Contraction,
    "Diagonal"           : modules.Diagonal,
    "DiagonallyDominant" : modules.DiagonallyDominant,
    "Hamiltonian"        : modules.Hamiltonian,
    "Identity"           : modules.Identity,
    "LowRank"            : modules.LowRank,
    "LowerTriangular"    : modules.LowerTriangular,
    "Masked"             : modules.Masked,
    "Normal"             : modules.Normal,
    "Orthogonal"         : modules.Orthogonal,
    "RankOne"            : modules.RankOne,
    "SkewSymmetric"      : modules.SkewSymmetric,
    "Symmetric"          : modules.Symmetric,
    "Symplectic"         : modules.Symplectic,
    "Traceless"          : modules.Traceless,
    "Tridiagonal"        : modules.Tridiagonal,
    "UpperTriangular"    : modules.UpperTriangular,
}  # fmt: skip
r"""Dictionary of all available modular metrics."""

PROJECTIONS: dict[str, FunctionalProjection | type[ProjectionBase]] = {
    **FUNCTIONAL_PROJECTIONS,
    **MODULAR_PROJECTIONS,
}
r"""Dictionary containing all available projections."""
