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
    # ABCs & Protocols
    "Surjection",
    "SurjectionBase",
    "FunctionalProjection",
    "Projection",
    "ProjectionBase",
    # Functions
    "banded",
    "contraction",
    "diagonal",
    "diagonally_dominant",
    "hamiltonian",
    "identity",
    "low_rank",
    "lower_triangular",
    "masked",
    "normal",
    "orthogonal",
    "skew_symmetric",
    "symmetric",
    "symplectic",
    "traceless",
    "upper_triangular",
    # Classes
    "Banded",
    "Contraction",
    "Diagonal",
    "DiagonallyDominant",
    "Hamiltonian",
    "Identity",
    "LowRank",
    "LowerTriangular",
    "Masked",
    "Normal",
    "Orthogonal",
    "SkewSymmetric",
    "Symmetric",
    "Symplectic",
    "Traceless",
    "UpperTriangular",
    # Surjections
    "ConcatProjection",
]


from linodenet.projections import functional, modules, surjections
from linodenet.projections.functional import (
    FunctionalProjection,
    banded,
    contraction,
    diagonal,
    diagonally_dominant,
    hamiltonian,
    identity,
    low_rank,
    lower_triangular,
    masked,
    normal,
    orthogonal,
    skew_symmetric,
    symmetric,
    symplectic,
    traceless,
    upper_triangular,
)
from linodenet.projections.modules import (
    Banded,
    Contraction,
    Diagonal,
    DiagonallyDominant,
    Hamiltonian,
    Identity,
    LowerTriangular,
    LowRank,
    Masked,
    Normal,
    Orthogonal,
    Projection,
    ProjectionBase,
    SkewSymmetric,
    Symmetric,
    Symplectic,
    Traceless,
    UpperTriangular,
)
from linodenet.projections.surjections import (
    ConcatProjection,
    Surjection,
    SurjectionBase,
)

FUNCTIONAL_PROJECTIONS: dict[str, FunctionalProjection] = {
    "banded"              : banded,
    "contraction"         : contraction,
    "diagonal"            : diagonal,
    "diagonally_dominant" : diagonally_dominant,
    "hamiltonian"         : hamiltonian,
    "identity"            : identity,
    "low_rank"            : low_rank,
    "lower_triangular"    : lower_triangular,
    "masked"              : masked,
    "normal"              : normal,
    "orthogonal"          : orthogonal,
    "skew_symmetric"      : skew_symmetric,
    "symmetric"           : symmetric,
    "symplectic"          : symplectic,
    "traceless"           : traceless,
    "upper_triangular"    : upper_triangular,
}  # fmt: skip
r"""Dictionary of all available modular metrics."""

MODULAR_PROJECTIONS: dict[str, type[ProjectionBase]] = {
    "Banded"             : Banded,
    "Contraction"        : Contraction,
    "Diagonal"           : Diagonal,
    "DiagonallyDominant" : DiagonallyDominant,
    "Hamiltonian"        : Hamiltonian,
    "Identity"           : Identity,
    "LowRank"            : LowRank,
    "LowerTriangular"    : LowerTriangular,
    "Masked"             : Masked,
    "Normal"             : Normal,
    "Orthogonal"         : Orthogonal,
    "SkewSymmetric"      : SkewSymmetric,
    "Symmetric"          : Symmetric,
    "Symplectic"         : Symplectic,
    "Traceless"          : Traceless,
    "UpperTriangular"    : UpperTriangular,
}  # fmt: skip
r"""Dictionary of all available modular metrics."""

PROJECTIONS: dict[str, FunctionalProjection | type[ProjectionBase]] = {
    **FUNCTIONAL_PROJECTIONS,
    **MODULAR_PROJECTIONS,
}
r"""Dictionary containing all available projections."""
