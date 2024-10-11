r"""Projections for the Linear ODE Networks.

Notes:
    Contains projections in both modular and functional form.
      - See `~linodenet.projections.functional` for functional implementations.
      - See `~linodenet.projections.modular` for modular implementations.
"""

__all__ = [
    # Sub-Modules
    "functional",
    "modular",
    # Constants
    "FUNCTIONAL_PROJECTIONS",
    "MODULAR_PROJECTIONS",
    "PROJECTIONS",
    # ABCs & Protocols
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
]


from linodenet.projections import functional, modular
from linodenet.projections.functional import (
    Projection,
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
from linodenet.projections.modular import (
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
    ProjectionBase,
    SkewSymmetric,
    Symmetric,
    Symplectic,
    Traceless,
    UpperTriangular,
)

FUNCTIONAL_PROJECTIONS: dict[str, Projection] = {
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

MODULAR_PROJECTIONS: dict[str, type[Projection]] = {
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

PROJECTIONS: dict[str, Projection | type[Projection]] = {
    **FUNCTIONAL_PROJECTIONS,
    **MODULAR_PROJECTIONS,
}
r"""Dictionary containing all available projections."""
