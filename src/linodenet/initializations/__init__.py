r"""Initializations for the Linear ODE Networks.

All initializations are normalized such that if $x∼𝓝(0,1)$, then $Ax∼𝓝(0,1)$ as well.

Notes:
    - See `linodenet.initializations.functional` for functional implementations.
    - See `linodenet.initializations.modules` for all module-based initializations.
"""

__all__ = [
    # Sub-Modules
    "functional",
    "modules",
    # Constants
    "INITIALIZATIONS",
    # ABCs & Protocols
    "Initialization",
    # Functions
    "canonical_skew_symmetric",
    "canonical_symplectic",
    "diagonally_dominant",
    "gaussian",
    "low_rank",
    "orthogonal",
    "skew_symmetric",
    "special_orthogonal",
    "symmetric",
    "traceless",
]

from linodenet.initializations import functional, modules
from linodenet.initializations.functional import (
    Initialization,
    canonical_skew_symmetric,
    canonical_symplectic,
    diagonally_dominant,
    gaussian,
    low_rank,
    orthogonal,
    skew_symmetric,
    special_orthogonal,
    symmetric,
    traceless,
)

INITIALIZATIONS: dict[str, Initialization] = {
    "canonical_skew_symmetric" : canonical_skew_symmetric,
    "canonical_symplectic"     : canonical_symplectic,
    "diagonally_dominant"      : diagonally_dominant,
    "gaussian"                 : gaussian,
    "low_rank"                 : low_rank,
    "orthogonal"               : orthogonal,
    "skew_symmetric"           : skew_symmetric,
    "special_orthogonal"       : special_orthogonal,
    "symmetric"                : symmetric,
    "traceless"                : traceless,
}  # fmt: skip
r"""Dictionary containing all available initializations."""
