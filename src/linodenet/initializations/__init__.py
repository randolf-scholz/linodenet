r"""Initializations for the Linear ODE Networks.

All initializations are normalized such that if $x∼𝓝(0,1)$, then $Ax∼𝓝(0,1)$ as well.

Lemma
~~~~~

In this case: $e^{A}$

Notes
-----
Contains initializations in both modular and functional form.
  - See `~linodenet.initializations.functional` for functional implementations.
"""

__all__ = [
    # Constants
    "INITIALIZATIONS",
    # Types
    "Initialization",
    # Sub-Modules
    "functional",
    # Functions
    "canonical_skew_symmetric",
    "diagonally_dominant",
    "gaussian",
    "low_rank",
    "orthogonal",
    "skew_symmetric",
    "special_orthogonal",
    "symmetric",
    "traceless",
    # Classes
]

from linodenet.initializations import functional
from linodenet.initializations.functional import (
    Initialization,
    canonical_skew_symmetric,
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
    "canonical_skew_symmetric": canonical_skew_symmetric,
    "diagonally_dominant": diagonally_dominant,
    "gaussian": gaussian,
    "low_rank": low_rank,
    "orthogonal": orthogonal,
    "skew-symmetric": skew_symmetric,
    "special-orthogonal": special_orthogonal,
    "symmetric": symmetric,
    "traceless": traceless,
}
r"""Dictionary containing all available initializations."""
