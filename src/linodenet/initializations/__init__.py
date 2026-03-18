r"""Initializations for the Linear ODE Networks.

All initializations are normalized such that if $x∼𝓝(0,1)$, then $Ax∼𝓝(0,1)$ as well.

Notes:
    - See `linodenet.initializations.functional` for functional implementations.
    - See `linodenet.initializations.modules` for all module-based initializations.
"""
# ruff: noqa: F403, F405

__all__ = [
    # Sub-Modules
    "functional",
    "modules",
    # Constants
    "INITIALIZATIONS",
]

from linodenet.initializations import functional, modules
from linodenet.initializations.functional import *

__all__ += functional.__all__

INITIALIZATIONS: dict[str, Initialization] = {
    "symplectic"          : functional.symplectic,
    "diagonally_dominant" : functional.diagonally_dominant,
    "gaussian"            : functional.gaussian,
    "low_rank"            : functional.low_rank,
    "orthogonal"          : functional.orthogonal,
    "skew_symmetric"      : functional.skew_symmetric,
    "special_orthogonal"  : functional.special_orthogonal,
    "symmetric"           : functional.symmetric,
    "traceless"           : functional.traceless,
}  # fmt: skip
r"""Dictionary containing all available initializations."""
