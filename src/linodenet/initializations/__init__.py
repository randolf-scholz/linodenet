r"""Initializations for the Linear ODE Networks.

All initializations are normalized such that if $x∼𝓝(0,1)$, then $Ax∼𝓝(0,1)$ as well.

Notes:
    - See `linodenet.initializations.functional` for functional implementations.
    - See `linodenet.initializations.modules` for all module-based initializations.
"""
# ruff: noqa: F403

__all__ = [
    # Sub-Modules
    "functional",
    "modules",
    # Constants
    "INITIALIZATION_FNS",
    "INITIALIZATIONS",
    # protocols
    "InitializationFn",
    "Initialization",
    # extra
    "thomson_initialization",
    "wide_angle_sphere_init",
]

from . import functional, modules
from .base import Initialization, InitializationFn
from .functional import *
from .modules import *
from .thomson_initialization import thomson_initialization, wide_angle_sphere_init

__all__ += functional.__all__
__all__ += modules.__all__

INITIALIZATION_FNS: dict[str, InitializationFn] = {
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
r"""Dictionary containing all available initializations (functions)."""


INITIALIZATIONS: dict[str, type[Initialization]] = {
    "Constant"           : modules.Constant,
    "DiagonallyDominant" : modules.DiagonallyDominant,
    "Gaussian"           : modules.Gaussian,
    "LowRank"            : modules.LowRank,
    "Orthogonal"         : modules.Orthogonal,
    "SkewSymmetric"      : modules.SkewSymmetric,
    "SpecialOrthogonal"  : modules.SpecialOrthogonal,
    "Symmetric"          : modules.Symmetric,
    "Symplectic"         : modules.Symplectic,
    "Traceless"          : modules.Traceless,
}  # fmt: skip
r"""Dictionary containing all available initializations (nn.Modules)"""
