r"""Matrix and sphere-point initializations for linear ODE networks.

Matrix samplers use dimension-dependent scaling or structural constraints.
Their individual docstrings describe the distributions and invariants.

Notes:
    - See `imtskit.initializations.functional` for functional implementations.
    - See `imtskit.initializations.modules` for module wrappers.
"""
# ruff: file-ignore[F403]

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
    "resolve_kernel_initialization",
]

from . import functional, modules
from .base import Initialization, InitializationFn, resolve_kernel_initialization
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
r"""Registry of matrix initialization functions."""


INITIALIZATIONS: dict[str, type[Initialization]] = {
    "Fixed"              : modules.Fixed,
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
r"""Registry of initialization classes implemented as `torch.nn.Module` subclasses."""
