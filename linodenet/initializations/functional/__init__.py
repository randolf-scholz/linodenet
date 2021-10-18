r"""Initializations for the Linear ODE Networks.

All initializations are normalized such that if `x~𝓝(0,1)`, then `Ax~𝓝(0,1)` as well.

Notes
-----
Contains initializations in functional form.
  - See :mod:`~.modular` for modular implementations.
"""

__all__ = [
    # Types
    "SizeLike",
    "FunctionalInitialization",
    # Constants
    "FunctionalInitializations",
    # Functions
    "gaussian",
    "symmetric",
    "skew_symmetric",
    "orthogonal",
    "special_orthogonal",
    "diagonally_dominant",
    "canonical_skew_symmetric",
]

import logging
from typing import Callable, Final

from torch import Tensor

from linodenet.initializations.functional._functional import (
    SizeLike,
    canonical_skew_symmetric,
    diagonally_dominant,
    gaussian,
    orthogonal,
    skew_symmetric,
    special_orthogonal,
    symmetric,
)

LOGGER = logging.getLogger(__name__)

FunctionalInitialization = Callable[[SizeLike], Tensor]  # SizeLike to matrix
r"""Type hint for Initializations."""

FunctionalInitializations: Final[dict[str, FunctionalInitialization]] = {
    "gaussian": gaussian,
    "symmetric": symmetric,
    "skew-symmetric": skew_symmetric,
    "orthogonal": orthogonal,
    "special-orthogonal": special_orthogonal,
    "diagonally_dominant": diagonally_dominant,
    "canonical_skew_symmetric": canonical_skew_symmetric,
}
r"""Dictionary containing all available initializations."""
