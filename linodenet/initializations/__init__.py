r"""Initializations for the Linear ODE Networks."""

import logging
from typing import Callable, Final

from torch import Tensor

from linodenet.initializations.initializations import (
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

__all__: Final[list[str]] = [
    "INITIALIZATIONS",
    "Initialization",
    "SizeLike",
    "gaussian",
    "symmetric",
    "skew_symmetric",
    "orthogonal",
    "special_orthogonal",
    "diagonally_dominant",
    "canonical_skew_symmetric",
]

Initialization = Callable[[SizeLike], Tensor]  # SizeLike to matrix
r"""Type hint for Initializations."""

INITIALIZATIONS: Final[dict[str, Initialization]] = {
    "gaussian": gaussian,
    "symmetric": symmetric,
    "skew-symmetric": skew_symmetric,
    "orthogonal": orthogonal,
    "special-orthogonal": special_orthogonal,
    "diagonally_dominant": diagonally_dominant,
    "canonical_skew_symmetric": canonical_skew_symmetric,
}
r"""Dictionary containing all available initializations."""
