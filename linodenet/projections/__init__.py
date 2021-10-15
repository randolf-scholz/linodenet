r"""Projection Mappings."""

__all__ = [
    # Types
    "Projection",
    # Constants
    "PROJECTIONS",
    # Functions
    "symmetric",
    "skew_symmetric",
    "orthogonal",
    "diagonal",
    "identity",
    "normal",
]


import logging
from typing import Callable, Final

from torch import Tensor

from linodenet.projections.functional import (
    diagonal,
    identity,
    normal,
    orthogonal,
    skew_symmetric,
    symmetric,
)

LOGGER = logging.getLogger(__name__)


Projection = Callable[[Tensor], Tensor]  # matrix to matrix
r"""Type hint for projections."""

PROJECTIONS: Final[dict[str, Projection]] = {
    "symmetric": symmetric,
    "skew-symmetric": skew_symmetric,
    "orthogonal": orthogonal,
    "diagonal": diagonal,
    "identity": identity,
    "normal": normal,
}
r"""Dictionary containing all available projections."""
