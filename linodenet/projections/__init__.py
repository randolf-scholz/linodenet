r"""Projection Mappings."""
from __future__ import annotations

import logging
from typing import Callable, Final

from torch import Tensor

from linodenet.projections.projections import (
    diagonal,
    identity,
    normal,
    orthogonal,
    skew_symmetric,
    symmetric,
)

LOGGER = logging.getLogger(__name__)

__all__: Final[list[str]] = [
    "Projection",
    "PROJECTIONS",
    "symmetric",
    "skew_symmetric",
    "orthogonal",
    "diagonal",
    "identity",
    "normal",
]

Projection = Callable[[Tensor], Tensor]  # matrix to matrix
PROJECTIONS: Final[dict[str, Projection]] = {
    "symmetric": symmetric,
    "skew-symmetric": skew_symmetric,
    "orthogonal": orthogonal,
    "diagonal": diagonal,
    "identity": identity,
    "normal": normal,
}
