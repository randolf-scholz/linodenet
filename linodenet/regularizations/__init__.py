r"""Regularizations for LinODE kernel matrix."""

from __future__ import annotations

import logging
from typing import Callable, Final

from torch import Tensor

from linodenet.regularizations.regularizations import (
    diagonal,
    logdetexp,
    normal,
    orthogonal,
    skew_symmetric,
    symmetric,
)

logger = logging.getLogger(__name__)

__all__: Final[list[str]] = [
    "REGULARIZATIONS",
    "Regularization",
    "diagonal",
    "logdetexp",
    "normal",
    "orthogonal",
    "skew_symmetric",
    "symmetric",
]

Regularization = Callable[[Tensor], Tensor]  # matrix to scalar

REGULARIZATIONS: Final[dict[str, Regularization]] = {
    "diagonal": diagonal,
    "logdetexp": logdetexp,
    "normal": normal,
    "orthogonal": orthogonal,
    "symmetric": symmetric,
    "skew_symmetric": skew_symmetric,
}
