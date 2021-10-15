r"""Regularizations for LinODE kernel matrix."""

__all__ = [
    # Types
    "Regularization",
    # Constants
    "REGULARIZATIONS",
    # Functions
    "diagonal",
    "logdetexp",
    "normal",
    "orthogonal",
    "skew_symmetric",
    "symmetric",
]

import logging
from typing import Callable, Final

from torch import Tensor

from linodenet.regularizations.funcional import (
    diagonal,
    logdetexp,
    normal,
    orthogonal,
    skew_symmetric,
    symmetric,
)

LOGGER = logging.getLogger(__name__)

Regularization = Callable[[Tensor], Tensor]  # matrix to scalar
r"""Type hint for regularizations."""

REGULARIZATIONS: Final[dict[str, Regularization]] = {
    "diagonal": diagonal,
    "logdetexp": logdetexp,
    "normal": normal,
    "orthogonal": orthogonal,
    "symmetric": symmetric,
    "skew_symmetric": skew_symmetric,
}
r"""Dictionary containing all available regularizations."""
