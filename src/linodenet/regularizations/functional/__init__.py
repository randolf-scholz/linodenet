r"""Regularizations for the Linear ODE Networks.

Notes
-----
Contains regularizations in functional form.
  - See :mod:`~linodenet.regularizations.modular` for modular implementations.
"""


__all__ = [
    # Types
    "FunctionalRegularization",
    # Constants
    "FunctionalRegularizations",
    # Functions
    "diagonal",
    "logdetexp",
    "normal",
    "orthogonal",
    "skew_symmetric",
    "symmetric",
]

from collections.abc import Callable
from typing import Final

from torch import Tensor

from linodenet.regularizations.functional._functional import (
    diagonal,
    logdetexp,
    normal,
    orthogonal,
    skew_symmetric,
    symmetric,
)

FunctionalRegularization = Callable[[Tensor], Tensor]
r"""Type hint for modular regularizations."""

FunctionalRegularizations: Final[dict[str, FunctionalRegularization]] = {
    "diagonal": diagonal,
    "logdetexp": logdetexp,
    "normal": normal,
    "orthogonal": orthogonal,
    "skew_symmetric": skew_symmetric,
    "symmetric": symmetric,
}
r"""Dictionary of all available modular metrics."""
