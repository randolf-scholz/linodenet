r"""Projections for the Linear ODE Networks.

Notes
-----
Contains projections in functional form.
  - See :mod:`~linodenet.projections.modular` for modular implementations.
"""

__all__ = [
    # Types
    "FunctionalProjection",
    # Constants
    "FunctionalProjections",
    # Functions
    "identity",
    "symmetric",
    "skew_symmetric",
    "orthogonal",
    "diagonal",
    "normal",
]

from collections.abc import Callable
from typing import Final

from torch import Tensor

from linodenet.projections.functional._functional import (
    diagonal,
    identity,
    normal,
    orthogonal,
    skew_symmetric,
    symmetric,
)

FunctionalProjection = Callable[[Tensor], Tensor]
r"""Type hint for modular regularizations."""

FunctionalProjections: Final[dict[str, FunctionalProjection]] = {
    "identity": identity,
    "symmetric": symmetric,
    "skew_symmetric": skew_symmetric,
    "orthogonal": orthogonal,
    "diagonal": diagonal,
    "normal": normal,
}
r"""Dictionary of all available modular metrics."""
