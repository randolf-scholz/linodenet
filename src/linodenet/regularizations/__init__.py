r"""Regularizations for the Linear ODE Networks.

Notes
-----
Contains regularizations in both modular and functional form.
  - See `~linodenet.regularizations.functional` for functional implementations.
  - See `~linodenet.regularizations..modular` for modular implementations.
"""

__all__ = [
    # Constants
    "REGULARIZATIONS",
    "FUNCTIONAL_REGULARIZATIONS",
    "MODULAR_REGULARIZATIONS",
    # Types
    "Regularization",
    "FunctionalRegularization",
    "ModularRegularization",
    # Sub-Modules
    "functional",
    "modular",
    # Functions
    "diagonal",
    "logdetexp",
    "normal",
    "orthogonal",
    "skew_symmetric",
    "symmetric",
    # Classes
]


from collections.abc import Callable
from typing import Final, TypeAlias

from torch import Tensor, nn

from linodenet.regularizations import functional, modular
from linodenet.regularizations.functional import (
    diagonal,
    logdetexp,
    normal,
    orthogonal,
    skew_symmetric,
    symmetric,
)

FunctionalRegularization: TypeAlias = Callable[[Tensor], Tensor]
r"""Type hint for modular regularizations."""

ModularRegularization: TypeAlias = nn.Module
r"""Type hint for modular regularizations."""

Regularization: TypeAlias = FunctionalRegularization | ModularRegularization
r"""Type hint for projections."""

FUNCTIONAL_REGULARIZATIONS: Final[dict[str, FunctionalRegularization]] = {
    "diagonal": diagonal,
    "logdetexp": logdetexp,
    "normal": normal,
    "orthogonal": orthogonal,
    "skew_symmetric": skew_symmetric,
    "symmetric": symmetric,
}
r"""Dictionary of all available modular metrics."""

MODULAR_REGULARIZATIONS: Final[dict[str, type[nn.Module]]] = {}
r"""Dictionary of all available modular metrics."""

REGULARIZATIONS: Final[
    dict[str, FunctionalRegularization | type[ModularRegularization]]
] = {
    **FUNCTIONAL_REGULARIZATIONS,
    **MODULAR_REGULARIZATIONS,
}
r"""Dictionary containing all available projections."""
