r"""Regularizations for the Linear ODE Networks.

Notes
-----
Contains regularizations in both modular and functional form.
  - See :mod:`~linodenet.regularizations.functional` for functional implementations.
  - See :mod:`~linodenet.regularizations..modular` for modular implementations.
"""

__all__ = [
    # Types
    "Regularization",
    "FunctionalRegularization",
    "ModularRegularization",
    # Constants
    "REGULARIZATIONS",
    "FunctionalRegularizations",
    "ModularRegularizations",
    # Sub-Modules
    "functional",
    "modular",
]


import logging
from typing import Final, Union

from linodenet.regularizations import functional, modular
from linodenet.regularizations.functional import (
    FunctionalRegularization,
    FunctionalRegularizations,
)
from linodenet.regularizations.modular import (
    ModularRegularization,
    ModularRegularizations,
)

__logger__ = logging.getLogger(__name__)

Regularization = Union[
    FunctionalRegularization, ModularRegularization
]  # matrix to matrix
r"""Type hint for projections."""

REGULARIZATIONS: Final[
    dict[str, Union[FunctionalRegularization, type[ModularRegularization]]]
] = {
    **FunctionalRegularizations,
    **ModularRegularizations,
}
r"""Dictionary containing all available projections."""
