r"""Initializations for the Linear ODE Networks.

All initializations are normalized such that if `x~𝓝(0,1)`, then `Ax~𝓝(0,1)` as well.

Notes
-----
Contains initializations in both modular and functional form.
  - See :mod:`~.functional` for functional implementations.
  - See :mod:`~.modular` for modular implementations.
"""

__all__ = [
    # Types
    "Initialization",
    "FunctionalInitialization",
    "ModularInitialization",
    # Constants
    "FunctionalInitializations",
    "ModularInitializations",
    "INITIALIZATIONS",
    # Sub-Modules
    "functional",
    "modular",
]

import logging
from typing import Final, Union

from linodenet.initializations import functional, modular
from linodenet.initializations.functional import (
    FunctionalInitialization,
    FunctionalInitializations,
)
from linodenet.initializations.modular import (
    ModularInitialization,
    ModularInitializations,
)

Initialization = Union[FunctionalInitialization, ModularInitialization]
r"""Type hint for initializations."""

INITIALIZATIONS: Final[
    dict[str, Union[FunctionalInitialization, type[ModularInitialization]]]
] = {
    **FunctionalInitializations,
    **ModularInitializations,
}
r"""Dictionary of all available initializations."""

LOGGER = logging.getLogger(__name__)
