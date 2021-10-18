r"""Initializations for the Linear ODE Networks.

All initializations are normalized such that if `x~𝓝(0,1)`, then `Ax~𝓝(0,1)` as well.

Notes
-----
Contains initializations in modular form.
  - See :mod:`~.functional` for functional implementations.
"""

__all__ = [
    # Types
    "ModularInitialization",
    # Constants
    "ModularInitializations",
]

import logging
from typing import Final

from torch.nn import Module

__logger__ = logging.getLogger(__name__)

ModularInitialization = Module
r"""Type hint for modular regularizations."""

ModularInitializations: Final[dict[str, type[ModularInitialization]]] = {}
r"""Dictionary of all available modular metrics."""
