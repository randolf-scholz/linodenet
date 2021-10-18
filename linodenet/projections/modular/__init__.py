r"""Projections for the Linear ODE Networks.

Notes
-----
Contains projections in modular form.
  - See :mod:`~.functional` for functional implementations.
"""

__all__ = [
    # Types
    "ModularProjection",
    # Constants
    "ModularProjections",
]

import logging
from typing import Final

from torch.nn import Module

__logger__ = logging.getLogger(__name__)

ModularProjection = Module
r"""Type hint for modular regularizations."""

ModularProjections: Final[dict[str, type[ModularProjection]]] = {}
r"""Dictionary of all available modular metrics."""
