r"""Projections for the Linear ODE Networks.

Notes
-----
Contains projections in modular form.
  - See :mod:`~linodenet.projections.functional` for functional implementations.
"""

__all__ = [
    # Types
    "ModularProjection",
    # Constants
    "ModularProjections",
]

import logging
from typing import Final

from torch import nn

__logger__ = logging.getLogger(__name__)

ModularProjection = nn.Module
r"""Type hint for modular regularizations."""

ModularProjections: Final[dict[str, type[nn.Module]]] = {}
r"""Dictionary of all available modular metrics."""
