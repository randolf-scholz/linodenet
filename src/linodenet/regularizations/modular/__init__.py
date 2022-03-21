r"""Regularizations for the Linear ODE Networks.

Notes
-----
Contains regularizations in modular  form.
  - See :mod:`~linodenet.regularizations.functional` for functional implementations.
"""


__all__ = [
    # Types
    "ModularRegularization",
    # Constants
    "ModularRegularizations",
]

import logging
from typing import Final

from torch import nn

__logger__ = logging.getLogger(__name__)

ModularRegularization = nn.Module
r"""Type hint for modular regularizations."""

ModularRegularizations: Final[dict[str, type[nn.Module]]] = {}
r"""Dictionary of all available modular metrics."""
