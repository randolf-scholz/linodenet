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

from torch.nn import Module

__logger__ = logging.getLogger(__name__)

ModularRegularization = Module
r"""Type hint for modular regularizations."""

ModularRegularizations: Final[dict[str, type[ModularRegularization]]] = {}
r"""Dictionary of all available modular metrics."""
