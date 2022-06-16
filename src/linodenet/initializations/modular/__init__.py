r"""Initializations for the Linear ODE Networks.

All initializations are normalized such that if `x∼𝓝(0,1)`, then `Ax∼𝓝(0,1)` as well.

Notes
-----
Contains initializations in modular form.
  - See :mod:`~linodenet.initializations.initializations.functional` for functional implementations.
"""

__all__ = [
    # Types
    "ModularInitialization",
    # Constants
    "ModularInitializations",
]

from typing import Final

from torch.nn import Module

ModularInitialization = Module
r"""Type hint for modular regularizations."""

ModularInitializations: Final[dict[str, type[ModularInitialization]]] = {}
r"""Dictionary of all available modular metrics."""
