r"""Projections for the Linear ODE Networks.

Notes
-----
Contains projections in both modular and functional form.
  - See :mod:`~linodenet.projections.functional` for functional implementations.
  - See :mod:`~linodenet.projections.modular` for modular implementations.
"""


__all__ = [
    # Types
    "Projection",
    "FunctionalProjection",
    "ModularProjection",
    # Constants
    "PROJECTIONS",
    "FunctionalProjections",
    "ModularProjections",
    # Sub-Modules
    "functional",
    "modular",
]


import logging
from typing import Final, Union

from linodenet.projections import functional, modular
from linodenet.projections.functional import FunctionalProjection, FunctionalProjections
from linodenet.projections.modular import ModularProjection, ModularProjections

__logger__ = logging.getLogger(__name__)

Projection = Union[FunctionalProjection, ModularProjection]  # matrix to matrix
r"""Type hint for projections."""

PROJECTIONS: Final[dict[str, Union[FunctionalProjection, type[ModularProjection]]]] = {
    **FunctionalProjections,
    **ModularProjections,
}
r"""Dictionary containing all available projections."""
