r"""Models for the latent dynamical system."""

__all__ = [
    # Types
    "System",
    # Meta-Objects
    "SYSTEMS",
    # Classes
    "LinODECell",
]


from typing import Final

from torch import nn

from linodenet.models.system._system import LinODECell

System = nn.Module
"""Type hint for the system model."""


SYSTEMS: Final[dict[str, type[System]]] = {"LinODECell": LinODECell}
"""Dictionary of all available system models."""
