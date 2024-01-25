r"""Models for the latent dynamical system."""

__all__ = [
    # Constants
    "SYSTEMS",
    # ABCs & Protocols
    "System",
    "SystemABC",
    # Classes
    "LinODE",
    "LinODECell",
]

from linodenet.models.system.base import System, SystemABC
from linodenet.models.system.linode import LinODE, LinODECell

SYSTEMS: dict[str, type[System]] = {
    "LinODE": LinODE,
    "LinODECell": LinODECell,
}
r"""Dictionary of all available system components."""
