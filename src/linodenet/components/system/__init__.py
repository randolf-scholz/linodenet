r"""Models for the latent dynamical system."""

__all__ = [
    # Constants
    "SYSTEMS",
    # ABCs & Protocols
    "ContinuousSystem",
    "SystemABC",
    # Classes
    "LinODE",
    "LinODECell",
]

from linodenet.components.system.base import ContinuousSystem, SystemABC
from linodenet.components.system.linode import LinODE, LinODECell

SYSTEMS: dict[str, type[ContinuousSystem]] = {
    "LinODE": LinODE,
    "LinODECell": LinODECell,
}
r"""Dictionary of all available system components."""
