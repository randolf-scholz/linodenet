r"""Models for the latent dynamical system."""

__all__ = [
    # Types
    "System",
    "SystemABC",
    # Meta-Objects
    "SYSTEMS",
    # Classes
    "LinODECell",
]

from linodenet.models.system._system import LinODECell, System, SystemABC

SYSTEMS: dict[str, type[System]] = {
    "LinODECell": LinODECell,
}
r"""Dictionary of all available system components."""
