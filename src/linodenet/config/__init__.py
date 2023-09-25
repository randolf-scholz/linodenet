r"""LinODE-Net Configuration."""

__all__ = [
    # CONSTANTS
    "CONFIG",
    "PROJECT",
    # Classes
    "Config",
    "Project",
    # Functions
    "generate_folders",
    "get_package_structure",
]

from typing import Final

from linodenet.config._config import (
    Config,
    Project,
    generate_folders,
    get_package_structure,
)

PROJECT: Final[Project] = Project()
"""Project configuration."""

CONFIG: Final[Config] = Config()
"""Configuration Class."""

del Final
