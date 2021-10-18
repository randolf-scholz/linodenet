r"""LinODE-Net Configuration.

# TODO: There must be a better way to handle global config
"""

__all__ = [
    # Classes
    "Config",
    # Constants
    "conf",
]

import logging

from linodenet.config._config import Config, conf

LOGGER = logging.getLogger(__name__)
