r"""LinODE-Net Configuration.

# TODO: There must be a better way to handle global config
"""

__all__ = [
    # Constants
    "conf",
    # Classes
    "Config",
]

import logging
import os

__logger__ = logging.getLogger(__name__)

os.environ["LINODENET_AUTOJIT"] = "True"
"""Default value."""


class Config:
    r"""Configuration Interface."""

    # TODO: Should be initialized by a init/toml file.
    _autojit: bool = True
    __name__ = __name__
    __file__ = __file__

    @property
    def autojit(self) -> bool:
        r"""Whether to automatically jit-compile the models."""
        return self._autojit

    @autojit.setter
    def autojit(self, value: bool) -> None:
        assert isinstance(value, bool)
        self._autojit = bool(value)
        os.environ["LINODENET_AUTOJIT"] = str(value)


conf: Config = Config()  # = Config(__name__, __doc__)
"""The unique :class:`~linodenet.config.Config` instance used to configure :mod:`linodenet`."""
