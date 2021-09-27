r"""LinODE-Net Configuration."""
# TODO: There must be a better way to handle global config

from __future__ import annotations

__all__ = ["GlobalConfig"]

import logging
import os
from types import ModuleType

logger = logging.getLogger(__name__)


class Config(ModuleType):
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
    def autojit(self, value: bool):
        assert isinstance(value, bool)
        self._autojit = bool(value)
        print(__name__)
        os.environ["LINODENET_AUTOJIT"] = str(value)


GlobalConfig = Config(__name__, __doc__)
