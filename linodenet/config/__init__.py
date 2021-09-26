r"""LinODE-Net Configuration."""
# TODO: There must be a better way to handle global config

from __future__ import annotations

import logging
import os
import sys
from types import ModuleType
from typing import Final

logger = logging.getLogger(__name__)
__all__: Final[list[str]] = []


class Config(ModuleType):
    r"""Configuration Interface."""
    # TODO: Should be initialized by a init/toml file.

    _autojit: bool = True
    __name__ = __name__
    __file__ = __file__
    # ['__annotations__', '__builtins__', '__cached__', '__doc__', '__file__',
    # '__loader__', '__name__', '__package__', '__spec__']

    @property
    def autojit(cls) -> bool:
        r"""If true, auto-compile all functions with jit."""
        return cls._autojit

    @autojit.setter
    def autojit(cls, value: bool):
        assert isinstance(value, bool)
        cls._autojit = bool(value)
        print(__name__)
        os.environ["LINODENET_AUTOJIT"] = str(value)


sys.modules[__name__] = Config(__name__, __doc__)
