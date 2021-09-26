r"""Provides utility functions.

linodenet.util
==============

.. data:: ACTIVATIONS

    A :class:`dict` containing string names as keys and corresponding :mod:`torch` functions.
"""
import logging
from typing import Final

from linodenet.util.util import (
    ACTIVATIONS,
    Activation,
    autojit,
    deep_dict_update,
    deep_keyval_update,
)

LOGGER = logging.getLogger(__name__)

__all__: Final[list[str]] = [  # Meta-objects
    "Activation",
    "ACTIVATIONS",
] + [  # Functions
    "autojit",
    "deep_dict_update",
    "deep_keyval_update",
]
