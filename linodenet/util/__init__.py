r"""Utility functions."""

__all__ = [
    # Types
    "Activation",
    # Constants
    "ACTIVATIONS",
    # Classes
    # Functions
    "autojit",
    "deep_dict_update",
    "deep_keyval_update",
    "flatten",
]

import logging

from linodenet.util.util import (  # Types; Constants; Classes; Functions
    ACTIVATIONS,
    Activation,
    autojit,
    deep_dict_update,
    deep_keyval_update,
    flatten,
)

LOGGER = logging.getLogger(__name__)
