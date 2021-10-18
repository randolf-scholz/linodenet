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

from linodenet.util._util import (  # Types; Constants; Classes; Functions
    ACTIVATIONS,
    Activation,
    autojit,
    deep_dict_update,
    deep_keyval_update,
    flatten,
)

__logger__ = logging.getLogger(__name__)
