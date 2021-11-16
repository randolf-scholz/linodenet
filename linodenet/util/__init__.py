r"""Utility functions."""

__all__ = [
    # Types
    "Activation",
    "LookupTable",
    # Constants
    "ACTIVATIONS",
    # Classes
    # Functions
    "autojit",
    "deep_dict_update",
    "deep_keyval_update",
    "flatten",
    "initialize_from",
]

import logging

from linodenet.util._util import (
    ACTIVATIONS,
    Activation,
    LookupTable,
    autojit,
    deep_dict_update,
    deep_keyval_update,
    flatten,
    initialize_from,
)

__logger__ = logging.getLogger(__name__)
