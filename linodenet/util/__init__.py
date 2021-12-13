r"""Utility functions."""

__all__ = [
    # Types
    "Activation",
    "LookupTable",
    # Constants
    "ACTIVATIONS",
    # Classes
    "ReZero",
    # Functions
    "autojit",
    "deep_dict_update",
    "deep_keyval_update",
    "flatten",
    "initialize_from",
    "initialize_from_config",
]

import logging

from linodenet.util._util import (
    ACTIVATIONS,
    Activation,
    LookupTable,
    ReZero,
    autojit,
    deep_dict_update,
    deep_keyval_update,
    flatten,
    initialize_from,
    initialize_from_config,
)

__logger__ = logging.getLogger(__name__)
