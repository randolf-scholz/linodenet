r"""Utility functions."""

__all__ = [
    # Sub-Modules
    "layers",
    # Types
    "Activation",
    "LookupTable",
    # Constants
    "ACTIVATIONS",
    # Classes
    "ReZero",
    "ReverseDense",
    "Repeat",
    "Series",
    "Parallel",
    "Multiply",
    # Functions
    "autojit",
    "deep_dict_update",
    "deep_keyval_update",
    "flatten",
    "initialize_from",
    "initialize_from_config",
]

import logging

from linodenet.util import layers
from linodenet.util._util import (
    ACTIVATIONS,
    Activation,
    LookupTable,
    autojit,
    deep_dict_update,
    deep_keyval_update,
    flatten,
    initialize_from,
    initialize_from_config,
)
from linodenet.util.generic_layers import Multiply, Parallel, Repeat, Series
from linodenet.util.layers import ReverseDense, ReZero

__logger__ = logging.getLogger(__name__)
