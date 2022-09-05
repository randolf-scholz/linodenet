r"""Utility functions."""

__all__ = [
    # Sub-Modules
    "layers",
    # Functions
    "autojit",
    "deep_dict_update",
    "deep_keyval_update",
    "flatten",
    "initialize_from",
    "initialize_from_config",
    # Classes
    "ReZeroCell",
    "ReverseDense",
    "Repeat",
    "Series",
    "Parallel",
    "Multiply",
]
from linodenet.util import layers
from linodenet.util._util import (
    autojit,
    deep_dict_update,
    deep_keyval_update,
    flatten,
    initialize_from,
    initialize_from_config,
)
from linodenet.util.generic_layers import Multiply, Parallel, Repeat, Series
from linodenet.util.layers import ReverseDense, ReZeroCell
