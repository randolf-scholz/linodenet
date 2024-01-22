r"""Utility functions."""

__all__ = [
    # Sub-Modules
    "config",
    "context",
    "layers",
    # Classes
    "Multiply",
    "Parallel",
    "ReZeroCell",
    "Repeat",
    "ReverseDense",
    "Series",
    # Functions
    "autojit",
    "deep_dict_update",
    "deep_keyval_update",
    "initialize_from_dict",
    "initialize_from_type",
    "is_dunder",
    "try_initialize_from_config",
]

from linodenet.utils import config, context, layers
from linodenet.utils._utils import (
    autojit,
    deep_dict_update,
    deep_keyval_update,
    initialize_from_dict,
    initialize_from_type,
    is_dunder,
    try_initialize_from_config,
)
from linodenet.utils.generic_layers import Multiply, Parallel, Repeat, Series
from linodenet.utils.layers import ReverseDense, ReZeroCell
