r"""Utility functions."""

__all__ = [
    # Sub-Modules
    "config",
    "context",
    # Functions
    "deep_dict_update",
    "initialize_from_dict",
    "initialize_from_type",
    "is_dunder",
    "pad",
    "try_initialize_from_config",
]

from linodenet.utils import config, context
from linodenet.utils._utils import (
    deep_dict_update,
    initialize_from_dict,
    initialize_from_type,
    is_dunder,
    pad,
    try_initialize_from_config,
)
