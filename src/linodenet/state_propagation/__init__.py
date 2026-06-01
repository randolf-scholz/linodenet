r"""State propagation interfaces and implementations.

This package exposes the common propagation abstractions together with the
available flow-based implementations.
"""
# ruff: noqa: F403, F405

__all__ = [
    # submodules
    "base",
    "linear",
    # Constants
    "FLOWS",
]

from . import base, linear
from .base import *
from .linear import *

__all__ += base.__all__
__all__ += linear.__all__

FLOWS: dict[str, type[Flow]] = {
    "LinearFlow" : LinearFlow,
    "LinearGaussianFlow" : LinearGaussianFlow
}  # fmt: skip
r"""Dictionary of all avaible flows."""
