r"""Test utilities."""

__all__ = [
    "PROJECT",
    "DTYPES",
    "DEVICES",
    "PREFER_GPU",
    "SEEDS_10",
    "SEEDS_5",
    "SEED",
    # Classes
    "TestSuite",
    # Functions
    "camel2snake",
    "snake2camel",
    "visualize_distribution",
    "pytest_xfail",
    "timer",
]

from .assertions import TestSuite
from .constants import DEVICES, DTYPES, PREFER_GPU, SEED, SEEDS_5, SEEDS_10
from .misc import camel2snake, snake2camel
from .plotting import visualize_distribution
from .project import PROJECT
from .timer import timer
from .xfail import pytest_xfail
