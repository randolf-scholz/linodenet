r"""General-purpose neural network building blocks.

This subpackage provides layers, activations, and module containers.
Time-series-specific layers and forecasting models live in other subpackages.
"""
# ruff: file-ignore[E402, F403]

__all__ = [
    # submodules/packages
    "activations",
    "containers",
    "parametrize",
    # "embeddings",
    # "projections",
    # "surjections",
    # "bijections",
    # base classes & protocols
    "ModuleSequence",
    "ModuleMapping",
    # classes
    "MLP",
    "ResNet",
    "ReZero",
]

from . import activations, containers, parametrize
from .containers import ModuleMapping, ModuleSequence
from .mlp import MLP
from .resnet import ResNet
from .rezero import ReZero

# blocker statement to prevent formatter from changing import order.
# activations must be imported last.
pass  # ruff: ignore[PIE790]

from .activations import *

__all__ += activations.__all__
