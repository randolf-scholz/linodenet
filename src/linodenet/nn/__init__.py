r"""Neural Network subpackage of LinodeNet.

This contains general-purpose building blocks for neural networks, such as layers, activations, and containers.
This does not contain timeseries-specific layers and forecasting models.
"""
# ruff: noqa: E402, F403

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
    "ReZeroResNet",
    "ReverseDense",
]

from . import activations, containers, parametrize
from .containers import ModuleMapping, ModuleSequence
from .mlp import MLP
from .resnet import ResNet
from .reverse_dense import ReverseDense
from .rezero import ReZero, ReZeroResNet

pass  # noqa: PIE790

from .activations import *

__all__ += activations.__all__

# public re-export
# from linodenet.mappings import (
#     bijections,
#     embeddings,
#     projections,
#     surjections,
# )
# from linodenet.mappings.bijections import *
# from linodenet.mappings.embeddings import *
# from linodenet.mappings.projections import *
# from linodenet.mappings.surjections import *
#
# __all__ += embeddings.__all__
# __all__ += surjections.__all__
# __all__ += projections.__all__
# __all__ += bijections.__all__
