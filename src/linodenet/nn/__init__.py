r"""Neural Network subpackage of LinodeNet.

This contains general-purpose building blocks for neural networks, such as layers, activations, and containers.
This does not contain timeseries-specific layers and forecasting models.
"""
# ruff: noqa: E402, F403

__all__ = [
    # submodules/packages
    "activations",
    "containers",
    "embeddings",
    "projections",
    "surjections",
    "bijections",
    # base classes & protocols
    "ModuleSequence",
    "ModuleMapping",
    # classes
    "MLP",
    "ResNet",
    "ResNetBlock",
    "ReZero",
    "ReZeroResNet",
    "ReverseDense",
    "LinearContraction",
]

from linodenet.nn import activations, containers
from linodenet.nn.containers import ModuleMapping, ModuleSequence
from linodenet.nn.linear_contraction import LinearContraction
from linodenet.nn.mlp import MLP
from linodenet.nn.resnet import ResNet, ResNetBlock
from linodenet.nn.reverse_dense import ReverseDense
from linodenet.nn.rezero import ReZero, ReZeroResNet

pass  # noqa: PIE790

from linodenet.mappings import (
    bijections,
    embeddings,
    projections,
    surjections,
)
from linodenet.mappings.bijections import *
from linodenet.mappings.embeddings import *
from linodenet.mappings.projections import *
from linodenet.mappings.surjections import *
from linodenet.nn.activations import *

__all__ += activations.__all__
__all__ += embeddings.__all__
__all__ += surjections.__all__
__all__ += projections.__all__
__all__ += bijections.__all__
