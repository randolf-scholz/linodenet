r"""Neural Network subpackage of LinodeNet.

This contains general-purpose building blocks for neural networks, such as layers, activations, and containers.
This does not contain timeseries-specific layers and forecasting models.
"""

__all__ = [
    # submodules/packages
    "activations",
    "containers",
    "embeddings",
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

from linodenet.nn import activations, containers, embeddings
from linodenet.nn.containers import ModuleMapping, ModuleSequence
from linodenet.nn.linear_contraction import LinearContraction
from linodenet.nn.mlp import MLP
from linodenet.nn.resnet import ResNet, ResNetBlock
from linodenet.nn.reverse_dense import ReverseDense
from linodenet.nn.rezero import ReZero, ReZeroResNet

__all__ += activations.__all__
__all__ += embeddings.__all__

# __all__ += bijections.__all__


# __all__ += filters.__all__
# __all__ += forecasting.__all__
# __all__ += imputation.__all__
# __all__ += layers.__all__
# __all__ += projections.__all__
# __all__ += flows.__all__
