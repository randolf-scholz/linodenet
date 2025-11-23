r"""Encoder Models."""

__all__ = [
    # Constants
    "ENCODERS",
    "MLP",
    # ABCs & Protocols
    "Encoder",
    "EncoderABC",
    # Classes
    "ResNet",
    "ResNetBlock",
    "Transformer",
    "iResNet",
    "iResNetBlock",
]


from linodenet.bijections.iresnet import (
    iResNet,
    iResNetBlock,
)
from linodenet.components.encoders.base import Encoder, EncoderABC
from linodenet.components.encoders.mlp import MLP
from linodenet.components.encoders.resnet import ResNet, ResNetBlock
from linodenet.components.encoders.transformer import Transformer
from linodenet.layers.misc import Identity

ENCODERS: dict[str, type[Encoder]] = {
    "Identity": Identity,
    "MLP": MLP,
    "ResNet": ResNet,
    "Transformer": Transformer,
    "iResNet": iResNet,
}
r"""Dictionary containing all available encoders."""
