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


from linodenet.bijections.iresnet import iResNet, iResNetBlock
from linodenet.encoders.base import Encoder, EncoderABC
from linodenet.encoders.mlp import MLP
from linodenet.encoders.resnet import ResNet, ResNetBlock
from linodenet.encoders.transformer import Transformer

ENCODERS: dict[str, type[Encoder]] = {
    "MLP": MLP,
    "ResNet": ResNet,
    "Transformer": Transformer,
    "iResNet": iResNet,
}
r"""Dictionary containing all available encoders."""
