r"""Encoder Models."""

__all__ = [
    # Constants
    "ENCODERS",
    "MLP",
    # ABCs & Protocols
    "Encoder",
    "EncoderABC",
    # Classes
    "FTTransformer",
    "Identity",
    "LinearContraction",
    "ResNet",
    "ResNetBlock",
    "SpectralNorm",
    "Transformer",
    "iResNet",
    "iResNetBlock",
    # Functions
    "spectral_norm",
]


from linodenet.modules.encoders.base import Encoder, EncoderABC, Identity
from linodenet.modules.encoders.ft_transformer import FTTransformer
from linodenet.modules.encoders.iresnet import (
    LinearContraction,
    SpectralNorm,
    iResNet,
    iResNetBlock,
    spectral_norm,
)
from linodenet.modules.encoders.mlp import MLP
from linodenet.modules.encoders.resnet import ResNet, ResNetBlock
from linodenet.modules.encoders.transformer import Transformer

ENCODERS: dict[str, type[Encoder]] = {
    "FTTransformer": FTTransformer,
    "Identity": Identity,
    "MLP": MLP,
    "ResNet": ResNet,
    "Transformer": Transformer,
    "iResNet": iResNet,
}
r"""Dictionary containing all available encoders."""
