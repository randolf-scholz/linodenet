r"""Residual Network Implementation.

Modified variant of the implementation from https://github.com/yandex-research/rtdl

Original Licensed under Apache License 2.0
"""

__all__ = [
    # Classes
    "ResNet",
    "ResNetBlock",
]

from collections.abc import Iterable
from typing import Any

from torch import Tensor, jit, nn

from linodenet.modules.layers import (
    ReverseDense,
    ReZeroCell,
)
from linodenet.utils import (
    deep_dict_update,
    initialize_from_dict,
)


class ResNetBlock(nn.Sequential):
    r"""Pre-activation ResNet block.

    References:
        Identity Mappings in Deep Residual Networks
        Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
        European Conference on Computer Vision 2016
        https://link.springer.com/chapter/10.1007/978-3-319-46493-0_38
    """

    HP = {
        "__name__": __qualname__,
        "__module__": __name__,
        "input_size": None,
        "num_layers": 2,
        "layer": ReverseDense.HP,
        "layer_cfg": {},
        "rezero": True,
    }

    def __init__(self, *modules: nn.Module, **cfg: Any) -> None:
        config = deep_dict_update(self.HP, cfg)
        if config.get("input_size") is None:
            raise ValueError("input_size is required!")

        layer = config["layer"]
        if layer["__name__"] == "Linear":
            layer["in_features"] = config["input_size"]
            layer["out_features"] = config["input_size"]
        if layer["__name__"] == "BatchNorm1d":
            layer["num_features"] = config["input_size"]
        else:
            layer["input_size"] = config["input_size"]
            layer["output_size"] = config["input_size"]

        layers: list[nn.Module] = list(modules)

        for _ in range(config["num_layers"]):
            module = initialize_from_dict(config["layer"])
            # self.add_module(f"subblock{k}", module)
            layers.append(module)

        if config["rezero"]:
            layers.append(ReZeroCell())

        # self.subblocks = nn.Sequential(subblocks)
        super().__init__(*layers)


class ResNet(nn.ModuleList):
    r"""A ResNet model."""

    HP = {
        "__name__": __qualname__,
        "__module__": __name__,
        "input_size": None,
        "num_blocks": 5,
        "block": ResNetBlock.HP,
    }

    def __init__(self, modules: Iterable[nn.Module], **cfg: Any) -> None:
        config = deep_dict_update(self.HP, cfg)
        if config.get("input_size") is None:
            raise ValueError("input_size is required!")

        # pass the input_size to the subblocks
        block = config["block"]
        if "input_size" in block:
            block["input_size"] = config["input_size"]

        blocks = [
            initialize_from_dict(config["block"]) for _ in range(config["num_blocks"])
        ]

        super().__init__(blocks)

    @jit.export
    def forward(self, x: Tensor) -> Tensor:
        r""".. Signature:: ``(..., n) -> (..., n)``."""
        for block in self:
            x = x + block(x)
        return x
