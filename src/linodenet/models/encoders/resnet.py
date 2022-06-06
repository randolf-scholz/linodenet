r"""Residual Network Implementation.

Modified variant of the implementation from https://github.com/yandex-research/rtdl

Original Licensed under Apache License 2.0
"""

__all__ = [
    # Classes
    "ResNet",
    "ResNetBlock",
]


from collections import OrderedDict
from math import sqrt
from typing import Any, Optional, cast

import torch
from torch import Tensor, jit, nn
from torch.nn import functional as F

from linodenet.models.encoders.ft_transformer import (
    get_activation_fn,
    get_nonglu_activation_fn,
)
from linodenet.util import (
    ReverseDense,
    ReZeroCell,
    deep_dict_update,
    initialize_from_config,
)


class ResNet_(nn.Module):
    """Residual Network."""

    def __init__(
        self,
        *,
        d_numerical: int,
        categories: Optional[list[int]],
        d_embedding: int,
        d: int,
        d_hidden_factor: float,
        n_layers: int,
        activation: str,
        normalization: str,
        hidden_dropout: float,
        residual_dropout: float,
        d_out: int,
    ) -> None:
        super().__init__()

        def make_normalization():
            return {"batchnorm": nn.BatchNorm1d, "layernorm": nn.LayerNorm}[
                normalization
            ](d)

        self.main_activation = get_activation_fn(activation)
        self.last_activation = get_nonglu_activation_fn(activation)
        self.residual_dropout = residual_dropout
        self.hidden_dropout = hidden_dropout

        d_in = d_numerical
        d_hidden = int(d * d_hidden_factor)

        if categories is not None:
            d_in += len(categories) * d_embedding
            category_offsets = torch.tensor([0] + categories[:-1]).cumsum(0)
            self.register_buffer("category_offsets", category_offsets)
            self.category_embeddings = nn.Embedding(sum(categories), d_embedding)
            nn.init.kaiming_uniform_(self.category_embeddings.weight, a=sqrt(5))
            print(f"{self.category_embeddings.weight.shape=}")

        self.first_layer = nn.Linear(d_in, d)
        self.layers = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "norm": make_normalization(),
                        "linear0": nn.Linear(
                            d, d_hidden * (2 if activation.endswith("glu") else 1)
                        ),
                        "linear1": nn.Linear(d_hidden, d),
                    }
                )
                for _ in range(n_layers)
            ]
        )
        self.last_normalization = make_normalization()
        self.head = nn.Linear(d, d_out)

    def forward(self, x_num: Tensor, x_cat: Optional[Tensor] = None) -> Tensor:
        """Forward pass.

        Parameters
        ----------
        x_num: Tensor
        x_cat: Optional[Tensor]

        Returns
        -------
        Tensor
        """
        tensors = []
        if x_num is not None:
            tensors.append(x_num)
        if x_cat is not None:
            assert self.category_embeddings is not None, "No category embeddings!"
            assert self.category_offsets is not None, "No category offsets!"

            tensors.append(
                self.category_embeddings(
                    x_cat + self.category_offsets[None]  # type: ignore[index]
                ).view(x_cat.size(0), -1)
            )
        x = torch.cat(tensors, dim=-1)

        x = self.first_layer(x)
        for layer in self.layers:
            layer = cast(dict[str, nn.Module], layer)
            z = x
            z = layer["norm"](z)
            z = layer["linear0"](z)
            z = self.main_activation(z)

            if self.hidden_dropout:
                z = F.dropout(z, self.hidden_dropout, self.training)

            z = layer["linear1"](z)

            if self.residual_dropout:
                z = F.dropout(z, self.residual_dropout, self.training)
            x = x + z
        x = self.last_normalization(x)
        x = self.last_activation(x)
        x = self.head(x)
        x = x.squeeze(-1)

        return x


class ResNetBlock(nn.Sequential):
    """Pre-activation ResNet block.

    References
    ----------
    - | Identity Mappings in Deep Residual Networks
      | Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
      | European Conference on Computer Vision 2016
      | https://link.springer.com/chapter/10.1007/978-3-319-46493-0_38
    """

    HP = {
        "__name__": __qualname__,  # type: ignore[name-defined]
        "__module__": __module__,  # type: ignore[name-defined]
        "input_size": None,
        "num_subblocks": 2,
        "subblocks": [
            # {
            #     "__name__": "BatchNorm1d",
            #     "__module__": "torch.nn",
            #     "num_features": int,
            #     "eps": 1e-05,
            #     "momentum": 0.1,
            #     "affine": True,
            #     "track_running_stats": True,
            # },
            ReverseDense.HP,
        ],
    }

    def __init__(self, **HP: Any) -> None:
        super().__init__()

        self.CFG = HP = deep_dict_update(self.HP, HP)

        assert HP["input_size"] is not None, "input_size is required!"

        for layer in HP["subblocks"]:
            if layer["__name__"] == "Linear":
                layer["in_features"] = HP["input_size"]
                layer["out_features"] = HP["input_size"]
            if layer["__name__"] == "BatchNorm1d":
                layer["num_features"] = HP["input_size"]
            else:
                layer["input_size"] = HP["input_size"]
                layer["output_size"] = HP["input_size"]

        subblocks: OrderedDict[str, nn.Module] = OrderedDict()

        for k in range(HP["num_subblocks"]):
            key = f"subblock{k}"
            module = nn.Sequential(
                *[initialize_from_config(layer) for layer in HP["subblocks"]]
            )
            self.add_module(key, module)
            subblocks[key] = module

        # self.subblocks = nn.Sequential(subblocks)
        super().__init__(subblocks)


class ResNet(nn.ModuleList):
    """A ResNet model."""

    HP = {
        "__name__": __qualname__,  # type: ignore[name-defined]
        "__module__": __module__,  # type: ignore[name-defined]
        "input_size": None,
        "num_blocks": 5,
        "blocks": [
            ResNetBlock.HP,
            ReZeroCell.HP,
        ],
    }

    def __init__(self, **HP: Any) -> None:
        super().__init__()
        self.CFG = HP = deep_dict_update(self.HP, HP)

        assert HP["input_size"] is not None, "input_size is required!"

        # pass the input_size to the subblocks
        for block_cfg in HP["blocks"]:
            if "input_size" in block_cfg:
                block_cfg["input_size"] = HP["input_size"]

        blocks: list[nn.Module] = []

        for k in range(HP["num_blocks"]):
            key = f"block{k}"
            module = nn.Sequential(
                *[initialize_from_config(layer) for layer in HP["blocks"]]
            )
            self.add_module(key, module)
            blocks.append(module)

        super().__init__(blocks)

    @jit.export
    def forward(self, x: Tensor) -> Tensor:
        r"""Forward pass.

        Parameters
        ----------
        x: Tensor

        Returns
        -------
        Tensor
        """
        for block in self:
            x = x + block(x)
        return x
