r"""Residual Network Implementation.

Modified variant of the implementation from https://github.com/yandex-research/rtdl

Original Licensed under Apache License 2.0
"""

__all__ = [
    # Classes
    "ResNet",
    "ResNetBlock",
]


import logging
from math import sqrt
from typing import Any, Optional, cast

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from linodenet.models.encoders.ft_transformer import (
    get_activation_fn,
    get_nonglu_activation_fn,
)
from linodenet.util import ReZero, autojit, deep_dict_update, initialize_from_config

__logger__ = logging.getLogger(__name__)


@autojit
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


@autojit
class ResNetBlock(nn.Module):
    """Pre-activation ResNet block.

    References
    ----------
    - | Identity Mappings in Deep Residual Networks
      | Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
      | European Conference on Computer Vision 2016
      | https://link.springer.com/chapter/10.1007/978-3-319-46493-0_38
    """

    HP: dict = {
        "__name__": "ResNetBlock",
        "__module__": __module__,  # type: ignore[name-defined]
        "input_size": None,
        "num_blocks": 2,
        "rezero": True,
        "layers": [
            {
                "__name__": "BatchNorm1d",
                "__module__": "torch.nn",
                "num_features": None,
                "eps": 1e-05,
                "momentum": 0.1,
                "affine": True,
                "track_running_stats": True,
            },
            {
                "__name__": "ReLU",
                "__module__": "torch.nn",
                "inplace": False,
            },
            {
                "__name__": "Linear",
                "__module__": "torch.nn",
                "in_features": None,
                "out_features": None,
                "bias": True,
            },
        ],
    }

    def __init__(self, **HP: Any) -> None:
        super().__init__()

        deep_dict_update(self.HP, HP)
        HP = self.HP

        assert HP["input_size"] is not None, "input_size is required!"

        for layer in HP["layers"]:
            if layer["__name__"] == "Linear":
                layer["in_features"] = HP["input_size"]
                layer["out_features"] = HP["input_size"]
            if layer["__name__"] == "BatchNorm1d":
                layer["num_features"] = HP["input_size"]

        layers: list[nn.Module] = [
            nn.Sequential(*[initialize_from_config(layer) for layer in HP["layers"]])
            for _ in range(HP["num_blocks"])
        ]

        if HP["rezero"]:
            # self.rezero = ReZero()
            layers.append(ReZero())

        self.layers = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        r"""Forward pass.

        Parameters
        ----------
        x: Tensor

        Returns
        -------
        Tensor
        """
        return self.layers(x)


@autojit
class ResNet(nn.Module):
    """A ResNet model."""

    HP: dict = {
        "input_size": None,
        "num_blocks": 5,
        "rezero": True,
        "Block": ResNetBlock.HP,
    }

    def __init__(self, **HP: Any) -> None:
        super().__init__()
        deep_dict_update(self.HP, HP)
        HP = self.HP

        assert HP["input_size"] is not None, "input_size is required!"
        assert "input_size" in HP["Block"], "input_size must be a key!"
        assert "rezero" in HP["Block"], "rezero must be a key!"

        HP["Block"]["rezero"] = HP["rezero"]
        HP["Block"]["input_size"] = HP["input_size"]

        blocks = [initialize_from_config(HP["Block"]) for _ in range(HP["num_blocks"])]

        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: Tensor) -> Tensor:
        r"""Forward pass.

        Parameters
        ----------
        x: Tensor

        Returns
        -------
        Tensor
        """
        for block in self.blocks:
            x = x + block(x)

        return x
