r"""Residual Network Implementation.

Modified variant of the implementation from https://github.com/yandex-research/rtdl

Original Licensed under Apache License 2.0
"""

__all__ = [
    # Classes
    "ResNet",
]


import logging
import math
from typing import Optional, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from linodenet.models.encoders.ft_transformer import (
    get_activation_fn,
    get_nonglu_activation_fn,
)
from linodenet.util import autojit

__logger__ = logging.getLogger(__name__)


# %%
@autojit
class ResNet(nn.Module):
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
            nn.init.kaiming_uniform_(self.category_embeddings.weight, a=math.sqrt(5))
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
                self.category_embeddings(x_cat + self.category_offsets[None]).view(  # type: ignore[index]
                    x_cat.size(0), -1
                )
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
