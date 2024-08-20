r"""Feature Tokenizer FT_Transformer.

Modified variant of the implementation from https://github.com/yandex-research/rtdl

Original Licensed under Apache License 2.0
"""

__all__ = [
    # Classes
    "ResNet",
    "FTTransformer",
    "MultiheadAttention",
    "Tokenizer",
]

from collections.abc import Callable
from math import sqrt
from typing import Optional, cast

import torch
from torch import Tensor, nn
from torch.nn import init as nn_init
from torch.nn.functional import dropout, gelu, relu, softmax

from linodenet.activations import get_activation


def _get_nonglu_activation_fn(name: str) -> Callable[[Tensor], Tensor]:
    r"""Get activation function by name."""
    match name:
        case "reglu":
            return relu
        case "geglu":
            return gelu
        case _:
            return get_activation(name)


class Tokenizer(nn.Module):
    r"""Tokenizer Model."""

    # BUFFERS
    category_offsets: Optional[Tensor]

    # PARAMETERS
    bias: Optional[Tensor]
    weight: Tensor

    def __init__(
        self,
        d_numerical: int,
        d_token: int,
        *,
        categories: Optional[list[int]],
        bias: bool,
    ) -> None:
        super().__init__()

        if categories is None:
            d_bias = d_numerical
            self.category_offsets = None
            self.category_embeddings = None
        else:
            d_bias = d_numerical + len(categories)
            category_offsets = torch.tensor([0] + categories[:-1]).cumsum(0)
            self.register_buffer("category_offsets", category_offsets)
            self.category_embeddings = nn.Embedding(sum(categories), d_token)
            nn_init.kaiming_uniform_(self.category_embeddings.weight, a=sqrt(5))
            print(f"{self.category_embeddings.weight.shape=}")

        # take [CLS] token into account
        self.weight = nn.Parameter(Tensor(d_numerical + 1, d_token))
        self.bias = nn.Parameter(Tensor(d_bias, d_token)) if bias else None

        # The initialization is inspired by nn.Linear
        nn_init.kaiming_uniform_(self.weight, a=sqrt(5))
        if self.bias is not None:
            nn_init.kaiming_uniform_(self.bias, a=sqrt(5))

    @property
    def n_tokens(self) -> int:
        r"""Return number of tokens."""
        return len(self.weight) + (
            0 if self.category_offsets is None else len(self.category_offsets)
        )

    def forward(self, x_num: Tensor, x_cat: Optional[Tensor] = None) -> Tensor:
        r""".. Signature:: ``(..., d) -> (..., e)``."""
        x_some = x_num if x_cat is None else x_cat

        x_num = torch.cat(
            [torch.ones(len(x_some), 1, device=x_some.device)]  # [CLS]
            + ([] if x_num is None else [x_num]),
            dim=1,
        )

        x = self.weight[None] * x_num[:, :, None]

        if x_cat is not None:
            assert (
                self.category_embeddings is not None
            ), "No category embeddings defined!"
            assert self.category_offsets is not None, "No category offsets defined!"

            categories = self.category_embeddings(
                x_cat + self.category_offsets[None]  # pylint: disable=unsubscriptable-object
            )

            x = torch.cat([x, categories], dim=1)

        if self.bias is not None:
            bias = torch.cat([
                torch.zeros(1, self.bias.shape[1], device=x.device),
                self.bias,
            ])
            x += bias[None]

        return x


class MultiheadAttention(nn.Module):
    r"""Multihead attention."""

    def __init__(
        self, d: int, *, n_heads: int, dropout_rate: float, initialization: str
    ) -> None:
        super().__init__()

        if n_heads > 1:
            assert d % n_heads == 0
        assert initialization in {"xavier", "kaiming"}

        self.W_q = nn.Linear(d, d)
        self.W_k = nn.Linear(d, d)
        self.W_v = nn.Linear(d, d)
        self.W_out = nn.Linear(d, d) if n_heads > 1 else None
        self.n_heads = n_heads
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate else None

        for m in [self.W_q, self.W_k, self.W_v]:
            if initialization == "xavier" and (n_heads > 1 or m is not self.W_v):
                # gain is needed since W_qkv is represented with 3 separate layers
                nn_init.xavier_uniform_(m.weight, gain=1 / sqrt(2))
            nn_init.zeros_(m.bias)
        if self.W_out is not None:
            nn_init.zeros_(self.W_out.bias)

    def _reshape(self, x: Tensor) -> Tensor:
        batch_size, n_tokens, d = x.shape
        d_head = d // self.n_heads
        return (
            x.reshape(batch_size, n_tokens, self.n_heads, d_head)
            .transpose(1, 2)
            .reshape(batch_size * self.n_heads, n_tokens, d_head)
        )

    def forward(
        self,
        x_q: Tensor,
        x_kv: Tensor,
        key_compression: Optional[nn.Linear],
        value_compression: Optional[nn.Linear],
    ) -> Tensor:
        r""".. Signature:: ``[(..., q), (...k) -> (..., d)``."""
        q, k, v = self.W_q(x_q), self.W_k(x_kv), self.W_v(x_kv)
        for tensor in [q, k, v]:
            assert tensor.shape[-1] % self.n_heads == 0

        if key_compression is not None:
            assert value_compression is not None
            k = key_compression(k.transpose(1, 2)).transpose(1, 2)
            v = value_compression(v.transpose(1, 2)).transpose(1, 2)
        else:
            assert value_compression is None

        batch_size = len(q)
        d_head_key = k.shape[-1] // self.n_heads
        d_head_value = v.shape[-1] // self.n_heads
        n_q_tokens = q.shape[1]
        q = self._reshape(q)
        k = self._reshape(k)

        attention = softmax(q @ k.transpose(1, 2) / sqrt(d_head_key), dim=-1)

        if self.dropout is not None:
            attention = self.dropout(attention)

        x = attention @ self._reshape(v)
        x = (
            x.reshape(batch_size, self.n_heads, n_q_tokens, d_head_value)
            .transpose(1, 2)
            .reshape(batch_size, n_q_tokens, self.n_heads * d_head_value)
        )

        if self.W_out is not None:
            x = self.W_out(x)

        return x


class FTTransformer(nn.Module):
    r"""FT_Transformer Model.

    References:
      - https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html
      - https://github.com/facebookresearch/pytext/tree/master/pytext/models/representations/transformer
      - https://github.com/pytorch/fairseq/blob/1bba712622b8ae4efb3eb793a8a40da386fe11d0/examples/linformer/linformer_src/modules/multihead_linear_attention.py#L19
    """

    def __init__(
        self,
        *,
        # tokenizer
        d_numerical: int,
        categories: Optional[list[int]],
        token_bias: bool,
        # transformer
        n_layers: int,
        d_token: int,
        n_heads: int,
        d_ffn_factor: float,
        attention_dropout: float,
        ffn_dropout: float,
        residual_dropout: float,
        activation: str,
        prenormalization: bool,
        initialization: str,
        # linformer
        kv_compression: Optional[float],
        kv_compression_sharing: Optional[str],
        d_out: int,
    ) -> None:
        super().__init__()

        assert (kv_compression is None) ^ (kv_compression_sharing is not None)

        self.tokenizer = Tokenizer(
            d_numerical,
            d_token,
            categories=categories,
            bias=token_bias,
        )
        n_tokens = self.tokenizer.n_tokens

        def make_kv_compression():
            assert kv_compression
            compression = nn.Linear(
                n_tokens, int(n_tokens * kv_compression), bias=False
            )
            if initialization == "xavier":
                nn_init.xavier_uniform_(compression.weight)
            return compression

        self.shared_kv_compression = (
            make_kv_compression()
            if kv_compression and kv_compression_sharing == "layerwise"
            else None
        )

        def make_normalization():
            return nn.LayerNorm(d_token)

        d_hidden = int(d_token * d_ffn_factor)
        self.layers = nn.ModuleList([])
        for layer_idx in range(n_layers):
            layer = nn.ModuleDict({
                "attention": MultiheadAttention(
                    d_token,
                    n_heads=n_heads,
                    dropout_rate=attention_dropout,
                    initialization=initialization,
                ),
                "linear0": nn.Linear(
                    d_token, d_hidden * (2 if activation.endswith("glu") else 1)
                ),
                "linear1": nn.Linear(d_hidden, d_token),
                "norm1": make_normalization(),
            })

            if not prenormalization or layer_idx:
                layer["norm0"] = make_normalization()

            if kv_compression and self.shared_kv_compression is None:
                layer["key_compression"] = make_kv_compression()

                if kv_compression_sharing == "headwise":
                    layer["value_compression"] = make_kv_compression()
                else:
                    assert kv_compression_sharing == "key-value"
            self.layers.append(layer)

        self.activation = get_activation(activation)
        self.last_activation = _get_nonglu_activation_fn(activation)
        self.prenormalization = prenormalization
        self.last_normalization = make_normalization() if prenormalization else None
        self.ffn_dropout = ffn_dropout
        self.residual_dropout = residual_dropout
        self.head = nn.Linear(d_token, d_out)

    def _get_kv_compressions(self, layer):
        return (
            (self.shared_kv_compression, self.shared_kv_compression)
            if self.shared_kv_compression is not None
            else (
                (layer["key_compression"], layer["value_compression"])
                if "key_compression" in layer and "value_compression" in layer
                else (
                    (layer["key_compression"], layer["key_compression"])
                    if "key_compression" in layer
                    else (None, None)
                )
            )
        )

    def _start_residual(self, x, layer, norm_idx):
        x_residual = x
        if self.prenormalization:
            norm_key = f"norm{norm_idx}"
            if norm_key in layer:
                x_residual = layer[norm_key](x_residual)
        return x_residual

    def _end_residual(self, x, x_residual, layer, norm_idx):
        if self.residual_dropout:
            x_residual = dropout(x_residual, self.residual_dropout, self.training)
        x = x + x_residual
        if not self.prenormalization:
            x = layer[f"norm{norm_idx}"](x)
        return x

    def forward(self, x_num: Tensor, x_cat: Optional[Tensor] = None) -> Tensor:
        r""".. Signature:: ``(..., e) -> (..., e)``."""
        x = self.tokenizer(x_num, x_cat)

        for layer_idx, layer in enumerate(self.layers):
            is_last_layer = layer_idx + 1 == len(self.layers)
            layer = cast(dict[str, nn.Module], layer)

            x_residual = self._start_residual(x, layer, 0)
            x_residual = layer["attention"](
                # for the last attention, it is enough to process only [CLS]
                (x_residual[:, :1] if is_last_layer else x_residual),
                x_residual,
                *self._get_kv_compressions(layer),
            )
            if is_last_layer:
                x = x[:, : x_residual.shape[1]]
            x = self._end_residual(x, x_residual, layer, 0)

            x_residual = self._start_residual(x, layer, 1)
            x_residual = layer["linear0"](x_residual)
            x_residual = self.activation(x_residual)
            if self.ffn_dropout:
                x_residual = dropout(x_residual, self.ffn_dropout, self.training)
            x_residual = layer["linear1"](x_residual)
            x = self._end_residual(x, x_residual, layer, 1)

        assert x.shape[1] == 1
        x = x[:, 0]
        if self.last_normalization is not None:
            x = self.last_normalization(x)
        x = self.last_activation(x)
        x = self.head(x)
        x = x.squeeze(-1)
        return x


class ResNet(nn.Module):
    r"""Residual Network."""

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

        self.main_activation = get_activation(activation)
        self.last_activation = _get_nonglu_activation_fn(activation)
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
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "norm": make_normalization(),
                "linear0": nn.Linear(
                    d, d_hidden * (2 if activation.endswith("glu") else 1)
                ),
                "linear1": nn.Linear(d_hidden, d),
            })
            for _ in range(n_layers)
        ])
        self.last_normalization = make_normalization()
        self.head = nn.Linear(d, d_out)

    def forward(self, x_num: Tensor, x_cat: Optional[Tensor] = None) -> Tensor:
        r""".. Signature:: ``(..., d) -> (..., d)``."""
        tensors = []
        if x_num is not None:
            tensors.append(x_num)
        if x_cat is not None:
            assert self.category_embeddings is not None, "No category embeddings!"
            assert self.category_offsets is not None, "No category offsets!"

            tensors.append(
                self.category_embeddings(x_cat + self.category_offsets[None]).view(
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
                z = dropout(z, self.hidden_dropout, self.training)

            z = layer["linear1"](z)

            if self.residual_dropout:
                z = dropout(z, self.residual_dropout, self.training)
            x = x + z
        x = self.last_normalization(x)
        x = self.last_activation(x)
        x = self.head(x)
        x = x.squeeze(-1)

        return x
