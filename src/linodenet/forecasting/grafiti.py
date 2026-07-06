r"""GraFITi layers for irregular time-series forecasting."""

__all__ = [
    "MAB",
    "Grafiti",
    "gather_target_embeddings",
    "reconstruct_y",
]

from typing import Final, Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nan, nn

from .utils import EventBatch


class MAB(nn.Module):
    r"""Multi-head attention block with configurable hidden dimension.

    References:
        - | Set Transformer: A Framework for Attention-based Permutation-Invariant Neural Networks
          | Lee et al.
          | International Conference on Machine Learning (ICML) 2019.
          | http://proceedings.mlr.press/v97/lee19d.html
    """

    def __init__(
        self,
        *,
        dim_Q: int,
        dim_K: int,
        dim_V: int,
        dim_hidden: int,
        num_heads: int,
        layer_norm: bool = False,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if dim_hidden % num_heads != 0:
            raise ValueError(f"{dim_hidden=} must be divisible by {num_heads=}.")
        self.dim_hidden = dim_hidden
        self.num_heads = num_heads
        # A key bias adds the same constant to every logit for a fixed query, so
        # softmax cancels it and the parameter cannot affect the attention map.
        self.fc_q = nn.Linear(dim_Q, dim_hidden, bias=bias)
        self.fc_k = nn.Linear(dim_K, dim_hidden, bias=False)
        self.fc_v = nn.Linear(dim_V, dim_hidden, bias=bias)
        self.layer_norm0 = nn.LayerNorm(dim_hidden) if layer_norm else None
        self.layer_norm1 = nn.LayerNorm(dim_hidden) if layer_norm else None
        self.fc_o = nn.Linear(dim_hidden, dim_hidden)

    def forward(
        self,
        Q: Tensor,  # (..., Nq, dim_Q)
        K: Tensor,  # (..., Nk, dim_K)
        V: Tensor,  # (..., Nk, dim_V)
        *,
        mask: Tensor | None = None,  # (..., Nq, Nk)
    ) -> Tensor:  # (..., Nq, dim_hidden)
        r"""Apply the attention block.

        Args:
            Q: Query tensor with shape ``(..., query_len, dim_Q)``.
            K: Key tensor with shape ``(..., key_len, dim_K)``.
            V: Value tensor with shape ``(..., key_len, dim_V)``.
            mask: Optional boolean attention mask with shape
                ``(..., query_len, key_len)``. True entries participate in attention.

        Returns:
            Updated query embeddings with shape ``(..., query_len, dim_hidden)``.
        """
        # H is the number of attention heads, M is dim_hidden, and each head
        # contracts over M/H features.
        head_dim = self.dim_hidden // self.num_heads

        Q = self.fc_q(Q)  # (..., Nq, M)
        K = self.fc_k(K)  # (..., Nk, M)
        V = self.fc_v(V)  # (..., Nk, M)

        # Split hidden features into heads and per-head features.
        Q = Q.unflatten(dim=-1, sizes=(self.num_heads, head_dim))  # (..., Nq, H, M/H)
        K = K.unflatten(dim=-1, sizes=(self.num_heads, head_dim))  # (..., Nk, H, M/H)
        V = V.unflatten(dim=-1, sizes=(self.num_heads, head_dim))  # (..., Nk, H, M/H)

        # Move heads before the token axis for batched per-head attention.
        Q = Q.movedim(-2, -3)  # (..., H, Nq, M/H)
        K = K.movedim(-2, -3)  # (..., H, Nk, M/H)
        V = V.movedim(-2, -3)  # (..., H, Nk, M/H)

        Y = Q + F.scaled_dot_product_attention(  # (..., H, Nq, M/H)
            Q,
            K,
            V,
            attn_mask=None if mask is None else mask.unsqueeze(dim=-3),
            dropout_p=0.0,
        )
        Y = Y.movedim(-3, -2)  # (..., Nq, H, M/H)
        Y = Y.flatten(start_dim=-2)  # (..., Nq, M)

        if self.layer_norm0 is not None:
            Y = self.layer_norm0(Y)  # (..., Nq, M)

        Y = Y + F.relu(self.fc_o(Y))  # (..., Nq, M)

        if self.layer_norm1 is not None:
            Y = self.layer_norm1(Y)  # (..., Nq, M)

        return Y  # (..., Nq, M)


def gather_target_embeddings(
    h_edge: Tensor,  # (..., E, M)
    *,
    target_mask: Tensor,  # (..., E), bool
) -> Tensor:  # (..., K, M)
    r"""Select and pad edge embeddings at target positions.

    Args:
        h_edge: Edge embeddings with shape ``(..., max_edges, hidden_dim)``.
        target_mask: Boolean target mask with shape ``(..., max_edges)``.

    Returns:
        Target embeddings with shape ``(..., max_targets, hidden_dim)``.
    """
    assert target_mask.dtype == torch.bool

    *batch_shape, _, hidden_dim = h_edge.shape
    max_targets = int(target_mask.sum(dim=-1).max().item())

    *batch_indices, edge_indices = target_mask.nonzero(as_tuple=True)
    offsets = target_mask.cumsum(dim=-1) - 1
    target_indices = offsets[*batch_indices, edge_indices]

    y_padded = h_edge.new_full((*batch_shape, max_targets, hidden_dim), nan)
    y_padded[*batch_indices, target_indices] = h_edge[*batch_indices, edge_indices]
    return y_padded


def reconstruct_y(
    y_at_edge: Tensor,  # (..., E)
    *,
    edge_mask: Tensor,  # (..., E), bool
    target_mask: Tensor,  # (..., T, D), bool
) -> Tensor:  # (..., T, D)
    r"""Reconstruct a dense tensor from flattened masked values.

    Args:
        y_at_edge: Flattened values with shape ``(..., max_edges)``.
        edge_mask: Boolean mask selecting valid entries from ``y_flat``.
        target_mask: Boolean mask with shape ``(..., time, dim)``. True entries mark
            dense positions that should be filled.

    Returns:
        Reconstructed tensor with shape ``(..., time, dim)``.
    """
    assert target_mask.dtype == torch.bool
    assert edge_mask.dtype == torch.bool

    y_reconstructed = y_at_edge.new_full(target_mask.shape, nan)  # (..., T, D)
    y_reconstructed[target_mask] = y_at_edge[edge_mask]  # (sum(E_...))
    return y_reconstructed  # (..., T, D)


class Grafiti(nn.Module):
    r"""GraFITi forecaster for observed and target time-series entries.

    References:
        - | GraFITi: Graphs for Forecasting Irregularly Sampled Time Series
          | Yalavarthi et al.
          | The Thirty-Eighth AAAI Conference on Artificial Intelligence (AAAI-24)
          | https://ojs.aaai.org/index.php/AAAI/article/view/29560
    """

    output_mode: Final[Literal["forecast", "embeddings"]]

    def __init__(
        self,
        *,
        dim_input: int = 41,
        dim_latent: int = 128,
        num_layers: int = 3,
        num_heads: int = 4,
        bias: bool = True,
        output_mode: Literal["forecast", "embeddings"] = "forecast",
    ) -> None:
        r"""Initialize the GraFITi forecaster.

        Args:
            dim_input: Number of channels.
            dim_latent: Latent embedding size.
            num_layers: Number of GraFITi layers.
            num_heads: Number of attention heads.
            output_mode: Whether :meth:`forward` returns dense forecasts or
                target-edge embeddings.
            bias: Whether to use bias in attention layers.
        """
        super().__init__()
        self.latent_dim = dim_latent
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dim_input = dim_input
        self.output_mode = output_mode

        self.time_init = nn.Linear(1, dim_latent)
        self.edge_init = nn.Linear(2, dim_latent)
        self.channel_init = nn.Linear(dim_input, dim_latent)

        self.channel_time_attn = nn.ModuleList([
            MAB(
                dim_Q=dim_latent,
                dim_K=2 * dim_latent,
                dim_V=2 * dim_latent,
                dim_hidden=dim_latent,
                num_heads=num_heads,
                bias=bias,
            )
            for _ in range(num_layers)
        ])  # fmt: skip

        self.time_channel_attn = nn.ModuleList([
            MAB(
                dim_Q=dim_latent,
                dim_K=2 * dim_latent,
                dim_V=2 * dim_latent,
                dim_hidden=dim_latent,
                num_heads=num_heads,
                bias=bias,
            )
            for _ in range(num_layers)
        ])  # fmt: skip

        self.edge_nn = nn.ModuleList([
            nn.Linear(3 * dim_latent, dim_latent)
            for _ in range(num_layers)
        ])  # fmt: skip

        self.output = nn.Linear(3 * dim_latent, 1)

        if self.output_mode == "embeddings":
            frozen_modules = [
                self.output,
                self.channel_time_attn[-1],
                self.time_channel_attn[-1],
            ]
            for module in frozen_modules:
                for parameter in module.parameters():
                    parameter.requires_grad_(False)

    def _create_masks(
        self,
        *,
        num_steps: int,  # T
        num_channels: int,  # D
        edge_time_indices: Tensor,  # (..., E)
        edge_channel_indices: Tensor,  # (..., E)
        edge_mask: Tensor,  # (..., E)
    ) -> tuple[Tensor, Tensor]:  # (..., T, E), (..., D, E)
        r"""Create masks for time and channel attention.

        Args:
            num_steps: Number of time steps.
            num_channels: Number of channels.
            edge_time_indices: Time node index for each edge with shape ``(..., edges)``.
            edge_channel_indices: Channel node index for each edge with shape
                ``(..., edges)``.
            edge_mask: Boolean mask for real entries in the padded flattened
                edge list with shape ``(..., edges)``.

        Returns:
            Tuple containing boolean time and channel masks with shapes
            ``(..., time, edges)`` and ``(..., dim, edges)``.
        """
        device = edge_mask.device
        time_index = torch.arange(num_steps, device=device)  # (T)
        channel_index = torch.arange(num_channels, device=device)  # (D)

        time_edge_mask = (  # (..., T, E)
            edge_mask[..., None, :]  # (..., 1, E)
            # (..., 1, E) == (..., T, 1) -> (..., T, E)
            & (edge_time_indices[..., None, :] == time_index[..., :, None])
        )
        channel_edge_mask = (  # (..., D, E)
            edge_mask[..., None, :]  # (..., 1, E)
            # (..., 1, E) == (..., D, 1) -> (..., D, E)
            & (edge_channel_indices[..., None, :] == channel_index[..., :, None])
        )
        return time_edge_mask, channel_edge_mask  # (..., T, E), (..., D, E)

    def _encode_features(
        self,
        t: Tensor,  # (..., $T)
        *,
        num_channels: int,
        edge_values: Tensor,  # (..., $E)
        edge_target_mask: Tensor,  # (..., $E)
        edge_mask: Tensor,  # (..., $E)
    ) -> tuple[Tensor, Tensor, Tensor]:  # (..., $E, M), (..., $T, M), (..., D, M)
        r"""Encode edge, time-node, and channel-node features.

        Args:
            t: Time-node features with shape ``(..., time)``.
            num_channels: Number of channel nodes.
            edge_values: Edge values with shape ``(..., edges)``.
            edge_target_mask: Boolean target-query mask with shape ``(..., edges)``.
            edge_mask: Flattened observation/target mask with shape ``(..., edges)``.

        Returns:
            Encoded edge, time, and channel features.
        """
        *batch_shape, _ = t.shape

        # encode time
        t_encoded = torch.sin(self.time_init(t[..., None]))  # (..., T, M)

        # encode channels
        channel_indices = torch.arange(num_channels, device=t.device)  # (D)
        c_onehot = F.one_hot(channel_indices, num_classes=num_channels)
        c_onehot = c_onehot.expand(*batch_shape, num_channels, num_channels)
        c_encoded = torch.relu(self.channel_init(c_onehot.float()))  # (..., D, M)

        # Encode whether an edge should be treated as a prediction target:
        # valid_edge_mask => edge_target_mask, i.e. every padded edge is a target and every
        # valid edge is a target exactly when it came from target_mask.
        target_indicator = ~edge_mask | edge_target_mask  # (..., E)
        edge_values = torch.where(edge_target_mask, 0.0, edge_values)
        edge_input = torch.stack(  # (..., E, 2)
            [edge_values, target_indicator.to(dtype=edge_values.dtype)],
            dim=-1,
        )
        e_encoded = torch.where(  # (..., E, M)
            edge_mask[..., None],
            torch.relu(self.edge_init(edge_input)),
            0.0,
        )
        return e_encoded, t_encoded, c_encoded  # (..., E, M), (..., T, M), (..., D, M)

    def predict(
        self,
        query_times: Tensor,  # Float[(..., $K)], padded NaN, strictly increasing
        query_mask: Tensor,  # Bool[(..., $K, F)]  padded False
        *,
        context_times: Tensor,  # Float[(..., $N)], padded NaN, non-decreasing
        context_mask: Tensor,  # Bool[(..., $N, D)], padded False
        context_values: Tensor,  # Float[(..., $N, D)], padded NaN, sparse
    ) -> Tensor:  # (..., $K, F)
        combined = EventBatch.from_request(
            context_times=context_times,
            context_values=context_values,
            context_mask=context_mask,
            query_times=query_times,
            query_mask=query_mask,
        )
        result = self.forward(
            timestamps=combined.timestamps,
            query_mask=combined.query_mask,
            context_values=combined.context_values,
            context_mask=combined.context_mask,
        )
        return result[combined.query_indices]

    def forward(
        self,
        *,
        timestamps: Tensor,  # (..., $T), float, padded NaN
        query_mask: Tensor,  # (..., $T, D), bool, padded False
        context_values: Tensor,  # (..., $T, D), float, padded Nan, sparse
        context_mask: Tensor,  # (..., $T, D), bool, padded False
    ) -> Tensor:  # (..., $T, D) if forecast, (..., $K, M) if embeddings
        r"""Process observed values and target queries with GraFITi.

        Args:
            timestamps: Times for observed and target entries with shape ``(..., time)``.
            context_values: Observed values with shape ``(..., time, dim)``.
            context_mask: Boolean context observation mask with shape ``(..., time, dim)``.
            query_mask: Boolean target-query mask with shape ``(..., time, dim)``.

        Returns:
            Dense predictions with shape ``(..., time, dim)`` in forecast mode,
            or target embeddings with shape ``(..., max_targets, hidden_dim)``
            in embeddings mode.
        """
        # Note: Shape legend for the dense GraFITi path
        #   T: total time nodes, D: total channel nodes, M: latent embedding dimension.
        #   N: total edges (context or target) across all batch elements.
        #   E: max edges (context or target) across all batch elements.
        #   K: max edges (target only) across all batch elements.

        # input validation/sanitation
        assert context_mask.dtype == torch.bool
        assert query_mask.dtype == torch.bool
        context_values = context_values.masked_fill(~context_mask, nan)
        assert torch.equal(context_values.isfinite(), context_mask)

        *batch_shape, num_steps, num_channels = context_values.shape
        device = timestamps.device

        dense_edge_mask = context_mask | query_mask  # (..., $T, D)

        # nonzero returns one global list of N true entries in row-major batch order.
        # Subtract each batch item's global start offset to get its local slot in E.
        *batch_idx, t_idx, c_idx = dense_edge_mask.nonzero(as_tuple=True)  # (N)
        counts = dense_edge_mask.sum(dim=(-2, -1))  # (...)
        offsets = counts.flatten().cumsum(dim=0).reshape(batch_shape) - counts  # (...)
        positions = torch.arange(t_idx.numel(), device=device)  # (N)
        edge_indices = (*batch_idx, positions - offsets[*batch_idx])
        max_edges = int(counts.max().item())  # E

        # collect the results in tensors of shape (..., $E)
        edge_t_indices = t_idx.new_zeros(*batch_shape, max_edges)
        edge_c_indices = c_idx.new_zeros(*batch_shape, max_edges)
        edge_values = context_values.new_zeros(*batch_shape, max_edges)
        edge_target_mask = dense_edge_mask.new_zeros(*batch_shape, max_edges)
        edge_mask = dense_edge_mask.new_zeros(*batch_shape, max_edges)

        edge_t_indices[edge_indices] = t_idx
        edge_c_indices[edge_indices] = c_idx
        edge_values[edge_indices] = context_values[*batch_idx, t_idx, c_idx]
        edge_target_mask[edge_indices] = query_mask[*batch_idx, t_idx, c_idx]
        edge_mask[edge_indices] = True

        # Masks route each flattened edge to its incident time and channel nodes.
        time_edge_mask, channel_edge_mask = (
            self._create_masks(  # (..., $T, $E), (..., D, $E)
                num_steps=num_steps,
                num_channels=num_channels,
                edge_time_indices=edge_t_indices,
                edge_channel_indices=edge_c_indices,
                edge_mask=edge_mask,
            )
        )
        h_edge, h_time, h_channel = (  # (..., $E, M), (..., $T, M), (..., D, M)
            self._encode_features(
                t=timestamps,
                num_channels=num_channels,
                edge_values=edge_values,
                edge_target_mask=edge_target_mask,
                edge_mask=edge_mask,
            )
        )

        for channel_time_attn, time_channel_attn, edge_nn in zip(
            self.channel_time_attn,
            self.time_channel_attn,
            self.edge_nn,
            strict=True,
        ):
            # collect matching nodes for a given edge (..., $E, M)
            h_t_at_edge = h_time.take_along_dim(edge_t_indices[..., None], dim=-2)
            h_c_at_edge = h_channel.take_along_dim(edge_c_indices[..., None], dim=-2)

            # Hᵤ = concat([hᵥ, hₑ]) for each edge e={u,v} connected to u. (eq 12)
            time_context = torch.cat([h_c_at_edge, h_edge], dim=-1)  # (..., $E, 2M)
            channel_context = torch.cat([h_t_at_edge, h_edge], dim=-1)  # (..., $E, 2M)
            edge_context = torch.cat(  # (..., $E, 3M)
                [h_edge, h_t_at_edge, h_c_at_edge], dim=-1
            )

            # update time node embeddings (eq 11 and 12)
            h_time = time_channel_attn(  # (..., $T, M)
                h_time,
                time_context,
                time_context,
                mask=time_edge_mask,
            )

            # update context node embeddings (eq 11 and 12)
            h_channel = channel_time_attn(  # (..., D, M)
                h_channel,
                channel_context,
                channel_context,
                mask=channel_edge_mask,
            )

            # update edge embeddings (eq 13)
            h_edge = torch.where(  # (..., $E, M)
                edge_mask[..., None],
                torch.relu(h_edge + edge_nn(edge_context)),
                0.0,
            )

        if self.output_mode == "embeddings":
            return gather_target_embeddings(  # (..., $K, M)
                h_edge, target_mask=edge_target_mask
            )

        # collect matching nodes for a given edge # (..., $E, M)
        h_t_at_edge = h_time.take_along_dim(edge_t_indices[..., None], dim=-2)
        h_c_at_edge = h_channel.take_along_dim(edge_c_indices[..., None], dim=-2)
        y_at_edge = self.output(  # (..., $E)
            torch.cat([h_edge, h_t_at_edge, h_c_at_edge], dim=-1)
        ).squeeze(dim=-1)

        return reconstruct_y(
            y_at_edge,
            edge_mask=edge_target_mask,
            target_mask=query_mask,
        )

    def forward_triplet(
        self,
        context_times: Tensor,  # (..., $O)
        context_channels: Tensor,  # (..., $O)
        context_values: Tensor,  # (..., $O)
        query_times: Tensor,  # (..., $Q)
        query_channels: Tensor,  # (..., $Q)
    ) -> Tensor:  # (..., $K, M)
        r"""Encode observed values and target queries from sparse triplets.

        Args:
            context_times: Times for observed values with shape ``(..., $O)``.
            context_channels: Channel indices for observed values with shape ``(..., $O)``.
            context_values: Observed values with shape ``(..., $O)``.
            query_times: Times for target queries with shape ``(..., $Q)``.
            query_channels: Channel indices for target queries with shape ``(..., $Q)``.

        Returns:
            Target edge embeddings with shape ``(..., $K, M)``.
        """
        assert context_channels.dtype == torch.long
        assert query_channels.dtype == torch.long
        assert (context_channels < self.dim_input).all()
        assert (query_channels < self.dim_input).all()

        *batch_shape, num_context = context_times.shape
        *_, num_query = query_times.shape
        device = context_times.device

        context_valid = context_channels.ge(0)  # (..., $O)
        query_valid = query_channels.ge(0)  # (..., $Q)

        num_edges = num_context + num_query
        timestamps = torch.cat([context_times, query_times], dim=-1)  # (..., $E)
        edge_mask = torch.cat([context_valid, query_valid], dim=-1)  # (..., $E)

        edge_c_indices = torch.cat(  # (..., $E)
            [context_channels, query_channels], dim=-1
        )
        edge_values = torch.cat(  # (..., $E)
            [context_values, torch.zeros_like(query_times)], dim=-1
        )
        edge_target_mask = torch.cat(  # (..., $E)
            [torch.zeros_like(context_valid), query_valid], dim=-1
        )
        edge_t_indices = torch.arange(num_edges, device=device).expand(
            *batch_shape, num_edges
        )  # (..., $E)

        edge_c_indices = edge_c_indices.masked_fill(~edge_mask, 0)
        edge_t_indices = edge_t_indices.masked_fill(~edge_mask, 0)
        edge_values = edge_values.masked_fill(~edge_mask, 0.0)
        edge_target_mask = edge_target_mask & edge_mask

        time_edge_mask, channel_edge_mask = (
            self._create_masks(  # (..., $E, $E), (..., D, $E)
                num_steps=num_edges,
                num_channels=self.dim_input,
                edge_time_indices=edge_t_indices,
                edge_channel_indices=edge_c_indices,
                edge_mask=edge_mask,
            )
        )
        h_edge, h_time, h_channel = (  # (..., $E, M), (..., $E, M), (..., D, M)
            self._encode_features(
                t=timestamps,
                num_channels=self.dim_input,
                edge_values=edge_values,
                edge_target_mask=edge_target_mask,
                edge_mask=edge_mask,
            )
        )

        for channel_time_attn, time_channel_attn, edge_nn in zip(
            self.channel_time_attn,
            self.time_channel_attn,
            self.edge_nn,
            strict=True,
        ):
            # collect matching nodes for a given edge (..., $E, M)
            h_t_at_edge = h_time.take_along_dim(edge_t_indices[..., None], dim=-2)
            h_c_at_edge = h_channel.take_along_dim(edge_c_indices[..., None], dim=-2)

            # Hᵤ = concat([hᵥ, hₑ]) for each edge e={u,v} connected to u. (eq 12)
            time_context = torch.cat([h_c_at_edge, h_edge], dim=-1)  # (..., $E, 2M)
            channel_context = torch.cat([h_t_at_edge, h_edge], dim=-1)  # (..., $E, 2M)
            edge_context = torch.cat(  # (..., $E, 3M)
                [h_edge, h_t_at_edge, h_c_at_edge], dim=-1
            )

            # update time node embeddings (eq 11 and 12)
            h_time = time_channel_attn(  # (..., $T, M)
                h_time,
                time_context,
                time_context,
                mask=time_edge_mask,
            )

            # update context node embeddings (eq 11 and 12)
            h_channel = channel_time_attn(  # (..., D, M)
                h_channel,
                channel_context,
                channel_context,
                mask=channel_edge_mask,
            )

            # update edge embeddings (eq 13)
            h_edge = torch.where(  # (..., $E, M)
                edge_mask[..., None],
                torch.relu(h_edge + edge_nn(edge_context)),
                0.0,
            )

        return gather_target_embeddings(  # (..., $K, M)
            h_edge, target_mask=edge_target_mask
        )
