r"""GraFITi layers for irregular time-series forecasting."""

__all__ = [
    "MAB",
    "Grafiti",
    "batch_flatten",
    "gather_target_embeddings",
    "reconstruct_y",
]

import math
from collections.abc import Iterable

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class MAB(nn.Module):
    r"""Multi-head attention block with configurable hidden dimension."""

    def __init__(
        self,
        *,
        dim_Q: int,
        dim_K: int,
        dim_V: int,
        dim_hidden: int,
        num_heads: int,
        layer_norm: bool = False,
    ) -> None:
        super().__init__()
        if dim_hidden % num_heads != 0:
            raise ValueError(f"{dim_hidden=} must be divisible by {num_heads=}.")
        self.dim_hidden = dim_hidden
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_hidden)
        self.fc_k = nn.Linear(dim_K, dim_hidden)
        self.fc_v = nn.Linear(dim_V, dim_hidden)
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

        # Reference GraFITi divides by sqrt(dim_hidden) after splitting heads.
        # We use the standard scaled-attention factor for the contracted axis.
        attention_scores = (Q / math.sqrt(head_dim)) @ K.mT  # (..., H, Nq, Nk)

        if mask is not None:
            # Broadcast mask across heads: (..., Nq, Nk) -> (..., 1, Nq, Nk).
            attention_scores = attention_scores.masked_fill(
                ~mask.unsqueeze(dim=-3),
                -10e9,
            )

        attention = attention_scores.softmax(dim=-1)  # (..., H, Nq, Nk)
        Y = Q + attention @ V  # (..., H, Nq, M/H)
        Y = Y.movedim(-3, -2)  # (..., Nq, H, M/H)
        Y = Y.flatten(start_dim=-2)  # (..., Nq, M)

        if self.layer_norm0 is not None:
            Y = self.layer_norm0(Y)  # (..., Nq, M)

        Y = Y + F.relu(self.fc_o(Y))  # (..., Nq, M)

        if self.layer_norm1 is not None:
            Y = self.layer_norm1(Y)  # (..., Nq, M)

        return Y  # (..., Nq, M)


def batch_flatten(
    tensors: Iterable[Tensor],  # iterable[(..., T, D)]
    *,
    mask: Tensor,  # (..., T, D)
) -> list[Tensor]:  # list[(..., K')]
    r"""Flatten batched time-series tensors according to an observation mask.

    Args:
        tensors: Tensors with shape ``(..., time, dim)``.
        mask: Boolean mask tensor with shape ``(..., time, dim)``.

    Returns:
        List of padded flattened tensors, each with shape ``(..., max_observed)``.
    """
    assert mask.dtype == torch.bool

    *batch_shape, num_steps, num_channels = mask.shape
    num_batch = math.prod(batch_shape)
    num_edges = num_steps * num_channels
    device = mask.device
    mask_flat = mask.reshape(num_batch, num_edges)  # (B_flat, T*D)

    observed_counts = mask_flat.sum(dim=1)  # (B_flat)
    k = int(observed_counts.max().item())  # K'

    indices = torch.arange(k, device=device).expand(num_batch, k)  # (B_flat, K')
    mask_indices = indices < observed_counts.unsqueeze(dim=-1)  # (B_flat, K')

    y_padded: list[Tensor] = []
    for x in tensors:
        x_flat = x.reshape(num_batch, num_edges)  # (B_flat, T*D)
        observed_values = x_flat[mask_flat]  # (sum(K'_b))
        y_padded_flat = torch.full(  # (B_flat, K')
            (num_batch, k),
            0,
            device=device,
            dtype=x.dtype,
        )
        y_padded_flat[mask_indices] = observed_values
        y_padded.append(y_padded_flat.reshape(*batch_shape, k))  # (..., K')

    return y_padded  # list[(..., K')]


def gather_target_embeddings(
    x: Tensor,  # (..., K', M)
    *,
    mask: Tensor,  # (..., K'), bool
) -> Tensor:  # (..., K, M)
    r"""Select and pad edge embeddings at target positions.

    Args:
        x: Input tensor with shape ``(..., padded_edges, hidden_dim)``.
        mask: Boolean target mask with shape ``(..., padded_edges)``.

    Returns:
        Target embeddings with shape ``(..., max_targets, hidden_dim)``.
    """
    assert mask.dtype == torch.bool

    *batch_shape, num_edges, hidden_dim = x.shape
    num_batches = math.prod(batch_shape)
    mask_flat = mask.reshape(num_batches, num_edges)  # (B_flat, K')
    x_flat = x.reshape(num_batches, num_edges, hidden_dim)  # (B_flat, K', M)

    observed_counts = mask_flat.sum(dim=-1)  # (B_flat)
    k = int(observed_counts.max().to(torch.int64).item())  # K

    indices = torch.arange(k, device=mask.device).expand(num_batches, k)  # (B_flat, K)
    mask_indices = indices < observed_counts.unsqueeze(dim=-1)  # (B_flat, K)

    observed_values = x_flat[mask_flat]  # (sum(K_...), M)
    y_padded = torch.zeros(
        num_batches, k, hidden_dim, device=mask.device, dtype=x.dtype
    )
    y_padded[mask_indices] = observed_values
    return y_padded.reshape(*batch_shape, k, hidden_dim)  # (..., K, M)


def reconstruct_y(
    *,
    y_flat: Tensor,  # (..., K')
    y_mask: Tensor,  # (..., T, D), bool
    flat_edge_mask: Tensor,  # (..., K'), bool
) -> Tensor:  # (..., T, D)
    r"""Reconstruct a dense tensor from flattened masked values.

    Args:
        y_flat: Flattened values with shape ``(..., max_observed)``.
        y_mask: Boolean mask with shape ``(..., time, dim)``. True entries mark
            dense positions that should be filled.
        flat_edge_mask: Boolean mask selecting valid entries from ``y_flat``.

    Returns:
        Reconstructed tensor with shape ``(..., time, dim)``.
    """
    assert y_mask.dtype == torch.bool
    assert flat_edge_mask.dtype == torch.bool

    y_reconstructed = torch.zeros_like(y_mask, dtype=y_flat.dtype)  # (..., T, D)
    y_reconstructed[y_mask] = y_flat[flat_edge_mask]  # (sum(K_...))
    return y_reconstructed  # (..., T, D)


class Grafiti(nn.Module):
    r"""GraFITi encoder for observed and target time-series entries."""

    def __init__(
        self,
        *,
        input_dim: int = 41,
        hidden_dim: int = 128,
        num_layers: int = 3,
        num_heads: int = 4,
        device: str = "cuda",
    ) -> None:
        r"""Initialize the GraFITi encoder.

        Args:
            input_dim: Number of channels.
            hidden_dim: Latent embedding size.
            num_layers: Number of GraFITi layers.
            num_heads: Number of attention heads.
            device: Device name retained for compatibility with the reference API.
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.device = device
        self.num_layers = num_layers

        self.time_init = nn.Linear(1, hidden_dim)
        self.edge_init = nn.Linear(2, hidden_dim)
        self.channel_init = nn.Linear(input_dim, hidden_dim)

        self.channel_time_attn = nn.ModuleList([
            MAB(
                dim_Q=hidden_dim,
                dim_K=2 * hidden_dim,
                dim_V=2 * hidden_dim,
                dim_hidden=hidden_dim,
                num_heads=num_heads,
            )
            for _ in range(num_layers)
        ])  # fmt: skip

        self.time_channel_attn = nn.ModuleList([
            MAB(
                dim_Q=hidden_dim,
                dim_K=2 * hidden_dim,
                dim_V=2 * hidden_dim,
                dim_hidden=hidden_dim,
                num_heads=num_heads,
            )
            for _ in range(num_layers)
        ])  # fmt: skip

        self.edge_nn = nn.ModuleList([
            nn.Linear(3 * hidden_dim, hidden_dim)
            for _ in range(num_layers)
        ])  # fmt: skip

        self.output = nn.Linear(3 * hidden_dim, hidden_dim)

    def _one_hot_channels(
        self,
        *size: int,
        num_channels: int,
        device: torch.device,
    ) -> Tensor:  # (..., D, D)
        r"""Build one-hot channel identifiers.

        Args:
            size: Leading batch shape.
            num_channels: Number of channels.
            device: Device for the resulting tensor.

        Returns:
            One-hot channel encoding with shape ``(*size, dim, dim)``.
        """
        indices = torch.arange(num_channels, device=device)  # (D)
        one_hot = F.one_hot(indices, num_classes=num_channels).float()  # (D, D)
        return one_hot.expand(*size, num_channels, num_channels)  # (..., D, D)

    def _build_indices(
        self,
        t: Tensor,  # (..., T)
        *,
        num_channels: int,  # D
    ) -> tuple[Tensor, Tensor]:  # (..., T, D), (..., T, D)
        r"""Build dense time and channel index tensors.

        Args:
            t: Time tensor with shape ``(..., time)``.
            num_channels: Number of channels.

        Returns:
            Tuple containing time indices and channel indices, both with shape
            ``(..., time, dim)``.
        """
        *batch_shape, num_steps = t.shape

        # Time indices identify the time node attached to each dense edge.
        t_inds = torch.arange(num_steps, device=t.device).unsqueeze(-1)  # (T, 1)
        t_inds = t_inds.expand(*batch_shape, num_steps, num_channels)  # (..., T, D)

        # Channel indices identify the channel node attached to each dense edge.
        c_inds = torch.arange(num_channels, device=t.device)  # (D)
        c_inds = c_inds.expand(*batch_shape, num_steps, num_channels)  # (..., T, D)
        return t_inds, c_inds

    def _create_masks(
        self,
        *,
        num_steps: int,  # T
        num_channels: int,  # D
        t_inds_flat: Tensor,  # (..., K')
        c_inds_flat: Tensor,  # (..., K')
        valid_edge_mask: Tensor,  # (..., K')
    ) -> tuple[Tensor, Tensor]:  # (..., T, K'), (..., D, K')
        r"""Create masks for time and channel attention.

        Args:
            num_steps: Number of time steps.
            num_channels: Number of channels.
            t_inds_flat: Flattened time indices with shape ``(..., edges)``.
            c_inds_flat: Flattened channel indices with shape ``(..., edges)``.
            valid_edge_mask: Boolean mask for real entries in the padded flattened
                edge list with shape ``(..., edges)``.

        Returns:
            Tuple containing boolean time and channel masks with shapes
            ``(..., time, edges)`` and ``(..., dim, edges)``.
        """
        device = valid_edge_mask.device
        time_index = torch.arange(num_steps, device=device)  # (T)
        channel_index = torch.arange(num_channels, device=device)  # (D)

        t_mask = (  # (..., T, K')
            valid_edge_mask.unsqueeze(dim=-2)  # (..., 1, K')
            # (..., 1, K') == (..., T, 1) -> (..., T, K')
            & (t_inds_flat.unsqueeze(dim=-2) == time_index.unsqueeze(dim=-1))
        )
        c_mask = (  # (..., D, K')
            valid_edge_mask.unsqueeze(dim=-2)  # (..., 1, K')
            # (..., 1, K') == (..., D, 1) -> (..., D, K')
            & (c_inds_flat.unsqueeze(dim=-2) == channel_index.unsqueeze(dim=-1))
        )
        return t_mask, c_mask  # (..., T, K'), (..., D, K')

    def _encode_features(
        self,
        *,
        t: Tensor,  # (..., T)
        c_onehot: Tensor,  # (..., D, D)
        u_raw: Tensor,  # (..., K', 2)
        mask: Tensor,  # (..., K')
    ) -> tuple[Tensor, Tensor, Tensor]:  # (..., K', M), (..., T, M), (..., D, M)
        r"""Encode edge, time-node, and channel-node features.

        Args:
            u_raw: Edge features with shape ``(..., edges, 2)``.
            t: Time-node features with shape ``(..., time)``.
            c_onehot: Channel-node features with shape ``(..., dim, dim)``.
            mask: Flattened observation/target mask with shape ``(..., edges)``.

        Returns:
            Encoded edge, time, and channel features.
        """
        u_encoded = torch.relu(self.edge_init(u_raw))  # (..., K', M)
        u_encoded = torch.where(
            mask.unsqueeze(dim=-1),
            u_encoded,
            0.0,
        )  # (..., K', M)
        t_encoded = torch.sin(self.time_init(t.unsqueeze(dim=-1)))  # (..., T, M)
        c_encoded = torch.relu(self.channel_init(c_onehot))  # (..., D, M)
        return u_encoded, t_encoded, c_encoded  # (..., K', M), (..., T, M), (..., D, M)

    def forward_triplet(
        self,
        context_times: Tensor,  # (..., O)
        context_channels: Tensor,  # (..., O)
        context_values: Tensor,  # (..., O)
        query_times: Tensor,  # (..., Q)
        query_channels: Tensor,  # (..., Q)
    ) -> Tensor:  # (..., K, M)
        r"""Encode observed values and target queries from sparse triplets.

        Args:
            context_times: Times for observed values with shape
                ``(..., num_context)``.
            context_channels: Channel indices for observed values with shape
                ``(..., num_context)``.
            context_values: Observed values with shape ``(..., num_context)``.
            query_times: Times for target queries with shape ``(..., num_query)``.
            query_channels: Channel indices for target queries with shape
                ``(..., num_query)``.

        Returns:
            Target edge embeddings with shape ``(..., max_targets, hidden_dim)``.
        """
        assert context_channels.dtype == torch.long
        assert query_channels.dtype == torch.long

        *batch_shape, num_context = context_times.shape
        *_, num_query = query_times.shape
        num_batches = math.prod(batch_shape)
        num_channels = self.channel_init.in_features
        device = context_times.device

        context_times_flat = context_times.reshape(num_batches, num_context)  # (B, O)
        context_channels_flat = context_channels.reshape(
            num_batches, num_context
        )  # (B, O)
        context_values_flat = context_values.reshape(num_batches, num_context)  # (B, O)
        query_times_flat = query_times.reshape(num_batches, num_query)  # (B, Q)
        query_channels_flat = query_channels.reshape(num_batches, num_query)  # (B, Q)

        context_valid = (  # (B, O)
            context_times_flat.isfinite()
            & context_channels_flat.ge(0)
            & context_values_flat.isfinite()
        )
        query_valid = query_times_flat.isfinite() & query_channels_flat.ge(0)  # (B, Q)

        if (context_channels_flat[context_valid] >= num_channels).any():
            raise ValueError("Expected context channel indices below input_dim.")
        if (query_channels_flat[query_valid] >= num_channels).any():
            raise ValueError("Expected query channel indices below input_dim.")

        edge_times = torch.cat([context_times_flat, query_times_flat], dim=-1)  # (B, E)
        edge_valid_mask = torch.cat([context_valid, query_valid], dim=-1)  # (B, E)
        sort_indices = torch.argsort(  # (B, E)
            edge_times.nan_to_num(nan=torch.inf),
            dim=-1,
            stable=True,
        )
        edge_channel_indices = torch.cat(  # (B, E)
            [context_channels_flat, query_channels_flat], dim=-1
        )
        edge_values = torch.cat(  # (B, E)
            [context_values_flat, torch.zeros_like(query_times_flat)], dim=-1
        )
        edge_target_mask = torch.cat(
            [torch.zeros_like(context_valid), query_valid], dim=-1
        )  # (B, E)

        num_edges = num_context + num_query
        num_steps = num_edges
        time_points, c_inds_f, obs_vals, tgt_mask_f, mask_f = [
            torch.take_along_dim(tensor, sort_indices, dim=-1).reshape(
                *batch_shape, num_edges
            )
            for tensor in (
                edge_times,
                edge_channel_indices,
                edge_values,
                edge_target_mask,
                edge_valid_mask,
            )
        ]  # (..., E)
        t_inds_f = (
            torch.arange(num_edges, device=device)
            .expand(num_batches, num_edges)
            .reshape(*batch_shape, num_edges)
        )  # (..., E)
        c_inds_f = c_inds_f.masked_fill(~mask_f, 0)
        t_inds_f = t_inds_f.masked_fill(~mask_f, 0)
        obs_vals = obs_vals.masked_fill(~mask_f, 0.0)
        tgt_mask_f = tgt_mask_f & mask_f

        c_onehot = self._one_hot_channels(  # (..., D, D)
            *batch_shape,
            num_channels=num_channels,
            device=device,
        )

        target_indicator = ~mask_f | tgt_mask_f  # (..., O+Q)
        edge_input = torch.stack([  # (..., O+Q, 2)
            obs_vals,
            target_indicator.to(dtype=obs_vals.dtype),
        ], dim=-1)  # fmt: skip

        t_mask, c_mask = self._create_masks(  # (..., T', O+Q), (..., D, O+Q)
            num_steps=num_steps,
            num_channels=num_channels,
            t_inds_flat=t_inds_f,
            c_inds_flat=c_inds_f,
            valid_edge_mask=mask_f,
        )
        edge_emb, t_emb, c_emb = (  # (..., O+Q, M), (..., T', M), (..., D, M)
            self._encode_features(
                t=time_points,
                u_raw=edge_input,
                c_onehot=c_onehot,
                mask=mask_f,
            )
        )

        for channel_time_attn, time_channel_attn, edge_nn in zip(
            self.channel_time_attn,
            self.time_channel_attn,
            self.edge_nn,
            strict=True,
        ):
            t_gathered = torch.take_along_dim(  # (..., O+Q, M)
                t_emb, t_inds_f.unsqueeze(dim=-1), dim=-2
            )
            c_gathered = torch.take_along_dim(  # (..., O+Q, M)
                c_emb, c_inds_f.unsqueeze(dim=-1), dim=-2
            )

            channel_context = torch.cat(
                [t_gathered, edge_emb], dim=-1
            )  # (..., O+Q, 2*M)
            c_emb = channel_time_attn(  # (..., D, M)
                c_emb,
                channel_context,
                channel_context,
                mask=c_mask,
            )

            time_context = torch.cat([c_gathered, edge_emb], dim=-1)  # (..., O+Q, 2*M)
            t_emb = time_channel_attn(  # (..., T', M)
                t_emb,
                time_context,
                time_context,
                mask=t_mask,
            )

            edge_update = torch.cat(  # (..., O+Q, 3*M)
                [edge_emb, t_gathered, c_gathered],
                dim=-1,
            )
            edge_emb = torch.where(  # (..., O+Q, M)
                mask_f.unsqueeze(dim=-1),
                torch.relu(edge_emb + edge_nn(edge_update)),
                0.0,
            )

        return gather_target_embeddings(edge_emb, mask=tgt_mask_f)  # (..., K, M)

    def forward_combined(
        self,
        time_points: Tensor,  # (..., T)
        context_values: Tensor,  # (..., T, D)
        target_mask: Tensor,  # (..., T, D)
    ) -> Tensor:  # (..., K, M)
        r"""Encode observed values and target queries.

        Args:
            time_points: Times for observed and target entries with shape ``(..., time)``.
            context_values: Observed values with shape ``(..., time, dim)``.
            target_mask: Boolean target-query mask with shape ``(..., time, dim)``.

        Returns:
            Target edge embeddings with shape ``(..., max_targets, hidden_dim)``.
        """
        assert target_mask.dtype == torch.bool

        *batch_shape, _, num_channels = context_values.shape
        device = time_points.device

        c_onehot = self._one_hot_channels(  # (..., D, D)
            *batch_shape, num_channels=num_channels, device=device
        )

        t_inds, c_inds = self._build_indices(  # (..., T, D)
            time_points, num_channels=num_channels
        )
        obs_mask = context_values.isfinite()  # (..., T, D)
        mask = obs_mask | target_mask  # (..., T, D)

        # Flatten observed and target edges into padded edge lists. All outputs
        # have shape (..., K'), where K' is the max observed+target edge count.
        t_inds_f, obs_vals, tgt_mask_f, c_inds_f, mask_f = batch_flatten(
            [t_inds, context_values, target_mask, c_inds, mask], mask=mask
        )

        # Encode whether an edge should be treated as a prediction target:
        # mask_f => tgt_mask_f, i.e. every padded edge is a target and every
        # valid edge is a target exactly when it came from target_mask.
        target_indicator = ~mask_f | tgt_mask_f  # (..., K')
        obs_vals = torch.where(tgt_mask_f, 0.0, obs_vals)
        edge_input = torch.stack([  # (..., K', 2)
            obs_vals,
            target_indicator.to(dtype=obs_vals.dtype),
        ], dim=-1)  # fmt: skip

        # Masks route each flattened edge to its incident time and channel nodes.
        t_mask, c_mask = self._create_masks(  # (..., T, K'), (..., D, K')
            num_steps=time_points.shape[-1],
            num_channels=num_channels,
            t_inds_flat=t_inds_f,
            c_inds_flat=c_inds_f,
            valid_edge_mask=mask_f,
        )
        edge_emb, t_emb, c_emb = (  # (..., K', M), (..., T, M), (..., D, M)
            self._encode_features(
                t=time_points,
                u_raw=edge_input,
                c_onehot=c_onehot,
                mask=mask_f,
            )
        )

        for channel_time_attn, time_channel_attn, edge_nn in zip(
            self.channel_time_attn,
            self.time_channel_attn,
            self.edge_nn,
            strict=True,
        ):
            t_gathered = torch.take_along_dim(  # (..., K', M)
                t_emb, t_inds_f.unsqueeze(dim=-1), dim=-2
            )
            c_gathered = torch.take_along_dim(  # (..., K', M)
                c_emb, c_inds_f.unsqueeze(dim=-1), dim=-2
            )

            # (..., K', 2*M)
            channel_context = torch.cat([t_gathered, edge_emb], dim=-1)
            c_emb = channel_time_attn(  # (..., D, M)
                c_emb,
                channel_context,
                channel_context,
                mask=c_mask,
            )

            time_context = torch.cat([c_gathered, edge_emb], dim=-1)  # (..., K', 2*M)
            t_emb = time_channel_attn(  # (..., T, M)
                t_emb,
                time_context,
                time_context,
                mask=t_mask,
            )

            edge_update = torch.cat([  # (..., K', 3*M)
                edge_emb,
                t_gathered,
                c_gathered,
            ], dim=-1)  # fmt: skip

            edge_emb = torch.where(  # (..., K', M)
                mask_f.unsqueeze(dim=-1),
                torch.relu(edge_emb + edge_nn(edge_update)),
                0.0,
            )

        return gather_target_embeddings(edge_emb, mask=tgt_mask_f)  # (..., K, M)
