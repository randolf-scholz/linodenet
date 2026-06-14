r"""GraFITi layers for irregular time-series forecasting."""

__all__ = [
    "MAB",
    "Grafiti",
]

import math
from collections.abc import Sequence

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


def batch_flatten(*, x_list: Sequence[Tensor], mask: Tensor) -> list[Tensor]:
    r"""Flatten batched time-series tensors according to an observation mask.

    Args:
        x_list: Tensors with shape ``(batch, time, dim)``.
        mask: Mask tensor with shape ``(batch, time, dim)``.

    Returns:
        List of padded flattened tensors, each with shape ``(batch, max_observed)``.
    """
    b, t, d = x_list[0].shape
    m_flat = mask.bool().view(b, t * d)

    observed_counts = m_flat.sum(dim=1)
    k = int(observed_counts.max().to(torch.int64).item())

    indices = torch.arange(k, device=mask.device).expand(b, k)
    mask_indices = indices < observed_counts.unsqueeze(1)

    y_padded: list[Tensor] = []
    for x in x_list:
        x_flat = x.reshape(b, t * d)
        observed_values = x_flat[m_flat]
        y_padded_ = torch.full((b, k), 0, device=mask.device, dtype=x_flat.dtype)
        y_padded_[mask_indices] = observed_values
        y_padded.append(y_padded_)

    return y_padded


def gather_target_embeddings(x: Tensor, *, mask: Tensor) -> Tensor:
    r"""Select and pad edge embeddings at target positions.

    Args:
        x: Input tensor with shape ``(batch, padded_edges, hidden_dim)``.
        mask: Target mask with shape ``(batch, padded_edges)``.

    Returns:
        Target embeddings with shape ``(batch, max_targets, hidden_dim)``.
    """
    b, _, d = x.shape

    observed_counts = mask.sum(dim=1)
    k = int(observed_counts.max().to(torch.int64).item())

    indices = torch.arange(k, device=mask.device).expand(b, k)
    mask_indices = indices < observed_counts.unsqueeze(1)

    observed_values = x[mask.bool()]
    y_padded = torch.full((b, k, d), 0, device=mask.device, dtype=x.dtype)
    y_padded[mask_indices] = observed_values
    return y_padded


def reconstruct_y(*, y_mask: Tensor, y_flat: Tensor, mask_flat: Tensor) -> Tensor:
    r"""Reconstruct a dense tensor from flattened masked values.

    Args:
        y_mask: Boolean mask with shape ``(batch, time, dim)``. True entries mark
            dense positions that should be filled.
        y_flat: Flattened values with shape ``(batch, max_observed)``.
        mask_flat: Boolean mask selecting valid entries from ``y_flat``.

    Returns:
        Reconstructed tensor with shape ``(batch, time, dim)``.
    """
    y_reconstructed = torch.zeros_like(y_mask, dtype=y_flat.dtype)
    # Dense coordinates of all True values in y_mask.
    true_indices = torch.nonzero(y_mask, as_tuple=True)
    y_reconstructed[true_indices] = y_flat[mask_flat.bool()]
    return y_reconstructed


def gather(x: Tensor, inds: Tensor) -> Tensor:
    r"""Gather rows from a batched tensor.

    Args:
        x: Tensor with shape ``(batch, points, hidden_dim)``.
        inds: Indices with shape ``(batch, selected_points)``.

    Returns:
        Gathered tensor with shape ``(batch, selected_points, hidden_dim)``.
    """
    return x.gather(1, inds[:, :, None].repeat(1, 1, x.shape[-1]))


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

        self.edge_init = nn.Linear(2, hidden_dim)
        self.chan_init = nn.Linear(input_dim, hidden_dim)
        self.time_init = nn.Linear(1, hidden_dim)

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
        self, *, batch_size: int, num_channels: int, device: torch.device
    ) -> Tensor:
        r"""Build one-hot channel identifiers.

        Args:
            batch_size: Batch size.
            num_channels: Number of channels.
            device: Device for the resulting tensor.

        Returns:
            One-hot channel encoding with shape ``(batch, dim, dim)``.
        """
        indices = torch.arange(num_channels, device=device).expand(
            batch_size, num_channels
        )
        return F.one_hot(indices, num_classes=num_channels).float()

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
        t: Tensor,  # (..., T)
        c: Tensor,  # (..., D, D)
        t_inds_flat: Tensor,  # (..., K')
        c_inds_flat: Tensor,  # (..., K')
        valid_edge_mask: Tensor,  # (..., K')
    ) -> tuple[Tensor, Tensor]:  # (..., T, K'), (..., D, K')
        r"""Create masks for time and channel attention.

        Args:
            t: Time tensor with shape ``(..., time)``.
            c: One-hot channel encoding with shape ``(..., dim, dim)``.
            t_inds_flat: Flattened time indices with shape ``(..., edges)``.
            c_inds_flat: Flattened channel indices with shape ``(..., edges)``.
            valid_edge_mask: Boolean mask for real entries in the padded flattened
                edge list with shape ``(..., edges)``.

        Returns:
            Tuple containing boolean time and channel masks with shapes
            ``(..., time, edges)`` and ``(..., dim, edges)``.
        """
        device = valid_edge_mask.device
        num_steps = t.shape[-1]
        num_channels = c.shape[-1]

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
        u_encoded = u_encoded * mask.unsqueeze(dim=-1)  # (..., K', M)
        t_encoded = torch.sin(self.time_init(t.unsqueeze(dim=-1)))  # (..., T, M)
        c_encoded = torch.relu(self.chan_init(c_onehot))  # (..., D, M)
        return u_encoded, t_encoded, c_encoded  # (..., K', M), (..., T, M), (..., D, M)

    def forward(
        self,
        time_points: Tensor,
        values: Tensor,
        obs_mask: Tensor,
        target_mask: Tensor,
    ) -> Tensor:
        r"""Encode observed values and target queries.

        Args:
            time_points: Times for observed and target entries with shape
                ``(batch, time)``.
            values: Observed values with shape ``(batch, time, dim)``.
            obs_mask: Observed-value mask with shape ``(batch, time, dim)``.
            target_mask: Target-query mask with shape ``(batch, time, dim)``.

        Returns:
            Target edge embeddings with shape ``(batch, max_targets, hidden_dim)``.
        """
        b, _, d = values.shape
        c_onehot = self._one_hot_channels(
            batch_size=b, num_channels=d, device=time_points.device
        )  # (B, D, D)

        t_inds, c_inds = self._build_indices(time_points, num_channels=d)  # (B, T, D)
        mask = obs_mask + target_mask
        mask_bool = mask.bool()  # (B, T, D)

        # Flatten observed and target edges into padded edge lists. All outputs
        # have shape (B, K'), where K' is the max observed+target edge count.
        t_inds_f, obs_vals, tgt_mask_f, c_inds_f, mask_f = batch_flatten(
            x_list=[t_inds, values, target_mask, c_inds, mask_bool], mask=mask
        )

        target_indicator = (1 - mask_f.float()) + tgt_mask_f  # (B, K')
        edge_input = torch.cat(
            [obs_vals.unsqueeze(-1), target_indicator.unsqueeze(-1)], dim=-1
        )  # (B, K', 2)

        # Masks route each flattened edge to its incident time and channel nodes.
        t_mask, c_mask = self._create_masks(
            t=time_points,
            c=c_onehot,
            t_inds_flat=t_inds_f,
            c_inds_flat=c_inds_f,
            valid_edge_mask=mask_f,
        )  # (..., T, K'), (..., D, K')
        edge_emb, t_emb, c_emb = self._encode_features(
            u_raw=edge_input, t=time_points, c_onehot=c_onehot, mask=mask_f
        )

        for i in range(self.num_layers):
            t_gathered = gather(t_emb, t_inds_f)  # (B, K', M)
            c_gathered = gather(c_emb, c_inds_f)  # (B, K', M)

            channel_context = torch.cat([t_gathered, edge_emb], dim=-1)  # (B, K', 2*M)
            c_emb = self.channel_time_attn[i](
                c_emb, channel_context, channel_context, mask=c_mask
            )  # (B, D, M)
            time_context = torch.cat([c_gathered, edge_emb], dim=-1)  # (B, K', 2*M)
            t_emb = self.time_channel_attn[i](
                t_emb, time_context, time_context, mask=t_mask
            )  # (B, T, M)

            edge_update = torch.cat(
                [edge_emb, t_gathered, c_gathered], dim=-1
            )  # (B, K', 3*M)
            edge_emb = (
                torch.relu(edge_emb + self.edge_nn[i](edge_update)) * mask_f[:, :, None]
            )  # (B, K', M)

        return gather_target_embeddings(edge_emb, mask=tgt_mask_f)  # (B, K, M)
