r"""GraFITi layers for irregular time-series forecasting."""

__all__ = [
    "GraFITi",
    "IMAB",
    "MAB",
    "MAB2",
    "MultiHeadAttention",
    "ScaledDotProductAttention",
]

import math
from collections.abc import Callable, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class ScaledDotProductAttention(nn.Module):
    r"""Scaled dot-product attention."""

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        r"""Apply attention to a query, key, and value tensor.

        Args:
            query: Query tensor with shape ``(batch, query_len, dim)``.
            key: Key tensor with shape ``(batch, key_len, dim)``.
            value: Value tensor with shape ``(batch, key_len, dim)``.
            mask: Optional attention mask with shape ``(batch, query_len, key_len)``.

        Returns:
            Tuple containing the attended values and attention weights.
        """
        dk = query.size()[-1]
        scores = query.matmul(key.transpose(-2, -1)) / math.sqrt(dk)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -10e9)

        attention = F.softmax(scores, dim=-1)
        return attention.matmul(value), attention


class MultiHeadAttention(nn.Module):
    r"""Multi-head attention layer."""

    def __init__(
        self,
        in_features: int,
        head_num: int,
        bias: bool = True,
        activation: Callable[[Tensor], Tensor] | None = F.relu,
    ) -> None:
        r"""Initialize the attention projections.

        Args:
            in_features: Feature dimension of each input token.
            head_num: Number of attention heads.
            bias: Whether linear projections include bias terms.
            activation: Optional activation applied after each linear projection.

        Raises:
            ValueError: If ``in_features`` is not divisible by ``head_num``.
        """
        super().__init__()
        if in_features % head_num != 0:
            raise ValueError(
                f"`in_features`({in_features}) should be divisible by "
                f"`head_num`({head_num})"
            )
        self.in_features = in_features
        self.head_num = head_num
        self.activation = activation
        self.bias = bias
        self.linear_q = nn.Linear(in_features, in_features, bias)
        self.linear_k = nn.Linear(in_features, in_features, bias)
        self.linear_v = nn.Linear(in_features, in_features, bias)
        self.linear_o = nn.Linear(in_features, in_features, bias)

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        r"""Apply multi-head attention.

        Args:
            q: Query tensor with shape ``(batch, query_len, in_features)``.
            k: Key tensor with shape ``(batch, key_len, in_features)``.
            v: Value tensor with shape ``(batch, key_len, in_features)``.
            mask: Optional attention mask with shape ``(batch, query_len, key_len)``.

        Returns:
            Tuple containing the attended values and attention weights.
        """
        q, k, v = self.linear_q(q), self.linear_k(k), self.linear_v(v)
        if self.activation is not None:
            q = self.activation(q)
            k = self.activation(k)
            v = self.activation(v)
        q = self._reshape_to_batches(q)
        k = self._reshape_to_batches(k)
        v = self._reshape_to_batches(v)
        if mask is not None:
            mask = mask.repeat(self.head_num, 1, 1)
        y, attn = ScaledDotProductAttention()(q, k, v, mask)
        y = self._reshape_from_batches(y)
        y = self.linear_o(y)
        if self.activation is not None:
            y = self.activation(y)
        return y, attn

    @staticmethod
    def gen_history_mask(x: Tensor) -> Tensor:
        r"""Generate a causal mask for history-only attention.

        Args:
            x: Input tensor with shape ``(batch, seq_len, dim)``.

        Returns:
            Lower-triangular mask with shape ``(batch, seq_len, seq_len)``.
        """
        batch_size, seq_len, _ = x.size()
        return (
            torch.tril(torch.ones(seq_len, seq_len, device=x.device))
            .view(1, seq_len, seq_len)
            .repeat(batch_size, 1, 1)
        )

    def _reshape_to_batches(self, x: Tensor) -> Tensor:
        batch_size, seq_len, in_feature = x.size()
        sub_dim = in_feature // self.head_num
        return (
            x.reshape(batch_size, seq_len, self.head_num, sub_dim)
            .permute(0, 2, 1, 3)
            .reshape(batch_size * self.head_num, seq_len, sub_dim)
        )

    def _reshape_from_batches(self, x: Tensor) -> Tensor:
        batch_size, seq_len, in_feature = x.size()
        batch_size //= self.head_num
        out_dim = in_feature * self.head_num
        return (
            x.reshape(batch_size, self.head_num, seq_len, in_feature)
            .permute(0, 2, 1, 3)
            .reshape(batch_size, seq_len, out_dim)
        )

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, head_num={self.head_num}, "
            f"bias={self.bias}, activation={self.activation}"
        )


class MAB2(nn.Module):
    r"""Multi-head attention block with configurable hidden dimension."""

    def __init__(
        self,
        dim_Q: int,
        dim_K: int,
        dim_V: int,
        n_dim: int,
        num_heads: int,
        ln: bool = False,
    ) -> None:
        super().__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.n_dim = n_dim
        self.fc_q = nn.Linear(dim_Q, n_dim)
        self.fc_k = nn.Linear(dim_K, n_dim)
        self.fc_v = nn.Linear(dim_K, n_dim)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(n_dim, n_dim)

    def forward(self, Q: Tensor, K: Tensor, mask: Tensor | None = None) -> Tensor:
        r"""Apply the attention block.

        Args:
            Q: Query tensor with shape ``(batch, query_len, dim_Q)``.
            K: Key/value tensor with shape ``(batch, key_len, dim_K)``.
            mask: Optional attention mask with shape ``(batch, query_len, key_len)``.

        Returns:
            Updated query embeddings with shape ``(batch, query_len, n_dim)``.
        """
        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)

        dim_split = self.n_dim // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K = torch.cat(K.split(dim_split, 2), 0)
        V = torch.cat(V.split(dim_split, 2), 0)

        Att_mat = Q_.bmm(K.transpose(1, 2)) / math.sqrt(self.n_dim)
        if mask is not None:
            Att_mat = Att_mat.masked_fill(mask.repeat(self.num_heads, 1, 1) == 0, -10e9)
        A = torch.softmax(Att_mat, 2)
        O = torch.cat((Q_ + A.bmm(V)).split(Q.size(0), 0), 2)
        O = O if getattr(self, "ln0", None) is None else self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        return O if getattr(self, "ln1", None) is None else self.ln1(O)


class MAB(nn.Module):
    r"""Multi-head attention block."""

    def __init__(
        self,
        dim_Q: int,
        dim_K: int,
        dim_V: int,
        num_heads: int,
        ln: bool = False,
    ) -> None:
        super().__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)

    def forward(self, Q: Tensor, K: Tensor, mask: Tensor | None = None) -> Tensor:
        r"""Apply the attention block.

        Args:
            Q: Query tensor with shape ``(batch, query_len, dim_Q)``.
            K: Key/value tensor with shape ``(batch, key_len, dim_K)``.
            mask: Optional attention mask.

        Returns:
            Updated query embeddings with shape ``(batch, query_len, dim_V)``.
        """
        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)

        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)
        Att_mat = Q_.bmm(K_.transpose(1, 2)) / math.sqrt(self.dim_V)
        if mask is not None:
            Att_mat = Att_mat.masked_fill(mask == 0, -10e9)
        A = torch.softmax(Att_mat, 2)
        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        O = O if getattr(self, "ln0", None) is None else self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        return O if getattr(self, "ln1", None) is None else self.ln1(O)


class indMAB(nn.Module):
    r"""Induced multi-head attention block."""

    def __init__(
        self,
        induced_dims: int,
        value_dims: int,
        hidden_dims: int,
        num_heads: int,
        ln: bool = False,
    ) -> None:
        super().__init__()
        self.mab0 = MAB(induced_dims, value_dims, hidden_dims, num_heads, ln=ln)
        self.mab1 = MAB(value_dims, hidden_dims, hidden_dims, num_heads, ln=ln)
        self.head_num = num_heads

    def forward(
        self,
        X: Tensor,
        Y: Tensor,
        att_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        r"""Apply induced attention from ``X`` to ``Y``.

        Args:
            X: Inducing point tensor.
            Y: Value tensor.
            att_mask: Optional attention mask.

        Returns:
            Tuple containing induced and output embeddings.
        """
        mask_r: Tensor | None = None
        mask_o: Tensor | None = None
        induced_points = X.shape[-2]
        if att_mask is not None:
            mask_r = att_mask.unsqueeze(-2).repeat(self.head_num, induced_points, 1)
            mask_o = att_mask.unsqueeze(-1).repeat(self.head_num, 1, induced_points)
        I = self.mab0(X, Y, mask_r)
        H = self.mab1(Y, I, mask_o)
        return I, H


class IMAB(nn.Module):
    r"""Induced multi-head attention block with learned inducing points."""

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        num_heads: int,
        num_inds: int,
        ln: bool = False,
    ) -> None:
        super().__init__()
        self.I = nn.Parameter(torch.tensor(1, num_inds, dim_out))
        nn.init.xavier_uniform_(self.I)
        self.mab0 = MAB(dim_out, dim_in, dim_out, num_heads, ln=ln)
        self.mab1 = MAB(dim_in, dim_out, dim_out, num_heads, ln=ln)
        self.head_num = num_heads
        self.num_inds = num_inds

    def forward(
        self,
        X: Tensor,
        Y: Tensor,
        mask1: Tensor | None = None,
        mask2: Tensor | None = None,
    ) -> Tensor:
        r"""Apply induced attention from learned inducing points.

        Args:
            X: Query tensor.
            Y: Key/value tensor.
            mask1: Optional mask for the inducing-to-value attention.
            mask2: Optional mask for the query-to-inducing attention.

        Returns:
            Updated query embeddings.
        """
        mask_r: Tensor | None = None
        mask_o: Tensor | None = None
        if mask1 is not None:
            mask_r = mask1.unsqueeze(-2).repeat(self.head_num, self.num_inds, 1)
        H = self.mab0(self.I.repeat(X.size(0), 1, 1), Y, mask_r)
        if mask2 is not None:
            mask_o = mask2.unsqueeze(-1).repeat(self.head_num, 1, self.num_inds)
        return self.mab1(X, H, mask_o)


def batch_flatten(x_list: Sequence[Tensor], mask: Tensor) -> list[Tensor]:
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


def hembed(x: Tensor, mask: Tensor) -> Tensor:
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


def reconstruct_y(Y_mask: Tensor, Y_flat: Tensor, mask_f: Tensor) -> Tensor:
    r"""Reconstruct a dense tensor from flattened masked values.

    Args:
        Y_mask: Boolean mask with shape ``(batch, time, dim)``. True entries mark
            dense positions that should be filled.
        Y_flat: Flattened values with shape ``(batch, max_observed)``.
        mask_f: Boolean mask selecting valid entries from ``Y_flat``.

    Returns:
        Reconstructed tensor with shape ``(batch, time, dim)``.
    """
    Y_reconstructed = torch.zeros_like(Y_mask, dtype=Y_flat.dtype)
    # Dense coordinates of all True values in Y_mask.
    true_indices = torch.nonzero(Y_mask, as_tuple=True)
    Y_reconstructed[true_indices] = Y_flat[mask_f.bool()]
    return Y_reconstructed


def gather(x: Tensor, inds: Tensor) -> Tensor:
    r"""Gather rows from a batched tensor.

    Args:
        x: Tensor with shape ``(batch, points, hidden_dim)``.
        inds: Indices with shape ``(batch, selected_points)``.

    Returns:
        Gathered tensor with shape ``(batch, selected_points, hidden_dim)``.
    """
    return x.gather(1, inds[:, :, None].repeat(1, 1, x.shape[-1]))


class grafiti_(nn.Module):
    r"""GraFITi encoder for observed and target time-series entries."""

    def __init__(
        self,
        dim: int = 41,
        nkernel: int = 128,
        n_layers: int = 3,
        attn_head: int = 4,
        device: str = "cuda",
    ) -> None:
        r"""Initialize the GraFITi encoder.

        Args:
            dim: Number of channels.
            nkernel: Latent dimension size.
            n_layers: Number of GraFITi layers.
            attn_head: Number of attention heads.
            device: Device name retained for compatibility with the reference API.
        """
        super().__init__()
        self.nkernel = nkernel
        self.nheads = attn_head
        self.device = device
        self.n_layers = n_layers

        self.edge_init = nn.Linear(2, nkernel)
        self.chan_init = nn.Linear(dim, nkernel)
        self.time_init = nn.Linear(1, nkernel)

        self.channel_time_attn = nn.ModuleList(
            [
                MAB2(nkernel, 2 * nkernel, 2 * nkernel, nkernel, attn_head)
                for _ in range(n_layers)
            ]
        )
        self.time_channel_attn = nn.ModuleList(
            [
                MAB2(nkernel, 2 * nkernel, 2 * nkernel, nkernel, attn_head)
                for _ in range(n_layers)
            ]
        )
        self.edge_nn = nn.ModuleList(
            [nn.Linear(3 * nkernel, nkernel) for _ in range(n_layers)]
        )

        self.output = nn.Linear(3 * nkernel, nkernel)
        self.relu = nn.ReLU()

    def _one_hot_channels(
        self, batch_size: int, num_channels: int, device: torch.device
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
        time_steps: Tensor,  # (B, T)
        num_channels: int,
        device: torch.device,
    ) -> tuple[Tensor, Tensor]:
        r"""Build dense time and channel index tensors.

        Args:
            time_steps: Time tensor with shape ``(batch, time)``.
            num_channels: Number of channels.
            device: Device for the resulting tensors.

        Returns:
            Tuple containing time indices and channel indices, both with shape
            ``(batch, time, dim)``.
        """
        b, t = time_steps.shape[0], time_steps.shape[1]

        # Time indices identify the time node attached to each dense edge.
        t_inds = (
            torch.arange(t, device=device).expand(b, num_channels, -1).permute(0, 2, 1)
        )  # (B, T, D)

        # Channel indices identify the channel node attached to each dense edge.
        c_inds = torch.arange(num_channels, device=device).expand(b, t, -1)  # (B, T, D)
        return t_inds, c_inds

    def _create_masks(
        self,
        mk: Tensor,
        t_inds_flat: Tensor,
        c_inds_flat: Tensor,
        t: Tensor,
        c: Tensor,
        device: torch.device,
    ) -> tuple[Tensor, Tensor]:
        r"""Create masks for time and channel attention.

        Args:
            mk: Flattened observation/target mask with shape ``(batch, edges)``.
            t_inds_flat: Flattened time indices with shape ``(batch, edges)``.
            c_inds_flat: Flattened channel indices with shape ``(batch, edges)``.
            t: Time tensor with shape ``(batch, time, 1)``.
            c: One-hot channel encoding with shape ``(batch, dim, dim)``.
            device: Device for generated channel indices.

        Returns:
            Tuple containing time and channel masks with shapes
            ``(batch, time, edges)`` and ``(batch, dim, edges)``.
        """
        b, t_len = t.shape[:2]
        num_channels = c.shape[1]
        indices = torch.arange(num_channels, device=device).expand(b, num_channels)
        c_mask = (indices[:, :, None] == c_inds_flat[:, None, :]).float() * mk[
            :, None, :
        ]
        t_seq = torch.arange(t_len, device=t.device)[None, :, None]
        t_mask = (t_inds_flat[:, None, :] == t_seq).float() * mk[:, None, :]
        return t_mask, c_mask

    def _encode_features(
        self,
        u_raw: Tensor,
        t: Tensor,
        c_onehot: Tensor,
        mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        r"""Encode edge, time-node, and channel-node features.

        Args:
            u_raw: Edge features with shape ``(batch, edges, 2)``.
            t: Time-node features with shape ``(batch, time, 1)``.
            c_onehot: Channel-node features with shape ``(batch, dim, dim)``.
            mask: Flattened observation/target mask with shape ``(batch, edges)``.

        Returns:
            Encoded edge, time, and channel features.
        """
        u_encoded = self.relu(self.edge_init(u_raw)) * mask[:, :, None]  # (B, K', M)
        t_encoded = torch.sin(self.time_init(t))  # (B, T, M)
        c_encoded = self.relu(self.chan_init(c_onehot))  # (B, D, M)
        return u_encoded, t_encoded, c_encoded

    def gatherhedge(
        self,
        U_: Tensor,
        indices: tuple[Tensor, ...],
        mk_: Tensor,
        shapes: tuple[int, int, int],
    ) -> Tensor:
        r"""Scatter flattened edge embeddings back into a dense hedge tensor.

        Args:
            U_: Flattened edge embeddings.
            indices: Dense indices produced by ``torch.where``.
            mk_: Mask selecting valid flattened entries.
            shapes: Leading dense output shape ``(batch, time, dim)``.

        Returns:
            Dense hedge tensor with shape ``(*shapes, hidden_dim)``.
        """
        X = torch.zeros((*shapes, U_.shape[-1]), device=U_.device, dtype=U_.dtype)
        values = U_[mk_.to(torch.bool)]
        X[indices] = values
        return X

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
            Target edge embeddings with shape ``(batch, max_targets, nkernel)``.
        """
        b, _, d = values.shape
        t = time_points.unsqueeze(-1)  # (B, T, 1)
        c_onehot = self._one_hot_channels(b, d, device=t.device)  # (B, D, D)

        t_inds, c_inds = self._build_indices(time_points, d, t.device)  # (B, T, D)
        mask = obs_mask + target_mask
        mask_bool = mask.bool()  # (B, T, D)

        # Flatten observed and target edges into padded edge lists. All outputs
        # have shape (B, K'), where K' is the max observed+target edge count.
        t_inds_f, obs_vals, tgt_mask_f, c_inds_f, mask_f = batch_flatten(
            [t_inds, values, target_mask, c_inds, mask_bool], mask
        )

        target_indicator = (1 - mask_f.float()) + tgt_mask_f  # (B, K')
        edge_input = torch.cat(
            [obs_vals.unsqueeze(-1), target_indicator.unsqueeze(-1)], dim=-1
        )  # (B, K', 2)

        # Masks route each flattened edge to its incident time and channel nodes.
        t_mask, c_mask = self._create_masks(
            mask_f, t_inds_f, c_inds_f, t, c_onehot, t.device
        )  # t_mask: (B, T, K'), c_mask: (B, D, K')
        edge_emb, t_emb, c_emb = self._encode_features(edge_input, t, c_onehot, mask_f)

        for i in range(self.n_layers):
            t_gathered = gather(t_emb, t_inds_f)  # (B, K', M)
            c_gathered = gather(c_emb, c_inds_f)  # (B, K', M)

            c_emb = self.channel_time_attn[i](
                c_emb, torch.cat([t_gathered, edge_emb], -1), c_mask
            )  # (B, D, M)
            t_emb = self.time_channel_attn[i](
                t_emb, torch.cat([c_gathered, edge_emb], -1), t_mask
            )  # (B, T, M)

            edge_update = torch.cat(
                [edge_emb, t_gathered, c_gathered], dim=-1
            )  # (B, K', 3*M)
            edge_emb = (
                self.relu(edge_emb + self.edge_nn[i](edge_update)) * mask_f[:, :, None]
            )  # (B, K', M)

        return hembed(edge_emb, tgt_mask_f)  # (B, K, M)


class GraFITi(nn.Module):
    r"""GraFITi conditioning encoder for ProFITi-style models."""

    def __init__(
        self,
        input_dim: int = 41,
        attn_head: int = 4,
        latent_dim: int = 128,
        n_layers: int = 2,
        device: str = "cuda",
    ) -> None:
        super().__init__()
        self.dim = input_dim
        self.attn_head = attn_head
        self.latent_dim = latent_dim
        self.n_layers = n_layers
        self.device = device
        self.grafiti_ = grafiti_(
            self.dim, self.latent_dim, self.n_layers, self.attn_head, device=device
        )

    def forward(
        self,
        x_time: Tensor,
        x_vals: Tensor,
        x_mask: Tensor,
        y_mask: Tensor,
    ) -> Tensor:
        r"""Encode observations into target conditioning embeddings.

        Args:
            x_time: Observation and query times with shape ``(batch, time)``.
            x_vals: Observed values with shape ``(batch, time, dim)``.
            x_mask: Observation mask with shape ``(batch, time, dim)``.
            y_mask: Query mask with shape ``(batch, time, dim)``.

        Returns:
            Conditioning embeddings with shape ``(batch, max_targets, latent_dim)``.
        """
        return self.grafiti_(x_time, x_vals, x_mask, y_mask)
