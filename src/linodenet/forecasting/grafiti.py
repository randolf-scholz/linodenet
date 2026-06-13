import math

import torch
import torch.nn.functional as F
from torch import nn

__all__ = ["MultiHeadAttention", "ScaledDotProductAttention", "IMAB", "MAB", "MAB2"]


class ScaledDotProductAttention(nn.Module):
    def forward(self, query, key, value, mask=None):
        dk = query.size()[-1]
        scores = query.matmul(key.transpose(-2, -1)) / math.sqrt(dk)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -10e9)

        attention = F.softmax(scores, dim=-1)

        return attention.matmul(value), attention


class MultiHeadAttention(nn.Module):
    def __init__(self, in_features, head_num, bias=True, activation=F.relu):
        """Multi-head attention.
        :param in_features: Size of each input sample.
        :param head_num: Number of heads.
        :param bias: Whether to use the bias term.
        :param activation: The activation after each linear transformation.
        """
        super().__init__()
        if in_features % head_num != 0:
            raise ValueError(
                f"`in_features`({in_features}) should be divisible by `head_num`({head_num})"
            )
        self.in_features = in_features
        self.head_num = head_num
        self.activation = activation
        self.bias = bias
        self.linear_q = nn.Linear(in_features, in_features, bias)
        self.linear_k = nn.Linear(in_features, in_features, bias)
        self.linear_v = nn.Linear(in_features, in_features, bias)
        self.linear_o = nn.Linear(in_features, in_features, bias)

    def forward(self, q, k, v, mask=None):
        q, k, v = self.linear_q(q), self.linear_k(k), self.linear_v(v)
        if self.activation is not None:
            q = self.activation(q)
            k = self.activation(k)
            v = self.activation(v)
        # pdb.set_trace()
        q = self._reshape_to_batches(q)
        k = self._reshape_to_batches(k)
        v = self._reshape_to_batches(v)
        if mask is not None:
            mask = mask.repeat(self.head_num, 1, 1)
        y, attn = ScaledDotProductAttention()(q, k, v, mask)

        y = self._reshape_from_batches(y)

        y = self.linear_o(y)
        # pdb.set_trace()
        if self.activation is not None:
            y = self.activation(y)
        return y, attn

    @staticmethod
    def gen_history_mask(x):
        """Generate the mask that only uses history data.
        :param x: Input tensor.
        :return: The mask.
        """
        batch_size, seq_len, _ = x.size()
        return (
            torch.tril(torch.ones(seq_len, seq_len))
            .view(1, seq_len, seq_len)
            .repeat(batch_size, 1, 1)
        )

    def _reshape_to_batches(self, x):
        batch_size, seq_len, in_feature = x.size()
        sub_dim = in_feature // self.head_num
        return (
            x.reshape(batch_size, seq_len, self.head_num, sub_dim)
            .permute(0, 2, 1, 3)
            .reshape(batch_size * self.head_num, seq_len, sub_dim)
        )

    def _reshape_from_batches(self, x):
        batch_size, seq_len, in_feature = x.size()
        batch_size //= self.head_num
        out_dim = in_feature * self.head_num
        return (
            x.reshape(batch_size, self.head_num, seq_len, in_feature)
            .permute(0, 2, 1, 3)
            .reshape(batch_size, seq_len, out_dim)
        )

    def extra_repr(self):
        return f"in_features={self.in_features}, head_num={self.head_num}, bias={self.bias}, activation={self.activation}"


class MAB2(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, n_dim, num_heads, ln=False):
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

    def forward(self, Q, K, mask=None):
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
        O = O if getattr(self, "ln1", None) is None else self.ln1(O)
        return O


class MAB(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False):
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

    def forward(self, Q, K, mask):
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
        O = O if getattr(self, "ln1", None) is None else self.ln1(O)
        return O


class indMAB(nn.Module):
    def __init__(self, induced_dims, value_dims, hidden_dims, num_heads, ln=False):
        super().__init__()
        self.mab0 = MAB(induced_dims, value_dims, hidden_dims, num_heads, ln=ln)
        self.mab1 = MAB(value_dims, hidden_dims, hidden_dims, num_heads, ln=ln)
        self.head_num = num_heads

    def forward(self, X, Y, att_mask):
        induced_points = X.shape[-2]
        if att_mask is not None:
            mask_r = att_mask.unsqueeze(-2).repeat(self.head_num, induced_points, 1)
            mask_o = att_mask.unsqueeze(-1).repeat(self.head_num, 1, induced_points)
        I = self.mab0(X, Y, mask_r)
        H = self.mab1(Y, I, mask_o)
        return I, H


class IMAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, num_inds, ln=False):
        super().__init__()
        self.I = nn.Parameter(torch.Tensor(1, num_inds, dim_out))
        nn.init.xavier_uniform_(self.I)
        self.mab0 = MAB(dim_out, dim_in, dim_out, num_heads, ln=ln)
        self.mab1 = MAB(dim_in, dim_out, dim_out, num_heads, ln=ln)
        self.head_num = num_heads
        self.num_inds = num_inds

    def forward(self, X, Y, mask1, mask2):

        if mask1 is not None:
            mask_r = mask1.unsqueeze(-2).repeat(self.head_num, self.num_inds, 1)
        H = self.mab0(self.I.repeat(X.size(0), 1, 1), Y, mask_r)
        if mask2 is not None:
            mask_o = mask2.unsqueeze(-1).repeat(self.head_num, 1, self.num_inds)
        return self.mab1(X, H, mask_o)


def batch_flatten(x_list, mask):
    """Flatten a batch of time series based on a mask.

    Args:
        x_list (List[Tensor]): List of tensors with shape (B, T, C)
        mask (Tensor): Mask tensor of shape (B, T, C)

    Returns:
        List[Tensor]: List of flattened tensors with shape (B, K)
    """
    b, t, d = x_list[0].shape
    m_flat = mask.bool().view(b, t * d)

    observed_counts = m_flat.sum(dim=1)
    k = observed_counts.max().to(torch.int64).item()

    indices = torch.arange(k, device=mask.device).expand(b, k)
    mask_indices = indices < observed_counts.unsqueeze(1)

    y_padded = []
    for x in x_list:
        x_flat = x.reshape(b, t * d)
        observed_values = x_flat[m_flat.bool()]
        y_padded_ = torch.full((b, k), 0, device=mask.device, dtype=x_flat.dtype)
        y_padded_[mask_indices] = observed_values
        y_padded.append(y_padded_)

    return y_padded


def hembed(x, mask):
    """Compute the final condidtioning embedding for the profiti model

    Args:
        x (Tensor): Input tensor of shape (B, K', M)
        mask (Tensor): Mask tensor of shape (B, K'); only K values are True

    Returns:
        Tensor: Embedded tensor of shape (B, K, M)
    """
    b, _, d = x.shape

    observed_counts = mask.sum(dim=1)
    k = observed_counts.max().to(torch.int64).item()

    indices = torch.arange(k, device=mask.device).expand(b, k)
    mask_indices = indices < observed_counts.unsqueeze(1)

    observed_values = x[mask.bool()]
    y_padded_ = torch.full((b, k, d), 0, device=mask.device, dtype=x.dtype)
    y_padded_[mask_indices] = observed_values
    return y_padded_


def reconstruct_y(
    Y_mask: torch.Tensor, Y_flat: torch.Tensor, mask_f: torch.Tensor
) -> torch.Tensor:
    """Reconstructs the original tensor Y from its flattened version Y_flat and the mask Y_mask using vectorized operations.

    Args:
        Y_flat: A tensor of shape (B, K), where B is the batch size and K is the maximum
                number of True values in Y_mask across all instances in the batch.
        Y_mask: A boolean tensor of shape (B, T, D), where B is the batch size, T is the
                first dimension of the original Y, and D is the second dimension of the original Y.
                The True values in Y_mask indicate the positions of the elements that were
                flattened into Y_flat.

    Returns:
        Y_reconstructed: A tensor of shape (B, T, D) representing the reconstructed original tensor Y.
    """
    Y_reconstructed = torch.zeros_like(Y_mask, dtype=Y_flat.dtype)

    # Get the indices of True values in Y_mask
    true_indices = torch.nonzero(
        Y_mask, as_tuple=True
    )  # (batch_indices, flattened_indices)
    Y_reconstructed[true_indices] = Y_flat[mask_f.bool()]
    return Y_reconstructed


def gather(x, inds):
    """Gather values from tensor based on indices.

    Args:
        x (Tensor): Tensor of shape (B, P, M)
        inds (Tensor): Indices of shape (B, K')

    Returns:
        Tensor: Gathered tensor of shape (B, K', M)
    """
    return x.gather(1, inds[:, :, None].repeat(1, 1, x.shape[-1]))


class grafiti_(nn.Module):
    """GraFITi model"""

    def __init__(
        self,
        dim: int = 41,
        nkernel: int = 128,
        n_layers: int = 3,
        attn_head: int = 4,
        device: str = "cuda",
    ):
        """Initializing grafiti model

        Args:
            dim (int, optional): number of channels. Defaults to 41.
            nkernel (int, optional): latent dimension size. Defaults to 128.
            n_layers (int, optional): number of grafiti layers. Defaults to 3.
            attn_head (int, optional): number of attention heads. Defaults to 4.
            device (str, optional): "cpu" or "cuda. Defaults to "cuda".
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
    ) -> torch.Tensor:
        """Creating onehot encoding of channel ids

        Args:
            batch_size (int): B
            num_channels (int): D
            device (torch.device): GPU or CPU

        Returns:
            torch.Tensor: onehot encoding of channels (B, D, D)
        """
        indices = torch.arange(num_channels, device=device).expand(
            batch_size, num_channels
        )
        return F.one_hot(indices, num_classes=num_channels).float()

    def _build_indices(
        self,
        time_steps: torch.Tensor,  # shape: (B, T, 1)
        num_channels: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Builds index tensors for time steps and channel IDs.

        Args:
            time_steps (torch.Tensor): Input tensor with shape (B, T, 1)
            num_channels (int): Number of channels (D)
            device (torch.device): CPU or GPU

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                - t_inds (torch.Tensor): Time indices of shape (B, T, D)
                - c_inds (torch.Tensor): Channel indices of shape (B, T, D)
        """
        b, t = time_steps.shape[0], time_steps.shape[1]

        # Create time indices (B, T, D)
        t_inds = (
            torch.arange(t, device=device).expand(b, num_channels, -1).permute(0, 2, 1)
        )

        # Create channel indices (B, T, D)
        c_inds = torch.arange(num_channels, device=device).expand(b, t, -1)

        return t_inds, c_inds

    def _create_masks(
        self,
        mk: torch.Tensor,
        t_inds_flat: torch.Tensor,
        c_inds_flat: torch.Tensor,
        t: torch.Tensor,
        c: torch.Tensor,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Creating masks for time and channel attentions in grafiti

        Args:
            mk (torch.Tensor): flattened mask; (B, K')
            t_inds_flat (torch.Tensor): flattened time indices; (B, K')
            c_inds_flat (torch.Tensor): flattened channel indices: (B, K')
            t (torch.Tensor): time points; (B, T)
            c (torch.Tensor): onhot channel encoding; (B, D, D)
            device (torch.Device): GPU or CPU

        Returns:
            tuple[torch.Tensor, torch.Tensor]
            t_mask: time attn mask (B, T, K')
            c_mask: channel attn mask (B, D, K')
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
        u_raw: torch.Tensor,
        t: torch.Tensor,
        c_onehot: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encoding edge, time node and channel node features

        Args:
            u_raw (torch.Tensor): input edge feature (B, K', 2)
            t (torch.Tensor): time node feature (B, T, 1)
            c_onehot (torch.Tensor): channel node feature (B, C, C)
            mask (torch.Tensor): input mask (B, K')

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            Encoded edge features (B, K', M),
            encoded time features (B, T, M),
            encoded channel features (B, C, M)

        """
        u_encoded = self.relu(self.edge_init(u_raw)) * mask[:, :, None]  # (B, K', M)
        t_encoded = torch.sin(self.time_init(t))  # (B, T, M)
        c_encoded = self.relu(self.chan_init(c_onehot))  # (B, C, M)
        return u_encoded, t_encoded, c_encoded

    def gatherhedge(self, U_, indices, mk_, shapes):
        X = torch.zeros([shapes[0], shapes[1], shapes[2], U_.shape[-1]]).to(U_.device)
        values = U_[mk_.to(torch.bool)]
        X[indices] = values
        return X

    def forward(
        self,
        time_points: torch.Tensor,
        values: torch.Tensor,
        obs_mask: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> torch.Tensor:
        """GraFITi model

        Args:
            time_points: time_points have both observed and target times; Tensor (B, T)
            values: Observed values; Tensor (B, T, D)
            obs_mask: Observed values mask; Tensor (B, T, D)
            target_mask: Target values mask; Tensor (B, T, D)

        Returns:
            yhat: Predictions; Tensor (B, T, D)
        """
        b, _, d = values.shape
        t = time_points.unsqueeze(-1)  # (B, T, 1)
        c_onehot = self._one_hot_channels(b, d, device=t.device)  # (B, D, D)

        t_inds, c_inds = self._build_indices(
            time_points, d, t.device
        )  # t_inds (B, T, D), c_inds (B, T, D)

        mask = obs_mask + target_mask
        mask_bool = mask.bool()  # (B, T, D)
        indices = torch.where(mask_bool)  # (B, K')
        flattened = batch_flatten(
            [t_inds, values, target_mask, c_inds, mask_bool], mask
        )
        t_inds_f, obs_vals, tgt_mask_f, c_inds_f, mask_f = (
            flattened  # all are of shape (B, K'); K' = K+N; K = Number of queries, N' = Total number of observations
        )

        target_indicator = (1 - mask_f.float()) + tgt_mask_f  # target indicator (B, K')
        edge_input = torch.cat(
            [obs_vals.unsqueeze(-1), target_indicator.unsqueeze(-1)], dim=-1
        )  # edge feature (B, K', 2)

        t_mask, c_mask = self._create_masks(
            mask_f, t_inds_f, c_inds_f, t, c_onehot, t.device
        )  # creating masks for attention for time nodes (B, T, K') and channel nodes (B, C, K') respectively

        edge_emb, t_emb, c_emb = self._encode_features(
            edge_input, t, c_onehot, mask_f
        )  # encoding edge features (B, K', M), time node features (B, T, M), channel node features (B, C, M); M is the embedding dimension

        for i in range(self.n_layers):
            t_gathered = gather(t_emb, t_inds_f)  # (B, K', M)

            c_gathered = gather(c_emb, c_inds_f)  # (B, K', M)

            c_emb = self.channel_time_attn[i](
                c_emb, torch.cat([t_gathered, edge_emb], -1), c_mask
            )  # updating channel embedding (B, C, M)
            t_emb = self.time_channel_attn[i](
                t_emb, torch.cat([c_gathered, edge_emb], -1), t_mask
            )  # updating time embedding (B, T, M)

            edge_update = torch.cat(
                [edge_emb, t_gathered, c_gathered], dim=-1
            )  # (B, K', 3*M)

            edge_emb = (
                self.relu(edge_emb + self.edge_nn[i](edge_update)) * mask_f[:, :, None]
            )  # updating edge embedding (B, K', M)
        # Final output layer
        h = hembed(edge_emb, tgt_mask_f)  # (B, K))
        return h


class GraFITi(nn.Module):
    def __init__(
        self, input_dim=41, attn_head=4, latent_dim=128, n_layers=2, device="cuda"
    ):
        super().__init__()
        self.dim = input_dim  # input dimensions
        self.attn_head = attn_head  # no. of attention heads
        self.latent_dim = latent_dim  # latend dimension
        self.n_layers = n_layers  # number of grafiti layers
        self.device = device  # cpu or gpu
        self.grafiti_ = grafiti_(
            self.dim, self.latent_dim, self.n_layers, self.attn_head, device=device
        )  # applying grafiti

    def forward(self, x_time, x_vals, x_mask, y_mask):
        """Forward pass of the GraFITi model.
        Parameters:
        x_time: Tensor - Time points of the observations.
        x_vals: Tensor - Values of the observations.
        x_mask: Tensor - Mask for the observations.
        y_mask: Tensor - Mask for the queries.

        Returns:
        h: Tensor - Output of the GraFITi model; conditioning module for profiti model.
        """
        h = self.grafiti_(x_time, x_vals, x_mask, y_mask)
        return h
