__all__ = ["GrafitiEncoder"]

from torch import Tensor, nn

from linodenet.forecasting.grafiti import Grafiti


class GrafitiEncoder(nn.Module):
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
        self.grafiti_ = Grafiti(
            self.dim,
            self.latent_dim,
            self.n_layers,
            self.attn_head,
            device=device,
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
