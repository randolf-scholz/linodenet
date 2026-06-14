r"""ProFITi-style forecasting components."""

__all__ = ["GrafitiEncoder"]

from torch import Tensor, nn

from .grafiti import Grafiti


class GrafitiEncoder(nn.Module):
    r"""GraFITi conditioning encoder for ProFITi-style models."""

    def __init__(
        self,
        *,
        input_dim: int = 41,
        num_heads: int = 4,
        latent_dim: int = 128,
        num_layers: int = 2,
        device: str = "cuda",
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.device = device
        self.grafiti_ = Grafiti(
            input_dim=input_dim,
            hidden_dim=latent_dim,
            num_layers=num_layers,
            num_heads=num_heads,
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
