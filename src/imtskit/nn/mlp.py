r"""Multi-Layer Perceptron."""

__all__ = [
    # Classes
    "MLP",
]


from typing import Optional

from torch import nn


class MLP(nn.Sequential):
    r"""A standard Multi-Layer Perceptron."""

    @property
    def config(self) -> dict:
        r"""Configuration dictionary."""
        return {
            "inputs_size": self.inputs_size,
            "output_size": self.output_size,
            "hidden_size": self.hidden_size,
            "num_blocks": self.num_blocks,
            "dropout": self.dropout,
        }

    def __init__(
        self,
        inputs_size: int,
        output_size: int,
        *,
        hidden_size: Optional[int] = None,
        num_blocks: int = 2,
        dropout: float = 0.2,
    ) -> None:
        self.dropout = dropout
        self.hidden_size = inputs_size if hidden_size is None else hidden_size
        self.inputs_size = inputs_size
        self.output_size = output_size
        self.num_blocks = num_blocks

        layers: list[nn.Module] = []

        # input layer
        layer = nn.Linear(self.inputs_size, self.hidden_size)
        nn.init.kaiming_normal_(layer.weight, nonlinearity="linear")
        nn.init.kaiming_normal_(layer.bias[None], nonlinearity="linear")
        layers.append(layer)

        # hidden layers
        for _ in range(num_blocks - 1):
            # initialize the layers
            act = nn.ReLU()
            drop = nn.Dropout(self.dropout)
            linear = nn.Linear(self.hidden_size, self.hidden_size)
            nn.init.kaiming_normal_(linear.weight, nonlinearity="relu")
            nn.init.kaiming_normal_(linear.bias[None], nonlinearity="relu")
            # Add the block
            layers.extend([act, drop, linear])

        # output_layer
        act = nn.ReLU()
        drop = nn.Dropout(self.dropout)
        linear = nn.Linear(self.hidden_size, self.output_size)
        nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
        nn.init.kaiming_normal_(layer.bias[None], nonlinearity="relu")
        # Add the output block
        layers.extend([act, drop, linear])

        super().__init__(*layers)
