r"""PyTorch recurrent cells exposed under state-update-oriented names."""

__all__ = [
    "RNN_Update",
    "GRU_Update",
    "LSTM_Update",
]

from torch.nn import GRUCell, LSTMCell, RNNCell


class RNN_Update(RNNCell):
    r"""State-update-oriented alias for :class:`torch.nn.RNNCell`."""


class GRU_Update(GRUCell):
    r"""State-update-oriented alias for :class:`torch.nn.GRUCell`."""


class LSTM_Update(LSTMCell):
    r"""State-update-oriented alias for :class:`torch.nn.LSTMCell`."""
