r"""Cell protocol test."""

from torch.nn import GRUCell, LSTMCell, Module, RNNCell, RNNCellBase

CELLS = {
    "RNNCell": RNNCell,
    "GRUCell": GRUCell,
    "LSTMCell": LSTMCell,
}


def test_shared_interface() -> None:
    r"""Check the shared interface of all cells."""
    shared_interface = set.intersection(*(set(dir(cell)) for cell in CELLS.values()))
    excluded_members = set(dir(Module))
    print(shared_interface - excluded_members)
