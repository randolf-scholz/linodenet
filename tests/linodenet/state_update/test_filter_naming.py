r"""Compatibility tests for the filter-oriented public API."""

from torch.nn import GRUCell

from linodenet.state_update import (
    CELLS,
    STATE_UPDATERS,
    GRU_Update,
    LinearUpdate,
    MissingValueUpdate,
    StateUpdater,
    UpdateSequence,
    is_square_state_updater,
)


def test_filter_aliases_resolve_to_existing_implementations() -> None:
    r"""The update-oriented names should resolve to the exported implementations."""
    assert UpdateSequence.__name__ == "UpdateSequence"


def test_filter_registry_exposes_legacy_and_filter_names() -> None:
    r"""The state updater registry should expose the canonical names."""
    assert STATE_UPDATERS["GRUCell"] is GRUCell
    assert STATE_UPDATERS["GRU_Update"] is GRU_Update
    assert STATE_UPDATERS["LinearUpdate"] is LinearUpdate
    assert STATE_UPDATERS["MissingValueUpdate"] is MissingValueUpdate
    assert CELLS is STATE_UPDATERS


def test_filter_protocol_covers_torch_cells() -> None:
    r"""PyTorch recurrent cells should satisfy the general state-updater protocol."""
    gru_cell = GRUCell(3, 5)
    gru_filter = GRU_Update(3, 5)

    assert isinstance(gru_cell, StateUpdater)
    assert isinstance(gru_filter, StateUpdater)
    assert not is_square_state_updater(gru_cell)
