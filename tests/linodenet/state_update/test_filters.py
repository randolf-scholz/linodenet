r"""Tests for the state-update idempotency criterion."""

from linodenet.state_update import LinearCell
from linodenet.state_update.base import is_idempotent_update


def test_filter_consistency() -> None:
    r"""Direct-observation linear cells should satisfy $x=y ⟹ F(y, x)=x$."""
    cell = LinearCell.from_direct_observation_model(16, gate="identity")

    assert is_idempotent_update(cell)
