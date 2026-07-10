r"""Tests for the state-update idempotency criterion."""

import torch
from torch import nn

from linodenet.state_update import InnovationCell, KalmanCell
from linodenet.state_update.base import is_consistent_update
from linodenet.state_update.gradient_based import GradientStepUpdater
from linodenet.state_update.linear import ConstantGain


def test_filter_consistency() -> None:
    r"""Identity-observation linear cells should satisfy $x=y ⟹ F(y, x)=x$."""
    decoder = nn.Identity()
    cell = InnovationCell(16, 16, gate="identity", observation_map=decoder)

    with torch.no_grad():
        assert isinstance(cell.gain, ConstantGain)
        cell.gain.weight.copy_(torch.eye(16))

    assert is_consistent_update(cell, decoder=decoder)


def test_filter_consistency_with_decoder() -> None:
    r"""Decoder-consistent cells should satisfy $ϕ(x)=y ⟹ F(y, x)=x$."""
    decoder = nn.Linear(16, 8, bias=False)
    cell = InnovationCell(8, 16, gate="identity", observation_map=decoder)

    assert is_consistent_update(cell, decoder=decoder)


def test_kalman_filter_consistency_with_decoder() -> None:
    r"""Kalman cells should satisfy $ϕ(x)=y ⟹ F(y, x)=x$."""
    decoder = nn.Linear(16, 8, bias=False)
    cell = KalmanCell(8, 16, gate="identity", observation_map=decoder)

    assert is_consistent_update(cell, decoder=decoder)


def test_gradient_filter_consistency_with_decoder() -> None:
    r"""Gradient-based updaters should satisfy $ϕ(x)=y ⟹ F(y, x)=x$."""
    decoder = nn.Linear(16, 8, bias=True)
    updater = GradientStepUpdater(
        decoder=decoder,
        input_size=8,
        hidden_size=16,
        loss="l2",
        regularizer="l2",
        regularization_strength=2.0,
        step_size=0.7,
    )

    assert is_consistent_update(updater, decoder=decoder)
