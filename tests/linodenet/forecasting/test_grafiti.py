r"""Tests for GraFITi components."""

import torch
from torch.testing import assert_close

from linodenet.forecasting.grafiti import Grafiti
from linodenet.forecasting.utils import BatchedTripletArgs


def test_grafiti_triplet_matches_combined_forward() -> None:
    r"""Check that sparse and combined GraFITi inputs produce the same embeddings."""
    torch.manual_seed(0)
    model = Grafiti(input_dim=3, hidden_dim=8, num_layers=2, num_heads=2)
    args = BatchedTripletArgs(
        context_times=torch.tensor(
            [
                [1.0, 3.0, 5.0, 7.0, torch.nan],
                [0.0, 2.0, 4.0, torch.nan, torch.nan],
            ]
        ),
        context_channels=torch.tensor(
            [
                [0, 2, 1, 0, -1],
                [1, 0, 2, -1, -1],
            ]
        ),
        context_values=torch.tensor(
            [
                [10.0, 32.0, 51.0, 70.0, torch.nan],
                [1.0, 20.0, 22.0, torch.nan, torch.nan],
            ]
        ),
        query_times=torch.tensor(
            [
                [2.0, 4.0, 6.0, torch.nan],
                [1.0, 3.0, 5.0, 7.0],
            ]
        ),
        query_channels=torch.tensor(
            [
                [0, 1, 2, -1],
                [0, 1, 2, 1],
            ]
        ),
        query_values=torch.tensor(
            [
                [200.0, 410.0, 620.0, torch.nan],
                [100.0, 310.0, 520.0, 710.0],
            ]
        ),
    )
    combined = args.to_combined()

    expected = model.forward_combined(
        combined.times,
        combined.context_values,
        combined.query_mask,
    )
    actual = model.forward_triplet(
        args.context_times,
        args.context_channels,
        args.context_values,
        args.query_times,
        args.query_channels,
    )

    assert_close(actual, expected)
