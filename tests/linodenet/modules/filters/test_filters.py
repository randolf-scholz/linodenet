r"""Test if filters satisfy idempotence property."""

import logging

import pytest
import torch

from linodenet.config import PROJECT
from linodenet.constants import NAN
from linodenet.filters import ResidualFilter

RESULT_DIR = PROJECT.RESULTS_DIR[__file__]


@pytest.mark.flaky(reruns=3)
def test_filter_consistency() -> None:
    r"""Check whether idempotency holds."""
    logger = logging.getLogger(f"{__name__}/test_filter_consistency")
    logger.info("Testing idempotency.")
    batch_dim, m, n = (3, 4, 5), 100, 100
    x = torch.randn(*batch_dim, n)
    y = torch.randn(*batch_dim, m)
    mask = y > 0
    y[mask] = NAN

    # ## Test KalmanCell
    # model = KalmanCell(
    #     input_size=n, hidden_size=m, autoregressive=True, activation="ReLU"
    # )
    # result = model(y, x)
    # assert not torch.isnan(result).any(), "Output contains NANs! ❌ "
    # logger.info("KalmanCell: No NaN outputs ✔ ")
    #
    # ## verify IDP condition
    # y[~mask] = x[~mask]
    # assert torch.allclose(x, model(y, x)), "Idempotency failed! ❌ "
    # logger.info("KalmanCell: Idempotency holds ✔ ")

    # Test SequentialFilterBlock
    model = ResidualFilter.from_config(
        input_size=n,
        hidden_size=m,
        autoregressive=True,
        activation="ReLU",
    )
    result = model(y, x)
    assert not torch.isnan(result).any(), "Output contains NANs! ❌ "
    logger.info("No NaN outputs ✔ ")

    # verify IDP condition
    y[~mask] = x[~mask]
    assert torch.allclose(x, model(y, x)), "Idempotency failed! ❌ "
    logger.info("Idempotency holds ✔ ")
