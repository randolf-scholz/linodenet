r"""Test if filters satisfy idempotence property."""

import logging

import pytest
import torch

from linodenet.config import PROJECT
from linodenet.constants import NAN
from linodenet.models.filters import ResidualFilterBlock

RESULT_DIR = PROJECT.RESULTS_DIR[__file__]
__logger__ = logging.getLogger(__name__)


@pytest.mark.flaky(reruns=3)
def test_filter_idempotency() -> None:
    r"""Check whether idempotency holds."""
    LOGGER = __logger__.getChild(__name__)
    LOGGER.info("Testing idempotency.")
    batch_dim, m, n = (3, 4, 5), 100, 100
    x = torch.randn(*batch_dim, n)
    y = torch.randn(*batch_dim, m)
    mask = y > 0
    y[mask] = NAN

    # # Test KalmanCel
    # model = KalmanCell(
    #     input_size=n, hidden_size=m, autoregressive=True, activation="ReLU"
    # )
    # result = model(y, x)
    # assert not torch.isnan(result).any(), "Output contains NANs! ❌ "
    # LOGGER.info("KalmanCell: No NaN outputs ✔ ")
    #
    # # verify IDP condition
    # y[~mask] = x[~mask]
    # assert torch.allclose(x, model(y, x)), "Idempotency failed! ❌ "
    # LOGGER.info("KalmanCell: Idempotency holds ✔ ")

    # Test SequentialFilterBlock
    model = ResidualFilterBlock(
        input_size=n, hidden_size=m, autoregressive=True, activation="ReLU"
    )
    result = model(y, x)
    assert not torch.isnan(result).any(), "Output contains NANs! ❌ "
    LOGGER.info("No NaN outputs ✔ ")

    # verify IDP condition
    y[~mask] = x[~mask]
    assert torch.allclose(x, model(y, x)), "Idempotency failed! ❌ "
    LOGGER.info("Idempotency holds ✔ ")
