#!/usr/bin/env python
r"""Test if filters satisfy idempotence property."""

import logging
from pathlib import Path

import pytest
import torch

from linodenet.config import PROJECT
from linodenet.models.filters import SequentialFilterBlock

logging.basicConfig(level=logging.INFO)
__logger__ = logging.getLogger(__name__)
LOGGER = __logger__.getChild(SequentialFilterBlock.__name__)
RESULT_DIR = PROJECT.TESTS_PATH / "results" / Path(__file__).stem
RESULT_DIR.mkdir(parents=True, exist_ok=True)
NAN = torch.tensor(float("nan"))


# @pytest.mark.parametrize("key", FunctionalInitializations)
@pytest.mark.flaky(reruns=3)
def test_filter_idempotency() -> None:
    r"""Check whether idempotency holds."""
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
    model = SequentialFilterBlock(
        input_size=n, hidden_size=m, autoregressive=True, activation="ReLU"
    )
    result = model(y, x)
    assert not torch.isnan(result).any(), "Output contains NANs! ❌ "
    LOGGER.info("No NaN outputs ✔ ")

    # verify IDP condition
    y[~mask] = x[~mask]
    assert torch.allclose(x, model(y, x)), "Idempotency failed! ❌ "
    LOGGER.info("Idempotency holds ✔ ")


def _main() -> None:
    test_filter_idempotency()


if __name__ == "__main__":
    _main()
