r"""Test if filters satisfy idempotence property."""

import logging

import torch

from linodenet.models.filters import KalmanBlockCell, KalmanCell

__logger__ = logging.getLogger(__name__)
NAN = torch.tensor(float("nan"))


def test_idempotency():
    r"""Check whether idempotency holds."""
    batch_dim, m, n = (3, 4, 5), 100, 100
    x = torch.randn(*batch_dim, n)
    y = torch.randn(*batch_dim, m)
    mask = y > 0
    y[mask] = NAN

    # Test KalmanCel
    model = KalmanCell(
        input_size=n, hidden_size=m, autoregressive=True, activation="ReLU"
    )
    result = model(y, x)
    assert not torch.isnan(result).any(), "Output contains NANs!"
    __logger__.info("KalmanCell: No NaN outputs ✔ ")

    # verify IDP condition
    y[~mask] = x[~mask]
    assert (x == model(y, x)).all(), "Idempotency failed!"
    __logger__.info("KalmanCell: Idempotency holds ✔ ")

    # Test KalmanBlockCell
    model = KalmanBlockCell(
        input_size=n, hidden_size=m, autoregressive=True, activation="ReLU"
    )
    result = model(y, x)
    assert not torch.isnan(result).any(), "Output contains NANs!"
    __logger__.info("KalmanBlockCell: No NaN outputs ✔ ")

    # verify IDP condition
    y[~mask] = x[~mask]
    assert (x == model(y, x)).all(), "Idempotency failed!"
    __logger__.info("KalmanBlockCell: Idempotency holds ✔ ")


def __main__():
    logging.basicConfig(level=logging.INFO)
    __logger__.info("Testing Filters started!")
    test_idempotency()
    __logger__.info("Testing Filters finished!")


if __name__ == "__main__":
    __main__()
