r"""Tests for linodenet.testing._testing."""

import torch

from linodenet.testing import check_model


def test_test_model() -> None:
    r"""Test test_model."""
    model = torch.nn.Linear(4, 4)
    x = torch.randn(3, 4)
    check_model(model, input_args=(x,), test_jit=True)
