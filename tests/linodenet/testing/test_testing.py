r"""Tests for linodenet.testing._testing."""

import torch

from linodenet.testing import assert_model_ok


def test_test_model() -> None:
    r"""Test test_model."""
    model = torch.nn.Linear(4, 4)
    x = torch.randn(3, 4)
    assert_model_ok(model, call_args=(x,), test_jit=True)
