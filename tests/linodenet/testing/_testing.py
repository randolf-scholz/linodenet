#!/usr/bin/env python
"""Tests for linodenet.testing._testing."""

import torch

from linodenet.testing import test_model


def test_test_model() -> None:
    """Test test_model."""
    model = torch.nn.Linear(4, 4)
    x = torch.randn(3, 4)
    test_model(model, inputs=x, test_jit=True)


if __name__ == "__main__":
    pass
