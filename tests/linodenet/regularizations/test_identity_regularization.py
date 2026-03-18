r"""Tests for the identity regularization."""

import torch

from linodenet.regularizations import functional, modules


def test_identity_regularization_biases_towards_identity() -> None:
    x = torch.eye(4)
    y = torch.zeros(4, 4)

    assert torch.allclose(functional.identity(x), torch.tensor(0.0))
    assert functional.identity(y) > 0

    regularization = modules.Identity()
    assert torch.allclose(regularization(x), torch.tensor(0.0))
    assert regularization(y) > 0
