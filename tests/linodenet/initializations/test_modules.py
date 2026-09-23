r"""Tests for module-based initializations."""

import torch

from linodenet.initializations import INITIALIZATIONS, Fixed


class TestFixed:
    r"""Validate constant initialization sampling semantics."""

    def test_exported(self) -> None:
        r"""The class is exported through the initialization registry."""
        assert INITIALIZATIONS["Fixed"] is Fixed

    def test_size_empty_tuple_preserves_shape(self) -> None:
        r"""Sampling without batch shape returns the stored tensor shape."""
        value = torch.arange(6, dtype=torch.float32).reshape(2, 3)
        initialization = Fixed(value)

        result = initialization(())

        assert result.shape == value.shape
        assert torch.equal(result, value)

    def test_sampling_duplicates_stored_tensor(self) -> None:
        r"""Sampling with batch shape duplicates the stored tensor."""
        value = torch.arange(6, dtype=torch.float32).reshape(2, 3)
        initialization = Fixed(value)

        result = initialization((4,))

        assert result.shape == (4, 2, 3)
        expected = value.expand(4, 2, 3)
        assert torch.equal(result, expected)
