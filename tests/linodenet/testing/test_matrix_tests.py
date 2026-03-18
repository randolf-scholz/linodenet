r"""Tests for `linodenet.testing.matrix_tests`."""

import torch

from linodenet.testing.matrix_tests import (
    is_banded,
    is_diagonal,
    is_low_rank,
    is_square,
    is_symmetric,
    is_upper_triangular,
)


def test_square_matrix_tests_return_false_for_size_mismatch() -> None:
    x = torch.eye(3).expand(2, -1, -1)

    result = is_symmetric(x, size=4)

    assert result.shape == (2,)
    assert not result.any()


def test_rectangular_matrix_tests_return_false_for_shape_mismatch() -> None:
    x = torch.randn(2, 3, 4)

    result = is_low_rank(x, 1, shape=(4, 3))

    assert result.shape == (2,)
    assert not result.any()


def test_non_batched_shape_checks_return_scalar_false() -> None:
    x = torch.eye(3)

    assert not is_square(x, shape=(4, 4)).item()
    assert not is_banded(x, -1, 1, shape=(4, 4)).item()


def test_matrix_tests_support_non_default_matrix_dims() -> None:
    diagonal = torch.eye(3).expand(2, -1, -1)
    diagonal = diagonal.movedim((-2, -1), (0, 2))

    assert is_diagonal(diagonal, size=3, dim=(0, 2)).all()
    assert is_upper_triangular(diagonal, shape=(3, 3), dim=(0, 2)).all()

    symmetric = torch.eye(4).expand(2, -1, -1).movedim((-2, -1), (0, 2))
    assert is_symmetric(symmetric, size=4, dim=(0, 2)).all()
