r"""Tests for `linodenet.testing.matrix_tests`."""

import torch

from linodenet.domains.matrix_tests import (
    is_banded,
    is_column_stochastic,
    is_diagonal,
    is_doubly_stochastic,
    is_low_rank,
    is_low_rank_skew_symmetric,
    is_low_rank_square,
    is_low_rank_symmetric,
    is_negative_definite,
    is_negative_semidefinite,
    is_positive_definite,
    is_positive_semidefinite,
    is_row_stochastic,
    is_skew_symmetric,
    is_square,
    is_symmetric,
    is_tall,
    is_triangular,
    is_upper_triangular,
    is_wide,
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
    assert not is_tall(x, shape=(4, 4)).item()
    assert not is_wide(x, shape=(4, 4)).item()
    assert not is_banded(x, -1, 1, shape=(4, 4)).item()


def test_matrix_tests_support_non_default_matrix_dims() -> None:
    diagonal = torch.eye(3).expand(2, -1, -1)
    diagonal = diagonal.movedim((-2, -1), (0, 2))

    assert is_diagonal(diagonal, size=3, dim=(0, 2)).all()
    assert is_upper_triangular(diagonal, shape=(3, 3), dim=(0, 2)).all()

    symmetric = torch.eye(4).expand(2, -1, -1).movedim((-2, -1), (0, 2))
    assert is_symmetric(symmetric, size=4, dim=(0, 2)).all()

    skew_symmetric = torch.tensor(
        [
            [[0.0, 1.0], [-1.0, 0.0]],
            [[0.0, 2.0], [-2.0, 0.0]],
        ]
    ).movedim((-2, -1), (0, 2))
    assert is_skew_symmetric(skew_symmetric, size=2, dim=(0, 2)).all()

    tall = torch.randn(2, 4, 3).movedim((-2, -1), (0, 2))
    wide = torch.randn(2, 3, 4).movedim((-2, -1), (0, 2))
    assert is_tall(tall, shape=(4, 3), dim=(0, 2)).all()
    assert is_wide(wide, shape=(3, 4), dim=(0, 2)).all()


def test_tall_and_wide_matrix_tests() -> None:
    tall = torch.randn(2, 4, 3)
    wide = torch.randn(2, 3, 4)
    square = torch.randn(2, 3, 3)

    assert is_tall(tall).all()
    assert not is_wide(tall).any()

    assert is_wide(wide).all()
    assert not is_tall(wide).any()

    assert is_tall(square).all()
    assert is_wide(square).all()


def test_triangular_matrix_test() -> None:
    lower = torch.tensor([[[1.0, 0.0], [2.0, 3.0]]])
    upper = torch.tensor([[[1.0, 2.0], [0.0, 3.0]]])
    non_triangular = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])

    assert is_triangular(lower).all()
    assert is_triangular(upper).all()
    assert not is_triangular(non_triangular).any()


def test_stochastic_matrix_tests() -> None:
    row_stochastic = torch.tensor([[[0.5, 0.5], [0.25, 0.75]]])
    column_stochastic = torch.tensor([[[0.5, 0.25], [0.5, 0.75]]])
    doubly_stochastic = torch.tensor([[[0.5, 0.5], [0.5, 0.5]]])
    non_stochastic = torch.tensor([[[1.2, -0.2], [0.0, 1.0]]])

    assert is_row_stochastic(row_stochastic).all()
    assert not is_column_stochastic(row_stochastic).all()

    assert is_column_stochastic(column_stochastic).all()
    assert not is_row_stochastic(column_stochastic).all()

    assert is_doubly_stochastic(doubly_stochastic).all()
    assert not is_row_stochastic(non_stochastic).any()
    assert not is_column_stochastic(non_stochastic).any()


def test_low_rank_symmetric_matrix_test() -> None:
    matrices = torch.tensor([
        [[2.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 0.0]],
        [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]],
        [[0.0, 1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
    ])  # fmt: skip

    assert torch.equal(
        is_low_rank_symmetric(matrices, 1),
        torch.tensor([True, False, False]),
    )


def test_low_rank_square_matrix_test() -> None:
    batched = torch.tensor([
        [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 0.0]],
        [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]],
    ])  # fmt: skip
    rectangular = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]]])

    assert torch.equal(is_low_rank_square(batched, 2), torch.tensor([True, False]))
    assert not is_low_rank_square(rectangular, 2).item()


def test_low_rank_skew_symmetric_matrix_test() -> None:
    low_rank = torch.tensor([[[0.0, 2.0, 0.0], [-2.0, 0.0, 0.0], [0.0, 0.0, 0.0]]])
    high_rank = torch.tensor(
        [
            [
                [0.0, 1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 3.0],
                [0.0, 0.0, -3.0, 0.0],
            ]
        ]
    )
    non_skew = torch.tensor([[[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]])

    assert torch.equal(is_low_rank_skew_symmetric(low_rank, 1), torch.tensor([True]))
    assert torch.equal(is_low_rank_skew_symmetric(high_rank, 1), torch.tensor([False]))
    assert torch.equal(is_low_rank_skew_symmetric(non_skew, 1), torch.tensor([False]))


class TestDefiniteness:
    def test_assumes_symmetry(self) -> None:
        positive = torch.tensor([[1.0, 2.0], [0.0, 1.0]])
        negative = -positive

        assert not is_positive_definite(positive).item()
        assert not is_positive_semidefinite(positive).item()
        assert not is_negative_definite(negative).item()
        assert not is_negative_semidefinite(negative).item()

    def test_definite(self) -> None:
        matrices = torch.tensor([
            [[2.0, 0.0], [0.0, 1.0]],
            [[1.0, 0.0], [0.0, 0.0]],
            [[1.0, 0.0], [0.0, -1.0]],
        ])  # fmt: skip

        assert torch.equal(
            is_positive_definite(matrices),
            torch.tensor([True, False, False]),
        )
        assert torch.equal(
            is_negative_definite(-matrices),
            torch.tensor([True, False, False]),
        )

    def test_semidefinite(self) -> None:
        matrices = torch.tensor([
            [[2.0, 0.0], [0.0, 1.0]],
            [[1.0, 0.0], [0.0, 0.0]],
            [[1.0, 0.0], [0.0, -1.0]],
        ])  # fmt: skip
        assert torch.equal(
            is_positive_semidefinite(matrices),
            torch.tensor([True, True, False]),
        )
        assert torch.equal(
            is_negative_semidefinite(-matrices),
            torch.tensor([True, True, False]),
        )
