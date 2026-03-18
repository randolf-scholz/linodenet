import pytest

from linodenet.domains import MatrixDomains


def test_matrix_domains_reflexive_order() -> None:
    assert MatrixDomains.SQUARE <= MatrixDomains.SQUARE


def test_matrix_domains_transitive_order() -> None:
    assert MatrixDomains.DIAGONAL <= MatrixDomains.SYMMETRIC
    assert MatrixDomains.DIAGONAL != MatrixDomains.SYMMETRIC
    assert MatrixDomains.DIAGONAL <= MatrixDomains.SQUARE
    assert MatrixDomains.DIAGONAL != MatrixDomains.SQUARE
    assert MatrixDomains.DIAGONAL <= MatrixDomains.RECTANGULAR
    assert MatrixDomains.DIAGONAL != MatrixDomains.RECTANGULAR
    assert MatrixDomains.RANK_ONE <= MatrixDomains.LOW_RANK
    assert MatrixDomains.RANK_ONE != MatrixDomains.LOW_RANK
    assert MatrixDomains.RANK_ONE <= MatrixDomains.RECTANGULAR
    assert MatrixDomains.RANK_ONE != MatrixDomains.RECTANGULAR


def test_matrix_domains_multiple_inheritance_paths() -> None:
    assert MatrixDomains.SPECIAL_ORTHOGONAL <= MatrixDomains.ORTHOGONAL
    assert MatrixDomains.SPECIAL_ORTHOGONAL != MatrixDomains.ORTHOGONAL
    assert MatrixDomains.SPECIAL_ORTHOGONAL <= MatrixDomains.INVERTIBLE
    assert MatrixDomains.SPECIAL_ORTHOGONAL != MatrixDomains.INVERTIBLE
    assert MatrixDomains.PERMUTATION <= MatrixDomains.ROW_STOCHASTIC
    assert MatrixDomains.PERMUTATION != MatrixDomains.ROW_STOCHASTIC
    assert MatrixDomains.PERMUTATION <= MatrixDomains.SQUARE
    assert MatrixDomains.PERMUTATION != MatrixDomains.SQUARE


def test_matrix_domains_incomparable_elements() -> None:
    assert not MatrixDomains.SYMMETRIC <= MatrixDomains.ORTHOGONAL
    assert not MatrixDomains.ORTHOGONAL <= MatrixDomains.SYMMETRIC
    assert not MatrixDomains.LOW_RANK <= MatrixDomains.BANDED
    assert not MatrixDomains.BANDED <= MatrixDomains.LOW_RANK


def test_matrix_domains_string_representation() -> None:
    assert str(MatrixDomains.SQUARE) == "square"


def test_matrix_domains_reject_cross_type_ordering() -> None:
    with pytest.raises(TypeError):
        _ = MatrixDomains.SQUARE <= "square"
