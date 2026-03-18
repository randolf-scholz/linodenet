import pytest

from linodenet.domains import MatrixDomains, VectorDomains


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


def test_vector_domains_reflexive_order() -> None:
    assert VectorDomains.REAL <= VectorDomains.REAL


def test_vector_domains_transitive_order() -> None:
    assert VectorDomains.ONE_HOT <= VectorDomains.STOCHASTIC
    assert VectorDomains.ONE_HOT != VectorDomains.STOCHASTIC
    assert VectorDomains.ONE_HOT <= VectorDomains.NONNEGATIVE
    assert VectorDomains.ONE_HOT != VectorDomains.NONNEGATIVE
    assert VectorDomains.ONE_HOT <= VectorDomains.REAL
    assert VectorDomains.ONE_HOT != VectorDomains.REAL
    assert VectorDomains.ONE_HOT <= VectorDomains.COMPLEX
    assert VectorDomains.ONE_HOT != VectorDomains.COMPLEX


def test_vector_domains_multiple_inheritance_paths() -> None:
    assert VectorDomains.STANDARDIZED <= VectorDomains.ZERO_MEAN
    assert VectorDomains.STANDARDIZED != VectorDomains.ZERO_MEAN
    assert VectorDomains.STANDARDIZED <= VectorDomains.NONZERO
    assert VectorDomains.STANDARDIZED != VectorDomains.NONZERO


def test_vector_domains_incomparable_elements() -> None:
    assert not VectorDomains.NONNEGATIVE <= VectorDomains.NONPOSITIVE
    assert not VectorDomains.NONPOSITIVE <= VectorDomains.NONNEGATIVE
    assert not VectorDomains.STOCHASTIC <= VectorDomains.UNIT_VECTOR
    assert not VectorDomains.UNIT_VECTOR <= VectorDomains.STOCHASTIC


def test_vector_domains_string_representation() -> None:
    assert str(VectorDomains.STOCHASTIC) == "stochastic"


def test_vector_domains_reject_cross_type_ordering() -> None:
    with pytest.raises(TypeError):
        _ = VectorDomains.REAL <= "real"
