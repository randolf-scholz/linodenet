import pytest
from torch import tensor

from linodenet.domains import (
    Interval,
    MatrixDomains,
    ScalarDomains,
    UnionOfIntervals,
    VectorDomains,
)


def test_scalar_domains_reflexive_order() -> None:
    assert ScalarDomains.REAL_LINE <= ScalarDomains.REAL_LINE


def test_scalar_domains_transitive_order() -> None:
    assert ScalarDomains.OPEN_UNIT_INTERVAL <= ScalarDomains.UNIT_INTERVAL
    assert ScalarDomains.OPEN_UNIT_INTERVAL != ScalarDomains.UNIT_INTERVAL
    assert ScalarDomains.OPEN_UNIT_INTERVAL <= ScalarDomains.NONNEGATIVE_REALS
    assert ScalarDomains.OPEN_UNIT_INTERVAL != ScalarDomains.NONNEGATIVE_REALS
    assert ScalarDomains.OPEN_UNIT_INTERVAL <= ScalarDomains.REAL_LINE
    assert ScalarDomains.OPEN_UNIT_INTERVAL != ScalarDomains.REAL_LINE
    assert ScalarDomains.OPEN_UNIT_INTERVAL <= ScalarDomains.EXTENDED_LINE
    assert ScalarDomains.OPEN_UNIT_INTERVAL != ScalarDomains.EXTENDED_LINE
    assert ScalarDomains.POSITIVE_REALS <= ScalarDomains.NONZERO
    assert ScalarDomains.POSITIVE_REALS != ScalarDomains.NONZERO
    assert ScalarDomains.NONZERO <= ScalarDomains.EXTENDED_LINE
    assert ScalarDomains.NONZERO != ScalarDomains.EXTENDED_LINE


def test_scalar_domains_incomparable_elements() -> None:
    assert not ScalarDomains.NONNEGATIVE_REALS <= ScalarDomains.NONPOSITIVE_REALS
    assert not ScalarDomains.NONPOSITIVE_REALS <= ScalarDomains.NONNEGATIVE_REALS
    assert not ScalarDomains.UNIT_INTERVAL <= ScalarDomains.NONPOSITIVE_REALS
    assert not ScalarDomains.NEGATIVE_REALS <= ScalarDomains.NONNEGATIVE_REALS
    assert not ScalarDomains.NONZERO <= ScalarDomains.NONNEGATIVE_REALS
    assert not ScalarDomains.NONNEGATIVE_REALS <= ScalarDomains.NONZERO


def test_scalar_domains_string_representation() -> None:
    assert str(ScalarDomains.OPEN_UNIT_INTERVAL) == "(0, 1)"


def test_scalar_domains_interval_membership() -> None:
    values = tensor([-0.5, 0.0, 0.5, 1.0, 1.5])
    mask = ScalarDomains.UNIT_INTERVAL.__contains__(values)
    assert mask.tolist() == [False, True, True, True, False]


def test_scalar_domains_nonzero_membership() -> None:
    values = tensor([-1.0, 0.0, 1.0])
    mask = ScalarDomains.NONZERO.__contains__(values)
    assert mask.tolist() == [True, False, True]


def test_scalar_domains_store_interval_values() -> None:
    assert isinstance(ScalarDomains.UNIT_INTERVAL.value, Interval)
    assert isinstance(ScalarDomains.NONZERO.value, UnionOfIntervals)


def test_scalar_domains_reject_cross_type_ordering() -> None:
    with pytest.raises(TypeError):
        _ = ScalarDomains.REAL_LINE <= "(-inf, inf)"


def test_union_of_intervals_merges_overlaps_and_touching_intervals() -> None:
    domain = UnionOfIntervals(
        Interval.from_string("[0, 1]"),
        Interval.from_string("(1, 2)"),
        Interval.from_string("[3, 4]"),
        Interval.from_string("[3.5, 5]"),
    )
    assert str(domain) == "[0, 2) | [3, 5]"


def test_union_of_intervals_from_string() -> None:
    domain = UnionOfIntervals.from_string("(-inf, 0) | (0, inf)")
    values = tensor([-1.0, 0.0, 1.0])
    assert str(domain) == "(-inf, 0) | (0, inf)"
    assert domain.__contains__(values).tolist() == [True, False, True]


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
