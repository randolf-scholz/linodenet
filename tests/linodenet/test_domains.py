import pytest
from torch import tensor

from linodenet.domains import (
    Interval,
    IntervalUnion,
    MatrixDomains,
    ScalarDomains,
    VectorDomains,
)


class TestScalarDomains:
    def test_partial_order_and_representation(self) -> None:
        assert ScalarDomains.REAL_LINE <= ScalarDomains.REAL_LINE
        assert ScalarDomains.REAL_LINE < ScalarDomains.EXTENDED_LINE
        assert ScalarDomains.REAL_LINE <= Interval("(-inf, inf)")
        assert ScalarDomains.OPEN_UNIT_INTERVAL < Interval("[0, 1]")
        assert ScalarDomains.NONZERO <= IntervalUnion("(-inf, 0) | (0, inf)")
        assert ScalarDomains.POSITIVE_REALS < IntervalUnion("(-inf, 0) | (0, inf)")

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

        assert not ScalarDomains.NONNEGATIVE_REALS <= ScalarDomains.NONPOSITIVE_REALS
        assert not ScalarDomains.NONPOSITIVE_REALS <= ScalarDomains.NONNEGATIVE_REALS
        assert not ScalarDomains.UNIT_INTERVAL <= ScalarDomains.NONPOSITIVE_REALS
        assert not ScalarDomains.NEGATIVE_REALS <= ScalarDomains.NONNEGATIVE_REALS
        assert not ScalarDomains.NONZERO <= ScalarDomains.NONNEGATIVE_REALS
        assert not ScalarDomains.NONNEGATIVE_REALS <= ScalarDomains.NONZERO
        assert not ScalarDomains.EXTENDED_LINE <= Interval("(-inf, inf)")
        assert not ScalarDomains.NONNEGATIVE_REALS <= Interval("(0, inf)")

        assert str(ScalarDomains.OPEN_UNIT_INTERVAL) == "(0, 1)"
        assert str(ScalarDomains.UNIT_INTERVAL.value) == "[0, 1]"
        assert isinstance(ScalarDomains.NONZERO.value, IntervalUnion)

        with pytest.raises(TypeError):
            _ = ScalarDomains.REAL_LINE <= "(-inf, inf)"

    def test_membership(self) -> None:
        values = tensor([-0.5, 0.0, 0.5, 1.0, 1.5])
        assert ScalarDomains.UNIT_INTERVAL.__contains__(values).tolist() == [
            False,
            True,
            True,
            True,
            False,
        ]

        values = tensor([-1.0, 0.0, 1.0])
        assert ScalarDomains.NONZERO.__contains__(values).tolist() == [
            True,
            False,
            True,
        ]

    def test_interval_union_normalization_and_membership(self) -> None:
        domain = IntervalUnion.from_string("[0, 1] | (1, 2) | [3, 4] | [3.5, 5]")
        assert str(domain) == "[0, 2) | [3, 5]"

        domain = IntervalUnion.from_string("(-inf, 0) | (0, inf)")
        values = tensor([-1.0, 0.0, 1.0])
        assert str(domain) == "(-inf, 0) | (0, inf)"
        assert domain.__contains__(values).tolist() == [True, False, True]

    def test_interval_infinity_edges(self) -> None:
        assert Interval("[-inf, inf]") <= Interval("[-inf, inf]")
        assert Interval("(-inf, inf)") <= Interval("[-inf, inf]")
        assert Interval("(-inf, inf)") < Interval("[-inf, inf]")
        assert not Interval("[-inf, inf]") <= Interval("(-inf, inf)")

        assert Interval("[-inf, 0)") <= Interval("[-inf, 0]")
        assert Interval("[-inf, 0)") < Interval("[-inf, 0]")
        assert Interval("(0, inf]") <= Interval("[0, inf]")
        assert not Interval("[-inf, 0]") <= Interval("(-inf, 0]")
        assert not Interval("[0, inf]") <= Interval("(0, inf)")

    def test_interval_union_strict_subset(self) -> None:
        assert IntervalUnion("[0, 1]") < IntervalUnion("[0, 2]")
        assert IntervalUnion("(-inf, 0) | (0, inf)") < IntervalUnion("[-inf, inf]")
        assert not IntervalUnion("[-inf, inf]") < IntervalUnion("[-inf, inf]")


class TestVectorDomains:
    def test_partial_order_and_representation(self) -> None:
        assert VectorDomains.REAL <= VectorDomains.REAL
        assert VectorDomains.ONE_HOT < VectorDomains.STOCHASTIC
        assert not VectorDomains.STOCHASTIC < VectorDomains.STOCHASTIC

        assert VectorDomains.ONE_HOT <= VectorDomains.STOCHASTIC
        assert VectorDomains.ONE_HOT != VectorDomains.STOCHASTIC
        assert VectorDomains.ONE_HOT <= VectorDomains.NONNEGATIVE
        assert VectorDomains.ONE_HOT != VectorDomains.NONNEGATIVE
        assert VectorDomains.ONE_HOT <= VectorDomains.REAL
        assert VectorDomains.ONE_HOT != VectorDomains.REAL
        assert VectorDomains.ONE_HOT <= VectorDomains.COMPLEX
        assert VectorDomains.ONE_HOT != VectorDomains.COMPLEX

        assert VectorDomains.STANDARDIZED <= VectorDomains.ZERO_MEAN
        assert VectorDomains.STANDARDIZED != VectorDomains.ZERO_MEAN
        assert VectorDomains.STANDARDIZED <= VectorDomains.NONZERO
        assert VectorDomains.STANDARDIZED != VectorDomains.NONZERO

        assert not VectorDomains.NONNEGATIVE <= VectorDomains.NONPOSITIVE
        assert not VectorDomains.NONPOSITIVE <= VectorDomains.NONNEGATIVE
        assert not VectorDomains.STOCHASTIC <= VectorDomains.UNIT_VECTOR
        assert not VectorDomains.UNIT_VECTOR <= VectorDomains.STOCHASTIC

        assert str(VectorDomains.STOCHASTIC) == "stochastic"

        with pytest.raises(TypeError):
            _ = VectorDomains.REAL <= "real"


class TestMatrixDomains:
    def test_partial_order_and_representation(self) -> None:
        assert MatrixDomains.SQUARE <= MatrixDomains.SQUARE

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

        assert MatrixDomains.SPECIAL_ORTHOGONAL <= MatrixDomains.ORTHOGONAL
        assert MatrixDomains.SPECIAL_ORTHOGONAL != MatrixDomains.ORTHOGONAL
        assert MatrixDomains.SPECIAL_ORTHOGONAL <= MatrixDomains.INVERTIBLE
        assert MatrixDomains.SPECIAL_ORTHOGONAL != MatrixDomains.INVERTIBLE
        assert MatrixDomains.PERMUTATION <= MatrixDomains.ROW_STOCHASTIC
        assert MatrixDomains.PERMUTATION != MatrixDomains.ROW_STOCHASTIC
        assert MatrixDomains.PERMUTATION <= MatrixDomains.SQUARE
        assert MatrixDomains.PERMUTATION != MatrixDomains.SQUARE

        assert not MatrixDomains.SYMMETRIC <= MatrixDomains.ORTHOGONAL
        assert not MatrixDomains.ORTHOGONAL <= MatrixDomains.SYMMETRIC
        assert not MatrixDomains.LOW_RANK <= MatrixDomains.BANDED
        assert not MatrixDomains.BANDED <= MatrixDomains.LOW_RANK

        assert str(MatrixDomains.SQUARE) == "square"

        with pytest.raises(TypeError):
            _ = MatrixDomains.SQUARE <= "square"
