import pytest
from torch import tensor

from linodenet.domains import (
    Interval,
    MatrixDomains as M,
    RealDomain,
    ScalarDomains,
    TensorDomains,
    VectorDomains,
)
from linodenet.domains.matrix_domains import ColumnOrthogonal, RowOrthogonal, Tall, Wide
from linodenet.testing import is_left_invertible, is_right_invertible


class TestScalarDomains:
    def test_partial_order_and_representation(self) -> None:
        assert ScalarDomains.REAL_LINE <= ScalarDomains.REAL_LINE
        assert ScalarDomains.REAL_LINE < ScalarDomains.EXTENDED_LINE
        assert ScalarDomains.REAL_LINE <= Interval("(-inf, inf)")
        assert ScalarDomains.OPEN_UNIT_INTERVAL < Interval("[0, 1]")
        assert ScalarDomains.NONZERO <= RealDomain("(-inf, 0) | (0, inf)")
        assert ScalarDomains.POSITIVE_REALS < RealDomain("(-inf, 0) | (0, inf)")

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
        assert isinstance(ScalarDomains.NONZERO.value, RealDomain)

        with pytest.raises(TypeError):
            _ = ScalarDomains.REAL_LINE <= "(-inf, inf)"

    def test_membership(self) -> None:
        values = tensor([-0.5, 0.0, 0.5, 1.0, 1.5])
        result = [v in ScalarDomains.UNIT_INTERVAL for v in values]
        expected = [False, True, True, True, False]
        assert result == expected

        values = tensor([-1.0, -0.0, 0.0, 1.0])
        result = [v in ScalarDomains.NONZERO for v in values]
        expected = [True, False, False, True]
        assert result == expected

    def test_interval_union_normalization_and_membership(self) -> None:
        domain = RealDomain.from_string("[0, 1] | (1, 2) | [3, 4] | [3.5, 5]")
        assert str(domain) == "[0, 2) | [3, 5]"

        domain = RealDomain.from_string("(-inf, 0) | (0, inf)")
        assert str(domain) == "(-inf, 0) | (0, inf)"

        values = tensor([-1.0, 0.0, 1.0])
        result = [v in domain for v in values]
        expected = [True, False, True]
        assert result == expected

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
        assert RealDomain("[0, 1]") < RealDomain("[0, 2]")
        assert RealDomain("(-inf, 0) | (0, inf)") < RealDomain("[-inf, inf]")
        assert not RealDomain("[-inf, inf]") < RealDomain("[-inf, inf]")


class TestVectorDomains:
    def test_none_factorizations(self) -> None:
        assert VectorDomains.NONE.factorizations == frozenset(
            {
                VectorDomains.NEGATIVE & VectorDomains.NONNEGATIVE,
                VectorDomains.NONZERO & VectorDomains.ZERO,
                VectorDomains.POSITIVE & VectorDomains.NONPOSITIVE,
                VectorDomains.POSITIVE & VectorDomains.NEGATIVE,
            }
        )

    def test_factorizations(self) -> None:
        assert VectorDomains.ONE_HOT.factorizations == frozenset(
            {
                VectorDomains.BOOLEAN & VectorDomains.STOCHASTIC,
                VectorDomains.STOCHASTIC & VectorDomains.UNIT_VECTOR,
            }
        )
        assert VectorDomains.ZERO.factorizations == frozenset(
            {
                VectorDomains.NONNEGATIVE & VectorDomains.NONPOSITIVE,
            }
        )

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


class TestTensorDomains:
    def test_partial_order_and_representation(self) -> None:
        assert TensorDomains.NONE < TensorDomains.ANY
        assert TensorDomains.NONE <= TensorDomains.ZERO
        assert TensorDomains.NONE <= TensorDomains.COMPLEX
        assert TensorDomains.ANY <= TensorDomains.ANY

        assert TensorDomains.BOOLEAN <= TensorDomains.REAL
        assert TensorDomains.BOOLEAN != TensorDomains.REAL
        assert TensorDomains.BOOLEAN <= TensorDomains.COMPLEX
        assert TensorDomains.BOOLEAN != TensorDomains.COMPLEX
        assert TensorDomains.BOOLEAN <= TensorDomains.ANY
        assert TensorDomains.BOOLEAN != TensorDomains.ANY

        assert TensorDomains.ZERO <= TensorDomains.SPARSE
        assert TensorDomains.ZERO != TensorDomains.SPARSE
        assert TensorDomains.ZERO <= TensorDomains.BOOLEAN
        assert TensorDomains.ZERO != TensorDomains.BOOLEAN
        assert TensorDomains.ONE <= TensorDomains.NONZERO
        assert TensorDomains.ONE != TensorDomains.NONZERO

        assert not TensorDomains.SPARSE <= TensorDomains.COMPLEX
        assert not TensorDomains.NONZERO <= TensorDomains.SPARSE

        assert str(TensorDomains.NONZERO) == "nonzero"

        with pytest.raises(TypeError):
            _ = TensorDomains.ANY <= "any"


class TestMatrixDomains:
    def test_poset_meet_expression(self) -> None:
        meet = M.TALL & M.WIDE & M.SQUARE
        assert len(meet) == 3
        assert set(meet) == {
            M.TALL,
            M.WIDE,
            M.SQUARE,
        }
        assert M.SQUARE.factorizations == frozenset({M.TALL & M.WIDE})
        assert M.ROW_STOCHASTIC & M.COLUMN_STOCHASTIC <= M.SQUARE
        assert M.DOUBLY_STOCHASTIC <= M.SQUARE

    def test_partial_order_and_representation(self) -> None:
        assert M.SQUARE <= M.SQUARE
        assert M.SQUARE <= M.TALL
        assert M.SQUARE <= M.WIDE
        assert M.ORTHOGONAL <= M.COLUMN_ORTHOGONAL
        assert M.ORTHOGONAL <= M.ROW_ORTHOGONAL
        assert M.ORTHOGONAL <= M.SQUARE
        assert M.COLUMN_ORTHOGONAL <= M.TALL
        assert M.COLUMN_ORTHOGONAL <= M.LEFT_INVERTIBLE
        assert M.ROW_ORTHOGONAL <= M.RIGHT_INVERTIBLE
        assert M.INVERTIBLE <= M.LEFT_INVERTIBLE
        assert M.INVERTIBLE <= M.RIGHT_INVERTIBLE
        assert M.INVERTIBLE <= M.SQUARE
        assert M.LOWER_INVERTIBLE <= M.LOWER_TRIANGULAR
        assert M.LOWER_INVERTIBLE <= M.INVERTIBLE
        assert M.UPPER_INVERTIBLE <= M.UPPER_TRIANGULAR
        assert M.UPPER_INVERTIBLE <= M.INVERTIBLE
        assert M.CHOLESKY_FACTOR <= M.LOWER_INVERTIBLE
        assert M.CHOLESKY_FACTOR <= M.POSITIVE_DIAGONAL_ENTRIES
        assert M.LEFT_INVERTIBLE <= M.TALL
        assert M.ROW_ORTHOGONAL <= M.WIDE
        assert M.RIGHT_INVERTIBLE <= M.WIDE
        assert M.TALL <= M.RECTANGULAR
        assert M.WIDE <= M.RECTANGULAR
        assert M.TALL != M.RECTANGULAR
        assert M.WIDE != M.RECTANGULAR
        assert M.LEFT_INVERTIBLE != M.TALL
        assert M.RIGHT_INVERTIBLE != M.WIDE
        assert M.COLUMN_ORTHOGONAL != M.TALL
        assert M.ROW_ORTHOGONAL != M.WIDE
        assert M.LOWER_INVERTIBLE != M.LOWER_TRIANGULAR
        assert M.UPPER_INVERTIBLE != M.UPPER_TRIANGULAR
        assert M.CHOLESKY_FACTOR != M.LOWER_INVERTIBLE

        assert M.DIAGONAL <= M.SYMMETRIC
        assert M.DIAGONAL != M.SYMMETRIC
        assert M.DIAGONAL <= M.SQUARE
        assert M.DIAGONAL != M.SQUARE
        assert M.DIAGONAL <= M.RECTANGULAR
        assert M.DIAGONAL != M.RECTANGULAR
        assert M.POSITIVE_DIAGONAL_ENTRIES <= M.RECTANGULAR
        assert M.NEGATIVE_DIAGONAL_ENTRIES <= M.RECTANGULAR
        assert M.ZERO_DIAGONAL <= M.RECTANGULAR
        assert M.SKEW_SYMMETRIC <= M.ZERO_DIAGONAL
        assert not M.ZERO_DIAGONAL <= M.DIAGONAL
        assert M.RANK_ONE <= M.LOW_RANK
        assert M.RANK_ONE != M.LOW_RANK
        assert M.RANK_ONE <= M.RECTANGULAR
        assert M.RANK_ONE != M.RECTANGULAR

        assert M.SPECIAL_ORTHOGONAL <= M.ORTHOGONAL
        assert M.SPECIAL_ORTHOGONAL != M.ORTHOGONAL
        assert M.SPECIAL_ORTHOGONAL <= M.INVERTIBLE
        assert M.SPECIAL_ORTHOGONAL != M.INVERTIBLE
        assert M.IDENTITY <= M.DIAGONAL
        assert M.IDENTITY <= M.PERMUTATION
        assert M.PERMUTATION <= M.ROW_STOCHASTIC
        assert M.PERMUTATION != M.ROW_STOCHASTIC
        assert M.PERMUTATION <= M.DOUBLY_STOCHASTIC
        assert M.PERMUTATION <= M.SPARSE
        assert M.DOUBLY_STOCHASTIC <= M.SQUARE
        assert M.PERMUTATION <= M.SQUARE
        assert M.PERMUTATION != M.SQUARE
        assert M.NONE <= M.CONTRACTION
        assert M.NONE <= M.SPECTRAL_NORMALIZED
        assert M.NONE <= M.POSITIVE_DEFINITE
        assert M.NONE <= M.NEGATIVE_DEFINITE

        assert not M.SYMMETRIC <= M.ORTHOGONAL
        assert not M.ORTHOGONAL <= M.SYMMETRIC
        assert not M.LOW_RANK <= M.BANDED
        assert not M.BANDED <= M.LOW_RANK
        assert not M.POSITIVE_DEFINITE <= M.NEGATIVE_DEFINITE
        assert not M.NEGATIVE_DEFINITE <= M.POSITIVE_DEFINITE

        assert M.EYE is M.IDENTITY
        assert str(M.SQUARE) == "square"

        with pytest.raises(TypeError):
            _ = M.SQUARE <= "square"

    def test_none_meet_rules(self) -> None:
        none_meets = M.NONE.factorizations
        assert M.CONTRACTION & M.SPECTRAL_NORMALIZED in none_meets
        assert M.INVERTIBLE & M.SINGULAR in none_meets
        assert M.NEGATIVE_DEFINITE & M.POSITIVE_DEFINITE in none_meets

    def test_zero_and_eye_meet_rules(self) -> None:
        zero_meets = M.ZERO.factorizations
        eye_meets = M.EYE.factorizations

        assert M.SYMMETRIC & M.SKEW_SYMMETRIC in zero_meets
        assert M.DIAGONAL, M.ZERO_DIAGONAL in zero_meets
        assert M.POSITIVE_SEMIDEFINITE & M.NEGATIVE_SEMIDEFINITE in zero_meets
        assert M.POSITIVE_SEMIDEFINITE & M.SKEW_SYMMETRIC in zero_meets
        assert M.POSITIVE_DEFINITE & M.ORTHOGONAL in eye_meets
        assert M.DIAGONAL & M.PERMUTATION in eye_meets

    def test_tall_and_wide_membership(self) -> None:
        tall = tensor([[1.0], [2.0]])
        wide = tensor([[1.0, 2.0]])
        square = tensor([[1.0, 2.0], [3.0, 4.0]])

        assert tall in Tall()
        assert tall not in Wide()

        assert wide in Wide()
        assert wide not in Tall()

        assert square in Tall()
        assert square in Wide()

        assert tall in Tall(2, 1)
        assert tall not in Tall(3, 1)
        assert wide in Wide(1, 2)
        assert wide not in Wide(1, 3)

        with pytest.raises(ValueError, match="Tall matrices"):
            Tall(1, 2)

        with pytest.raises(ValueError, match="Wide matrices"):
            Wide(2, 1)

    def test_column_and_row_orthogonal_membership(self) -> None:
        column_orthogonal = tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
        row_orthogonal = tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        square_orthogonal = tensor([[0.0, 1.0], [1.0, 0.0]])
        non_orthogonal = tensor([[1.0, 1.0], [0.0, 0.0], [0.0, 0.0]])

        assert column_orthogonal in ColumnOrthogonal()
        assert column_orthogonal not in RowOrthogonal()

        assert row_orthogonal in RowOrthogonal()
        assert row_orthogonal not in ColumnOrthogonal()

        assert square_orthogonal in ColumnOrthogonal()
        assert square_orthogonal in RowOrthogonal()

        assert non_orthogonal not in ColumnOrthogonal()
        assert non_orthogonal not in RowOrthogonal()

        assert column_orthogonal in ColumnOrthogonal(3, 2)
        assert column_orthogonal not in ColumnOrthogonal(2, 2)
        assert row_orthogonal in RowOrthogonal(2, 3)
        assert row_orthogonal not in RowOrthogonal(2, 2)

    def test_left_and_right_invertible_membership(self) -> None:
        left_invertible = tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
        right_invertible = tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        invertible = tensor([[1.0, 0.0], [0.0, 1.0]])
        rank_deficient_tall = tensor([[1.0, 1.0], [0.0, 0.0], [0.0, 0.0]])
        rank_deficient_wide = tensor([[1.0, 1.0, 0.0], [0.0, 0.0, 0.0]])

        assert is_left_invertible(left_invertible)
        assert not is_right_invertible(left_invertible)

        assert is_right_invertible(right_invertible)
        assert not is_left_invertible(right_invertible)

        assert is_left_invertible(invertible)
        assert is_right_invertible(invertible)

        assert not is_left_invertible(rank_deficient_tall)
        assert not is_right_invertible(rank_deficient_wide)
