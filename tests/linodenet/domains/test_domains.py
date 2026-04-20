import pytest
import torch
from torch import tensor

from linodenet.domains import (
    DomainMapping,
    Interval,
    Join,
    MatrixDomains as M,
    RealDomain,
    ScalarDomains as S,
    TensorDomains as T,
    VectorDomains as V,
    is_left_invertible,
    is_right_invertible,
)
from linodenet.domains.base import Meet
from linodenet.domains.matrix_domains import (
    BackwardStable,
    Banded,
    BlockDiagonal,
    Boolean,
    Circulant,
    ColumnCentered,
    ColumnOrthogonal,
    ColumnStochastic,
    Contraction,
    Diagonal,
    DiagonallyDominant,
    DoublyCentered,
    DoublyStochastic,
    ForwardStable,
    Hamiltonian,
    Identity,
    LeftInvertible,
    LipschitzBounded,
    LowerTriangular,
    LowRankSkewSymmetric,
    LowRankSquare,
    LowRankSymmetric,
    Masked,
    NegativeDefinite,
    NegativeDiagonal,
    NegativeSemidefinite,
    Normal,
    OneHot,
    Ones,
    Orthogonal,
    OrthogonalProjection,
    Permutation,
    PositiveDefinite,
    PositiveDiagonal,
    PositiveScalarMatrix,
    PositiveSemidefinite,
    Projection,
    RankOne,
    RightInvertible,
    RowCentered,
    RowOrthogonal,
    RowStochastic,
    SpecialOrthogonal,
    SpectralNormalized,
    Symplectic,
    Tall,
    Toeplitz,
    Traceless,
    Triangular,
    Tridiagonal,
    UpperTriangular,
    Wide,
    Zero,
)
from linodenet.domains.vector_domains import (
    Boolean as VectorBoolean,
    Complex as VectorComplex,
    Discrete,
    One as VectorOne,
    Real,
    Sparse,
    Zero as VectorZero,
)


class TestScalarDomains:
    def test_partial_order_and_representation(self) -> None:
        assert S.REAL_LINE <= S.REAL_LINE
        assert S.REAL_LINE < S.EXTENDED_LINE
        assert S.REAL_LINE <= "(-inf, inf)"
        assert S.OPEN_UNIT_INTERVAL < "[0, 1]"
        assert S.NONZERO <= "(-inf, 0) | (0, inf)"
        assert S.POSITIVE_REALS < "(-inf, 0) | (0, inf)"

        assert S.OPEN_UNIT_INTERVAL <= S.UNIT_INTERVAL
        assert S.OPEN_UNIT_INTERVAL != S.UNIT_INTERVAL
        assert S.OPEN_UNIT_INTERVAL <= S.NONNEGATIVE_REALS
        assert S.OPEN_UNIT_INTERVAL != S.NONNEGATIVE_REALS
        assert S.OPEN_UNIT_INTERVAL <= S.REAL_LINE
        assert S.OPEN_UNIT_INTERVAL != S.REAL_LINE
        assert S.OPEN_UNIT_INTERVAL <= S.EXTENDED_LINE
        assert S.OPEN_UNIT_INTERVAL != S.EXTENDED_LINE
        assert S.POSITIVE_REALS <= S.NONZERO
        assert S.POSITIVE_REALS != S.NONZERO
        assert S.NONZERO <= S.EXTENDED_LINE
        assert S.NONZERO != S.EXTENDED_LINE

        assert not S.NONNEGATIVE_REALS <= S.NONPOSITIVE_REALS
        assert not S.NONPOSITIVE_REALS <= S.NONNEGATIVE_REALS
        assert not S.UNIT_INTERVAL <= S.NONPOSITIVE_REALS
        assert not S.NEGATIVE_REALS <= S.NONNEGATIVE_REALS
        assert not S.NONZERO <= S.NONNEGATIVE_REALS
        assert not S.NONNEGATIVE_REALS <= S.NONZERO
        assert not S.EXTENDED_LINE <= "(-inf, inf)"
        assert not S.NONNEGATIVE_REALS <= "(0, inf)"

        assert str(S.OPEN_UNIT_INTERVAL) == "(0, 1)"
        assert str(S.UNIT_INTERVAL.value) == "[0, 1]"
        assert isinstance(S.NONZERO.value, RealDomain)

    def test_membership(self) -> None:
        values = tensor([-0.5, 0.0, 0.5, 1.0, 1.5])
        result = [v in S.UNIT_INTERVAL for v in values]
        expected = [False, True, True, True, False]
        assert result == expected

        values = tensor([-1.0, -0.0, 0.0, 1.0])
        result = [v in S.NONZERO for v in values]
        expected = [True, False, False, True]
        assert result == expected

    def test_interval_union_normalization_and_membership(self) -> None:
        domain = RealDomain("[0, 1] | (1, 2) | [3, 4] | [3.5, 5]")
        assert str(domain) == "[0, 2) | [3, 5]"

        domain = RealDomain("(-inf, 0) | (0, inf)")
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
        assert V.NONE.factorizations == frozenset(
            {
                V.NEGATIVE & V.NONNEGATIVE,
                V.NONZERO & V.ZERO,
                V.POSITIVE & V.NONPOSITIVE,
                V.POSITIVE & V.NEGATIVE,
            }
        )

    def test_factorizations(self) -> None:
        assert V.ONE_HOT.factorizations == frozenset(
            {
                V.BOOLEAN & V.STOCHASTIC,
                V.STOCHASTIC & V.UNIT_VECTOR,
            }
        )
        assert V.ZERO.factorizations == frozenset(
            {
                V.NONNEGATIVE & V.NONPOSITIVE,
            }
        )

    def test_partial_order_and_representation(self) -> None:
        assert V.REAL <= V.REAL
        assert V.DISCRETE <= V.REAL
        assert V.ONE_HOT < V.STOCHASTIC
        assert not V.STOCHASTIC < V.STOCHASTIC

        assert V.ONE_HOT <= V.STOCHASTIC
        assert V.ONE_HOT != V.STOCHASTIC
        assert V.ONE_HOT <= V.NONNEGATIVE
        assert V.ONE_HOT != V.NONNEGATIVE
        assert V.ONE_HOT <= V.REAL
        assert V.ONE_HOT != V.REAL
        assert V.ONE_HOT <= V.COMPLEX
        assert V.ONE_HOT != V.COMPLEX

        assert V.STANDARDIZED <= V.ZERO_MEAN
        assert V.STANDARDIZED != V.ZERO_MEAN
        assert V.STANDARDIZED <= V.NONZERO
        assert V.STANDARDIZED != V.NONZERO

        assert not V.NONNEGATIVE <= V.NONPOSITIVE
        assert not V.NONPOSITIVE <= V.NONNEGATIVE
        assert not V.STOCHASTIC <= V.UNIT_VECTOR
        assert not V.UNIT_VECTOR <= V.STOCHASTIC

        assert V("zero-mean") is V.ZERO_MEAN
        assert V("unit-vector") is V.UNIT_VECTOR
        assert V("simplex") is V.STOCHASTIC

        with pytest.raises(TypeError):
            _ = V.REAL <= "real"

    def test_membership(self) -> None:
        real = tensor([0.0, 1.0])
        discrete = tensor([0, 1], dtype=torch.int64)
        boolean = tensor([0.0, 1.0])
        complex_vector = tensor([1.0 + 0.0j, 0.0 + 1.0j])

        assert real in Real()
        assert discrete in Real()
        assert complex_vector not in Real()

        assert discrete in Discrete()
        assert tensor([True, False]) in Discrete()
        assert real not in Discrete()

        assert real in VectorComplex()
        assert complex_vector in VectorComplex()

        assert boolean in VectorBoolean()
        assert tensor([0, 2], dtype=torch.int64) not in VectorBoolean()

        assert tensor([0.0, 0.0]) in VectorZero()
        assert tensor([0, 0], dtype=torch.int64) in VectorZero()
        assert tensor([1.0, 0.0]) not in VectorZero()

        assert tensor([1.0, 1.0]) in VectorOne()
        assert tensor([True, True]) in VectorOne()
        assert tensor([1, 0], dtype=torch.int64) not in VectorOne()

    def test_sparse_membership(self) -> None:
        assert tensor([1.0, 0.0, 2.0]) in Sparse()
        assert tensor([1.0, 2.0, 3.0]) not in Sparse()

        assert tensor([0.0, 1.0, 0.0, 2.0]) in Sparse(sparsity=0.5)
        assert tensor([0.0, 1.0, 2.0, 3.0]) not in Sparse(sparsity=0.5)
        assert tensor([0.0, 0.0, 0.0, 1.0]) in Sparse(sparsity=0.75)

    def test_sparse_validation(self) -> None:
        with pytest.raises(ValueError, match=r"Expected sparsity"):
            Sparse(sparsity=-0.1)

        with pytest.raises(ValueError, match=r"Expected sparsity"):
            Sparse(sparsity=1.1)


class TestTensorDomains:
    def test_partial_order_and_representation(self) -> None:
        assert T.NONE < T.ANY
        assert T.NONE <= T.ZERO
        assert T.NONE <= T.COMPLEX
        assert T.ANY <= T.ANY

        assert T.BOOLEAN <= T.REAL
        assert T.BOOLEAN != T.REAL
        assert T.BOOLEAN <= T.COMPLEX
        assert T.BOOLEAN != T.COMPLEX
        assert T.BOOLEAN <= T.ANY
        assert T.BOOLEAN != T.ANY

        assert T.ZERO <= T.SPARSE
        assert T.ZERO != T.SPARSE
        assert T.ZERO <= T.BOOLEAN
        assert T.ZERO != T.BOOLEAN
        assert T.ONE <= T.NONZERO
        assert T.ONE != T.NONZERO

        assert not T.SPARSE <= T.COMPLEX
        assert not T.NONZERO <= T.SPARSE

        assert str(T.NONZERO) == "nonzero"

        with pytest.raises(TypeError):
            _ = T.ANY <= "any"


class TestMatrixDomains:
    def test_domain_mapping_exact_and_lub_lookup(self) -> None:
        mapping = DomainMapping(
            {
                M.RECTANGULAR: M.RECTANGULAR,
                M.SQUARE: M.SQUARE,
                M.SYMMETRIC: M.POSITIVE_DEFINITE,
            }
        )

        assert mapping[M.SYMMETRIC] is M.POSITIVE_DEFINITE
        assert mapping[M.SKEW_SYMMETRIC] is M.SQUARE
        assert mapping[M.TALL] is M.RECTANGULAR

    def test_domain_mapping_rejects_non_monotone_mapping(self) -> None:
        with pytest.raises(ValueError, match="monotone domain mapping"):
            DomainMapping(
                {
                    M.SQUARE: M.INVERTIBLE,
                    M.SYMMETRIC: M.RECTANGULAR,
                }
            )

    def test_domain_mapping_returns_join_for_ambiguous_lub_lookup(self) -> None:
        mapping = DomainMapping(
            {
                M.TALL: M.LEFT_INVERTIBLE,
                M.WIDE: M.RIGHT_INVERTIBLE,
            }
        )

        codomain = mapping[M.SQUARE]
        assert isinstance(codomain, Join)
        assert codomain.members == frozenset({M.LEFT_INVERTIBLE, M.RIGHT_INVERTIBLE})

    def test_poset_meet_expression(self) -> None:
        meet = M.TALL & M.WIDE & M.SQUARE
        assert isinstance(meet, Meet)
        assert len(meet) == 3
        assert set(meet) == {
            M.TALL,
            M.WIDE,
            M.SQUARE,
        }
        assert M.SQUARE.factorizations == frozenset(
            {M.TALL & M.WIDE, M.ROW_STOCHASTIC & M.COLUMN_STOCHASTIC}
        )
        assert M.ROW_STOCHASTIC & M.COLUMN_STOCHASTIC <= M.SQUARE
        assert M.DOUBLY_STOCHASTIC <= M.SQUARE

    def test_poset_join_expression(self) -> None:
        join = M.TALL | M.WIDE | M.SQUARE
        assert isinstance(join, Join)
        assert len(join) == 3
        assert set(join) == {
            M.TALL,
            M.WIDE,
            M.SQUARE,
        }
        assert M.SQUARE <= join
        assert join <= M.RECTANGULAR
        with pytest.raises(TypeError, match="could not be determined"):
            assert not M.RECTANGULAR <= join

    def test_meet_and_join_inequalities(self) -> None:
        meet = M.TALL & M.WIDE
        join = M.TALL | M.WIDE

        assert M.SQUARE <= meet
        assert meet <= M.SQUARE
        assert meet <= M.RECTANGULAR
        assert not meet <= M.INVERTIBLE
        with pytest.raises(TypeError, match="could not be determined"):
            assert not meet <= (M.LEFT_INVERTIBLE | M.RIGHT_INVERTIBLE)
        assert not M.RECTANGULAR <= meet

        assert join <= M.RECTANGULAR
        assert M.SQUARE <= join
        assert join >= M.SQUARE
        assert not join <= M.SQUARE

        assert meet <= join
        assert join >= meet

    def test_meet_and_join_sufficient_rules_return_notimplemented(self) -> None:
        meet = M.TALL & M.WIDE
        join = M.TALL | M.WIDE

        with pytest.raises(TypeError, match="could not be determined"):
            assert not meet <= (M.LEFT_INVERTIBLE | M.RIGHT_INVERTIBLE)
        with pytest.raises(TypeError, match="could not be determined"):
            assert not M.RECTANGULAR <= join

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
        assert M.NEGATIVE_DIAGONAL <= M.DIAGONAL
        assert M.NEGATIVE_DIAGONAL <= M.NEGATIVE_DIAGONAL_ENTRIES
        assert M.NEGATIVE_DIAGONAL <= M.NEGATIVE_DEFINITE
        assert M.POSITIVE_DIAGONAL <= M.DIAGONAL
        assert M.POSITIVE_DIAGONAL <= M.POSITIVE_DIAGONAL_ENTRIES
        assert M.POSITIVE_DIAGONAL <= M.POSITIVE_DEFINITE
        assert M.POSITIVE_SCALAR_MATRIX <= M.DIAGONAL
        assert M.POSITIVE_SCALAR_MATRIX <= M.POSITIVE_DIAGONAL
        assert M.POSITIVE_SCALAR_MATRIX <= M.POSITIVE_DEFINITE
        assert M.POSITIVE_DIAGONAL_ENTRIES <= M.RECTANGULAR
        assert M.NEGATIVE_DIAGONAL_ENTRIES <= M.RECTANGULAR
        assert M.ZERO_DIAGONAL <= M.RECTANGULAR
        assert M.SKEW_SYMMETRIC <= M.ZERO_DIAGONAL
        assert not M.ZERO_DIAGONAL <= M.DIAGONAL
        assert M.RANK_ONE <= M.LOW_RANK
        assert M.RANK_ONE != M.LOW_RANK
        assert M.RANK_ONE <= M.RECTANGULAR
        assert M.RANK_ONE != M.RECTANGULAR
        assert M.LOW_RANK_SQUARE <= M.LOW_RANK
        assert M.LOW_RANK_SQUARE <= M.SQUARE
        assert M.LOW_RANK_SYMMETRIC <= M.LOW_RANK_SQUARE
        assert M.LOW_RANK_SYMMETRIC <= M.SYMMETRIC
        assert M.LOW_RANK_SKEW_SYMMETRIC <= M.LOW_RANK_SQUARE
        assert M.LOW_RANK_SKEW_SYMMETRIC <= M.SKEW_SYMMETRIC

        assert M.SPECIAL_ORTHOGONAL <= M.ORTHOGONAL
        assert M.SPECIAL_ORTHOGONAL != M.ORTHOGONAL
        assert M.SPECIAL_ORTHOGONAL <= M.INVERTIBLE
        assert M.SPECIAL_ORTHOGONAL != M.INVERTIBLE
        assert M.IDENTITY <= M.DIAGONAL
        assert M.IDENTITY <= M.POSITIVE_SCALAR_MATRIX
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

    def test_low_rank_symmetric_membership(self) -> None:
        low_rank_symmetric = tensor([[2.0, 0.0], [0.0, -1.0]])
        high_rank_symmetric = tensor(
            [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]]
        )
        non_symmetric = tensor([[0.0, 1.0], [0.0, 0.0]])

        assert low_rank_symmetric in LowRankSymmetric(rank=1)
        assert low_rank_symmetric in LowRankSymmetric(2, rank=1)
        assert low_rank_symmetric not in LowRankSymmetric(3, rank=1)

        assert high_rank_symmetric not in LowRankSymmetric(rank=1)
        assert non_symmetric not in LowRankSymmetric(rank=1)

    def test_low_rank_square_membership(self) -> None:
        low_rank_square = tensor([[1.0, 0.0], [0.0, 0.0]])
        high_rank_square = tensor([[1.0, 0.0], [0.0, 2.0]])
        rectangular = tensor([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])

        assert low_rank_square in LowRankSquare(rank=1)
        assert low_rank_square in LowRankSquare(2, rank=1)
        assert low_rank_square not in LowRankSquare(3, rank=1)

        assert high_rank_square not in LowRankSquare(rank=1)
        assert rectangular not in LowRankSquare(rank=2)

    def test_low_rank_skew_symmetric_membership(self) -> None:
        low_rank_skew_symmetric = tensor([[0.0, 2.0], [-2.0, 0.0]])
        high_rank_skew_symmetric = tensor(
            [
                [0.0, 1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 3.0],
                [0.0, 0.0, -3.0, 0.0],
            ]
        )
        non_skew_symmetric = tensor([[0.0, 1.0], [1.0, 0.0]])

        assert low_rank_skew_symmetric in LowRankSkewSymmetric(rank=1)
        assert low_rank_skew_symmetric in LowRankSkewSymmetric(2, rank=1)
        assert low_rank_skew_symmetric not in LowRankSkewSymmetric(3, rank=1)

        assert high_rank_skew_symmetric not in LowRankSkewSymmetric(rank=1)
        assert non_skew_symmetric not in LowRankSkewSymmetric(rank=1)

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

    def test_matrix_domain_wrappers_cover_predicate_families(self) -> None:
        rank_one = tensor([[1.0, 2.0], [0.0, 0.0]])
        orthogonal = tensor([[0.0, 1.0], [1.0, 0.0]])
        special_orthogonal = tensor([[0.0, -1.0], [1.0, 0.0]])
        positive = tensor([[2.0, 0.0], [0.0, 1.0]])
        negative = -positive
        traceless = tensor([[1.0, 0.0], [0.0, -1.0]])
        symplectic = tensor([[0.0, 1.0], [-1.0, 0.0]])
        diagonal = tensor([[1.0, 0.0], [0.0, 2.0]])
        positive_scalar = tensor([[2.0, 0.0], [0.0, 2.0]])
        lower = tensor([[1.0, 0.0], [2.0, 3.0]])
        upper = tensor([[1.0, 2.0], [0.0, 3.0]])
        tridiagonal = tensor([[1.0, 2.0, 0.0], [3.0, 4.0, 5.0], [0.0, 6.0, 7.0]])
        toeplitz = tensor([[1.0, 2.0, 3.0], [4.0, 1.0, 2.0], [5.0, 4.0, 1.0]])
        circulant = tensor([[1.0, 2.0, 3.0], [3.0, 1.0, 2.0], [2.0, 3.0, 1.0]])
        block_diagonal = tensor([[1.0, 2.0, 0.0], [3.0, 4.0, 0.0], [0.0, 0.0, 5.0]])
        banded = tensor([[1.0, 2.0, 0.0], [3.0, 4.0, 5.0], [0.0, 6.0, 7.0]])
        masked = tensor([[1.0, 0.0], [0.0, 4.0]])
        contraction = 0.5 * tensor([[1.0, 0.0], [0.0, 1.0]])
        row_stochastic = tensor([[0.5, 0.5], [0.25, 0.75]])
        column_stochastic = tensor([[0.5, 0.25], [0.5, 0.75]])
        doubly_stochastic = tensor([[0.5, 0.5], [0.5, 0.5]])
        projection = tensor([[1.0, 1.0], [0.0, 0.0]])
        orthogonal_projection = tensor([[1.0, 0.0], [0.0, 0.0]])
        row_centered = tensor([[1.0, -1.0, 0.0], [0.5, 0.5, -1.0]])
        column_centered = tensor([[1.0, 0.5], [-1.0, 0.5], [0.0, -1.0]])
        doubly_centered = tensor([[1.0, -1.0], [-1.0, 1.0]])
        boolean_numeric = tensor([[0.0, 1.0], [1.0, 0.0]])
        boolean_bool = tensor([[False, True], [True, False]])
        zero = tensor([[0.0, 0.0], [0.0, 0.0]])
        ones = tensor([[1.0, 1.0], [1.0, 1.0]])
        one_hot_numeric = tensor([[0.0, 1.0], [0.0, 0.0]])
        one_hot_bool = tensor([[False, True], [False, False]])

        assert boolean_numeric in Boolean()
        assert boolean_bool in Boolean()
        assert zero in Zero()
        assert ones in Ones()
        assert one_hot_numeric in OneHot()
        assert one_hot_bool in OneHot()
        assert rank_one in RankOne()
        assert orthogonal in Normal()
        assert orthogonal in Orthogonal()
        assert special_orthogonal in SpecialOrthogonal()
        assert positive in PositiveSemidefinite()
        assert positive in PositiveDefinite()
        assert negative in NegativeSemidefinite()
        assert negative in NegativeDefinite()
        assert traceless in Traceless()
        assert symplectic in Symplectic()
        assert symplectic in Hamiltonian()
        assert diagonal in Diagonal()
        assert negative in NegativeDiagonal()
        assert diagonal in PositiveDiagonal()
        assert positive_scalar in PositiveScalarMatrix()
        assert lower in Triangular()
        assert upper in Triangular()
        assert lower in LowerTriangular()
        assert upper in UpperTriangular()
        assert tridiagonal in Tridiagonal()
        assert toeplitz in Toeplitz()
        assert circulant in Circulant()
        assert block_diagonal in BlockDiagonal()
        assert block_diagonal in BlockDiagonal(3, block_sizes=(2, 1))
        assert block_diagonal not in BlockDiagonal(3, block_sizes=(1, 2))
        assert banded in Banded(3, 3, lower=-1, upper=1)
        assert masked in Masked(2, 2, mask=tensor([[1, 0], [0, 1]], dtype=torch.bool))
        assert row_stochastic in RowStochastic()
        assert column_stochastic in ColumnStochastic()
        assert doubly_stochastic in DoublyStochastic()
        assert projection in Projection()
        assert orthogonal_projection in OrthogonalProjection()
        assert projection not in OrthogonalProjection()
        assert row_centered in RowCentered()
        assert column_centered in ColumnCentered()
        assert doubly_centered in DoublyCentered()
        assert tensor([[0.0, 1.0], [1.0, 0.0]]) in Permutation()
        assert orthogonal in SpectralNormalized()
        assert orthogonal in LipschitzBounded(lipschitz_bound=1.0)
        assert contraction in Contraction()
        assert positive in DiagonallyDominant()
        assert tensor([[-1.0, 0.0], [0.0, 0.0]]) not in NegativeDiagonal()
        assert tensor([[1.0, 0.0], [0.0, 0.0]]) not in PositiveDiagonal()
        assert diagonal not in PositiveScalarMatrix()
        assert diagonal not in Identity()
        assert tensor([[1.0, 0.0], [0.0, 1.0]]) in Identity()

    def test_stability_wrappers_smoke(self) -> None:
        matrix = tensor([[0.0, 0.0], [0.0, 0.0]])

        assert ForwardStable().check(matrix).shape == ()
        assert BackwardStable().check(matrix).shape == ()
        assert LeftInvertible().check(tensor([[1.0], [0.0]])).shape == ()
        assert RightInvertible().check(tensor([[1.0, 0.0]])).shape == ()
