r"""WORK IN PROGRESS.

Domains should allow:

1. checking membership of tensors
2. checking subset relations between domains
3. performing some basic operations (e.g. product of domains, union, intersection)
"""
# ruff: noqa: F403, F405

__all__ = [
    # Constants
    "MATRIX_TESTS",
    "VECTOR_TESTS",
    # Protocols
    "Domain",
    "DomainMapping",
    "Inverse",
    "Join",
    # Domains
    "Interval",
    "MatrixDomain",
    "RealDomain",
    "ScalarDomain",
    "VectorDomain",
    "TensorDomain",
    # Enums
    "ScalarDomains",
    "VectorDomains",
    "MatrixDomains",
    "TensorDomains",
]

from . import matrix_tests, vector_tests
from .base import (
    Domain,
    DomainMapping,
    Inverse,
    Join,
    MatrixDomain,
    ScalarDomain,
    TensorDomain,
    VectorDomain,
)
from .matrix_domains import MatrixDomains
from .matrix_tests import *
from .scalar_domains import Interval, RealDomain, ScalarDomains
from .tensor_domains import TensorDomains
from .vector_domains import VectorDomains
from .vector_tests import *

__all__ += matrix_tests.__all__
__all__ += vector_tests.__all__

VECTOR_TESTS: dict[VectorDomains, VectorTest | VectorTestWithArgs] = {
    VectorDomains.POSITIVE   : vector_tests.is_positive_vector,
    VectorDomains.STOCHASTIC : vector_tests.is_stochastic_vector,
    VectorDomains.UNIT_VECTOR: vector_tests.is_unit_vector,
}  # fmt: skip
r"""Map supported vector domains to their corresponding vector test."""

MATRIX_TESTS: dict[MatrixDomains, MatrixTest | MatrixTestWithArgs] = {
    MatrixDomains.SQUARE                  : matrix_tests.is_square,
    MatrixDomains.BOOLEAN                 : matrix_tests.is_boolean,
    MatrixDomains.ZERO                    : matrix_tests.is_zero,
    MatrixDomains.ONES                    : matrix_tests.is_ones,
    MatrixDomains.ONE_HOT                 : matrix_tests.is_one_hot,
    MatrixDomains.LOW_RANK                : matrix_tests.is_low_rank,
    MatrixDomains.LOW_RANK_SQUARE         : matrix_tests.is_low_rank_square,
    MatrixDomains.LOW_RANK_SKEW_SYMMETRIC: matrix_tests.is_low_rank_skew_symmetric,
    MatrixDomains.LOW_RANK_SYMMETRIC     : matrix_tests.is_low_rank_symmetric,
    MatrixDomains.RANK_ONE                : matrix_tests.is_rank_one,
    MatrixDomains.SYMMETRIC               : matrix_tests.is_symmetric,
    MatrixDomains.SKEW_SYMMETRIC          : matrix_tests.is_skew_symmetric,
    MatrixDomains.CONTRACTION             : matrix_tests.is_contraction,
    MatrixDomains.COLUMN_ORTHOGONAL       : matrix_tests.is_column_orthogonal,
    MatrixDomains.SPECTRAL_NORMALIZED     : matrix_tests.is_spectral_normalized,
    MatrixDomains.LEFT_INVERTIBLE         : matrix_tests.is_left_invertible,
    MatrixDomains.LIPSCHITZ_BOUNDED       : matrix_tests.is_lipschitz_bounded,
    MatrixDomains.DIAGONALLY_DOMINANT     : matrix_tests.is_diagonally_dominant,
    MatrixDomains.NEGATIVE_DEFINITE       : matrix_tests.is_negative_definite,
    MatrixDomains.NEGATIVE_SEMIDEFINITE: matrix_tests.is_negative_semidefinite,
    MatrixDomains.NORMAL                  : matrix_tests.is_normal,
    MatrixDomains.ORTHOGONAL              : matrix_tests.is_orthogonal,
    MatrixDomains.POSITIVE_DEFINITE       : matrix_tests.is_positive_definite,
    MatrixDomains.POSITIVE_SEMIDEFINITE: matrix_tests.is_positive_semidefinite,
    MatrixDomains.RIGHT_INVERTIBLE        : matrix_tests.is_right_invertible,
    MatrixDomains.ROW_ORTHOGONAL          : matrix_tests.is_row_orthogonal,
    MatrixDomains.ROW_STOCHASTIC          : matrix_tests.is_row_stochastic,
    MatrixDomains.ROW_CENTERED            : matrix_tests.is_row_centered,
    MatrixDomains.COLUMN_STOCHASTIC       : matrix_tests.is_column_stochastic,
    MatrixDomains.COLUMN_CENTERED         : matrix_tests.is_column_centered,
    MatrixDomains.DOUBLY_CENTERED         : matrix_tests.is_doubly_centered,
    MatrixDomains.DOUBLY_STOCHASTIC       : matrix_tests.is_doubly_stochastic,
    MatrixDomains.SPECIAL_ORTHOGONAL      : matrix_tests.is_special_orthogonal,
    MatrixDomains.TRACELESS               : matrix_tests.is_traceless,
    MatrixDomains.SYMPLECTIC              : matrix_tests.is_symplectic,
    MatrixDomains.HAMILTONIAN             : matrix_tests.is_hamiltonian,
    MatrixDomains.MASKED                  : matrix_tests.is_masked,
    MatrixDomains.IDENTITY                : matrix_tests.is_identity,
    MatrixDomains.PERMUTATION             : matrix_tests.is_permutation,
    MatrixDomains.DIAGONAL                : matrix_tests.is_diagonal,
    MatrixDomains.TRIDIAGONAL             : matrix_tests.is_tridiagonal,
    MatrixDomains.UPPER_TRIANGULAR        : matrix_tests.is_upper_triangular,
    MatrixDomains.LOWER_TRIANGULAR        : matrix_tests.is_lower_triangular,
    MatrixDomains.BANDED                  : matrix_tests.is_banded,
}  # fmt: skip
r"""Map supported matrix domains to their corresponding matrix test."""
