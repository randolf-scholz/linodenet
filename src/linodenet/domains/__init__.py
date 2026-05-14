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
    "Meet",
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
    Meet,
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
    VectorDomains.BOOLEAN        : vector_tests.is_boolean_vector,
    VectorDomains.COMPLEX        : vector_tests.is_complex_vector,
    VectorDomains.DISCRETE       : vector_tests.is_discrete_vector,
    VectorDomains.NEGATIVE       : vector_tests.is_negative_vector,
    VectorDomains.NONNEGATIVE    : vector_tests.is_nonnegative_vector,
    VectorDomains.NONPOSITIVE    : vector_tests.is_nonpositive_vector,
    VectorDomains.NONZERO        : vector_tests.is_nonzero_vector,
    VectorDomains.ONE            : vector_tests.is_one_vector,
    VectorDomains.ONE_HOT        : vector_tests.is_one_hot_vector,
    VectorDomains.POSITIVE       : vector_tests.is_positive_vector,
    VectorDomains.REAL           : vector_tests.is_real_vector,
    VectorDomains.SPARSE         : vector_tests.is_sparse_vector,
    VectorDomains.STANDARDIZED   : vector_tests.is_standardized_vector,
    VectorDomains.STOCHASTIC     : vector_tests.is_stochastic_vector,
    VectorDomains.UNIT_BALL      : vector_tests.is_unit_ball_vector,
    VectorDomains.UNIT_CUBE      : vector_tests.is_unit_cube_vector,
    VectorDomains.UNIT_L1_BALL   : vector_tests.is_unit_l1_ball_vector,
    VectorDomains.UNIT_L1_SPHERE : vector_tests.is_unit_l1_sphere_vector,
    VectorDomains.UNIT_VECTOR    : vector_tests.is_unit_vector,
    VectorDomains.ZERO           : vector_tests.is_zero_vector,
    VectorDomains.ZERO_MEAN      : vector_tests.is_zero_mean_vector,
}  # fmt: skip
r"""Map supported vector domains to their corresponding vector test."""

MATRIX_TESTS: dict[MatrixDomains, MatrixTest | MatrixTestWithArgs] = {
    MatrixDomains.BANDED                  : matrix_tests.is_banded,
    MatrixDomains.BLOCK_DIAGONAL          : matrix_tests.is_block_diagonal,
    MatrixDomains.BOOLEAN                 : matrix_tests.is_boolean,
    MatrixDomains.CHOLESKY_FACTOR         : matrix_tests.is_cholesky_factor,
    MatrixDomains.CIRCULANT               : matrix_tests.is_circulant,
    MatrixDomains.COLUMN_CENTERED         : matrix_tests.is_column_centered,
    MatrixDomains.COLUMN_ORTHOGONAL       : matrix_tests.is_column_orthogonal,
    MatrixDomains.COLUMN_STOCHASTIC       : matrix_tests.is_column_stochastic,
    MatrixDomains.CONTRACTION             : matrix_tests.is_contraction,
    MatrixDomains.DIAGONAL                : matrix_tests.is_diagonal,
    MatrixDomains.DIAGONALLY_DOMINANT     : matrix_tests.is_diagonally_dominant,
    MatrixDomains.DOUBLY_CENTERED         : matrix_tests.is_doubly_centered,
    MatrixDomains.DOUBLY_STOCHASTIC       : matrix_tests.is_doubly_stochastic,
    MatrixDomains.HAMILTONIAN             : matrix_tests.is_hamiltonian,
    MatrixDomains.IDENTITY                : matrix_tests.is_identity,
    MatrixDomains.LEFT_INVERTIBLE         : matrix_tests.is_left_invertible,
    MatrixDomains.LIPSCHITZ_BOUNDED       : matrix_tests.is_lipschitz_bounded,
    MatrixDomains.LOWER_TRIANGULAR        : matrix_tests.is_lower_triangular,
    MatrixDomains.LOW_RANK                : matrix_tests.is_low_rank,
    MatrixDomains.LOW_RANK_SKEW_SYMMETRIC : matrix_tests.is_low_rank_skew_symmetric,
    MatrixDomains.LOW_RANK_SQUARE         : matrix_tests.is_low_rank_square,
    MatrixDomains.LOW_RANK_SYMMETRIC      : matrix_tests.is_low_rank_symmetric,
    MatrixDomains.MASKED                  : matrix_tests.is_masked,
    MatrixDomains.NEGATIVE_DEFINITE       : matrix_tests.is_negative_definite,
    MatrixDomains.NEGATIVE_DIAGONAL       : matrix_tests.is_negative_diagonal,
    MatrixDomains.NEGATIVE_SEMIDEFINITE   : matrix_tests.is_negative_semidefinite,
    MatrixDomains.NORMAL                  : matrix_tests.is_normal,
    MatrixDomains.ONES                    : matrix_tests.is_ones,
    MatrixDomains.ONE_HOT                 : matrix_tests.is_one_hot,
    MatrixDomains.ORTHOGONAL              : matrix_tests.is_orthogonal,
    MatrixDomains.ORTHOGONAL_PROJECTION   : matrix_tests.is_orthogonal_projection,
    MatrixDomains.PERMUTATION             : matrix_tests.is_permutation,
    MatrixDomains.POSITIVE_DEFINITE       : matrix_tests.is_positive_definite,
    MatrixDomains.POSITIVE_DIAGONAL       : matrix_tests.is_positive_diagonal,
    MatrixDomains.POSITIVE_SCALAR_MATRIX  : matrix_tests.is_positive_scalar_matrix,
    MatrixDomains.POSITIVE_SEMIDEFINITE   : matrix_tests.is_positive_semidefinite,
    MatrixDomains.PROJECTION              : matrix_tests.is_projection,
    MatrixDomains.RANK_ONE                : matrix_tests.is_rank_one,
    MatrixDomains.RIGHT_INVERTIBLE        : matrix_tests.is_right_invertible,
    MatrixDomains.ROW_CENTERED            : matrix_tests.is_row_centered,
    MatrixDomains.ROW_ORTHOGONAL          : matrix_tests.is_row_orthogonal,
    MatrixDomains.ROW_STOCHASTIC          : matrix_tests.is_row_stochastic,
    MatrixDomains.SKEW_SYMMETRIC          : matrix_tests.is_skew_symmetric,
    MatrixDomains.SPARSE                  : matrix_tests.is_sparse,
    MatrixDomains.SPECIAL_ORTHOGONAL      : matrix_tests.is_special_orthogonal,
    MatrixDomains.SPECTRAL_NORMALIZED     : matrix_tests.is_spectral_normalized,
    MatrixDomains.SQUARE                  : matrix_tests.is_square,
    MatrixDomains.SYMMETRIC               : matrix_tests.is_symmetric,
    MatrixDomains.SYMPLECTIC              : matrix_tests.is_symplectic,
    MatrixDomains.TOEPLITZ                : matrix_tests.is_toeplitz,
    MatrixDomains.TRACELESS               : matrix_tests.is_traceless,
    MatrixDomains.TRIDIAGONAL             : matrix_tests.is_tridiagonal,
    MatrixDomains.UPPER_TRIANGULAR        : matrix_tests.is_upper_triangular,
    MatrixDomains.ZERO                    : matrix_tests.is_zero,
}  # fmt: skip
r"""Map supported matrix domains to their corresponding matrix test."""
