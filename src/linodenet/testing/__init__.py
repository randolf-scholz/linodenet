r"""Utility functions for testing."""
# ruff: noqa: F403, F405

__all__ = [
    "matrix_tests",
    "vector_tests",
    "assertions",
    # CONSTANTS
    "MATRIX_TESTS",
    "VECTOR_TESTS",
]

from linodenet.domains import MatrixDomains, VectorDomains
from linodenet.testing import assertions, matrix_tests, vector_tests
from linodenet.testing.assertions import *
from linodenet.testing.matrix_tests import *
from linodenet.testing.vector_tests import *

__all__ += matrix_tests.__all__
__all__ += vector_tests.__all__
__all__ += assertions.__all__

VECTOR_TESTS: dict[VectorDomains, VectorTest | VectorTestWithArgs] = {
    VectorDomains.POSITIVE   : is_positive_vector,
    VectorDomains.STOCHASTIC : is_stochastic_vector,
    VectorDomains.UNIT_VECTOR: is_unit_vector,
}  # fmt: skip
r"""Map supported vector domains to their corresponding vector test."""

MATRIX_TESTS: dict[MatrixDomains, MatrixTest | MatrixTestWithArgs] = {
    MatrixDomains.SQUARE               : is_square,
    MatrixDomains.LOW_RANK             : is_low_rank,
    MatrixDomains.RANK_ONE             : is_rank_one,
    MatrixDomains.SYMMETRIC            : is_symmetric,
    MatrixDomains.SKEW_SYMMETRIC       : is_skew_symmetric,
    MatrixDomains.CONTRACTION          : is_contraction,
    MatrixDomains.SPECTRAL_NORMALIZED  : is_spectral_normalized,
    MatrixDomains.LIPSCHITZ_BOUNDED    : is_lipschitz_bounded,
    MatrixDomains.DIAGONALLY_DOMINANT  : is_diagonally_dominant,
    MatrixDomains.NEGATIVE_DEFINITE    : is_negative_definite,
    MatrixDomains.NEGATIVE_SEMIDEFINITE: is_negative_semidefinite,
    MatrixDomains.NORMAL               : is_normal,
    MatrixDomains.ORTHOGONAL           : is_orthogonal,
    MatrixDomains.POSITIVE_DEFINITE    : is_positive_definite,
    MatrixDomains.POSITIVE_SEMIDEFINITE: is_positive_semidefinite,
    MatrixDomains.SPECIAL_ORTHOGONAL   : is_special_orthogonal,
    MatrixDomains.TRACELESS            : is_traceless,
    MatrixDomains.SYMPLECTIC           : is_symplectic,
    MatrixDomains.HAMILTONIAN          : is_hamiltonian,
    MatrixDomains.MASKED               : is_masked,
    MatrixDomains.IDENTITY             : is_identity,
    MatrixDomains.DIAGONAL             : is_diagonal,
    MatrixDomains.TRIDIAGONAL          : is_tridiagonal,
    MatrixDomains.UPPER_TRIANGULAR     : is_upper_triangular,
    MatrixDomains.LOWER_TRIANGULAR     : is_lower_triangular,
    MatrixDomains.BANDED               : is_banded,
}  # fmt: skip
r"""Map supported matrix domains to their corresponding matrix test."""
del MatrixDomains, VectorDomains
