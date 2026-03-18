r"""Utility functions for testing."""
# ruff: noqa: F403, F405

__all__ = [
    "matrix_tests",
    "vector_tests",
    "assertions",
    # CONSTANTS
    "MATRIX_DOMAIN_TESTS",
    "MATRIX_TESTS_WITH_ARGS",
    "VECTOR_TESTS_WITH_ARGS",
    "MATRIX_TESTS",
    "VECTOR_TESTS",
    "TESTS",
]

from linodenet.domains import MatrixDomains
from linodenet.testing import assertions, matrix_tests, vector_tests
from linodenet.testing.assertions import *
from linodenet.testing.matrix_tests import *
from linodenet.testing.vector_tests import *

__all__ += matrix_tests.__all__
__all__ += vector_tests.__all__
__all__ += assertions.__all__

VECTOR_TESTS: dict[str, VectorTest] = {
    "is_positive_vector"   : is_positive_vector,
    "is_stochastic_vector" : is_stochastic_vector,
    "is_unit_vector"       : is_unit_vector,
}  # fmt: skip
r"""Dictionary of all available vector tests."""

VECTOR_TESTS_WITH_ARGS: dict[str, VectorTestWithArgs] = {}
r"""Dictionary of all available vector tests."""

MATRIX_TESTS: dict[str, MatrixTest] = {
    "is_contraction"         : is_contraction,
    "is_diagonal"            : is_diagonal,
    "is_diagonally_dominant" : is_diagonally_dominant,
    "is_hamiltonian"         : is_hamiltonian,
    "is_identity"            : is_identity,
    "is_lower_triangular"    : is_lower_triangular,
    "is_normal"              : is_normal,
    "is_orthogonal"          : is_orthogonal,
    "is_special_orthogonal"  : is_special_orthogonal,
    "is_rank_one"            : is_rank_one,
    "is_skew_symmetric"      : is_skew_symmetric,
    "is_spectral_normalized" : is_spectral_normalized,
    "is_square"              : is_square,
    "is_symmetric"           : is_symmetric,
    "is_symplectic"          : is_symplectic,
    "is_traceless"           : is_traceless,
    "is_tridiagonal"         : is_tridiagonal,
    "is_upper_triangular"    : is_upper_triangular,
}  # fmt: skip
r"""Dictionary of all available matrix tests."""

MATRIX_TESTS_WITH_ARGS: dict[str, MatrixTestWithArgs] = {
    "is_banded"            : is_banded,
    "is_lipschitz_bounded" : is_lipschitz_bounded,
    "is_low_rank"          : is_low_rank,
    "is_masked"            : is_masked,
}  # fmt: skip
r"""Matrix tests that require an additional argument."""

MATRIX_DOMAIN_TESTS = {
    MatrixDomains.SQUARE               : is_square,
    MatrixDomains.LOW_RANK             : is_low_rank,
    MatrixDomains.RANK_ONE             : is_rank_one,
    MatrixDomains.SYMMETRIC            : is_symmetric,
    MatrixDomains.SKEW_SYMMETRIC       : is_skew_symmetric,
    MatrixDomains.CONTRACTION          : is_contraction,
    MatrixDomains.SPECTRAL_NORMALIZED  : is_spectral_normalized,
    MatrixDomains.LIPSCHITZ_BOUNDED    : is_lipschitz_bounded,
    MatrixDomains.DIAGONALLY_DOMINANT  : is_diagonally_dominant,
    MatrixDomains.NORMAL               : is_normal,
    MatrixDomains.ORTHOGONAL           : is_orthogonal,
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
del MatrixDomains

TESTS: dict[str, VectorTest | MatrixTest | VectorTestWithArgs | MatrixTestWithArgs] = {
    **MATRIX_TESTS,
    **VECTOR_TESTS,
    **MATRIX_TESTS_WITH_ARGS,
    **VECTOR_TESTS_WITH_ARGS,
}
