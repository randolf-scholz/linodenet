r"""Utility functions for testing."""
# ruff: noqa: F403, F405

__all__ = [
    "matrix_tests",
    "vector_tests",
    "assertions",
    # CONSTANTS
    "MATRIX_TESTS_WITH_ARGS",
    "VECTOR_TESTS_WITH_ARGS",
    "MATRIX_TESTS",
    "VECTOR_TESTS",
    "TESTS",
]

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

TESTS: dict[str, VectorTest | MatrixTest | VectorTestWithArgs | MatrixTestWithArgs] = {
    **MATRIX_TESTS,
    **VECTOR_TESTS,
    **MATRIX_TESTS_WITH_ARGS,
    **VECTOR_TESTS_WITH_ARGS,
}
