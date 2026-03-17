r"""Utility functions for testing."""
# ruff: noqa: F403, F405

__all__ = [
    "matrix_tests",
    "assertions",
    # CONSTANTS
    "MATRIX_TESTS_WITH_EXTRA_ARG",
    "MATRIX_TESTS",
]

from linodenet.testing import assertions, matrix_tests
from linodenet.testing.assertions import *
from linodenet.testing.matrix_tests import *

__all__ += matrix_tests.__all__
__all__ += assertions.__all__

MATRIX_TESTS: dict[str, MatrixTest] = {
    "is_contraction"         : is_contraction,
    "is_diagonal"            : is_diagonal,
    "is_diagonally_dominant" : is_diagonally_dominant,
    "is_hamiltonian"         : is_hamiltonian,
    "is_lower_triangular"    : is_lower_triangular,
    "is_normal"              : is_normal,
    "is_orthogonal"          : is_orthogonal,
    "is_rank_one"            : is_rank_one,
    "is_skew_symmetric"      : is_skew_symmetric,
    "is_square"              : is_square,
    "is_symmetric"           : is_symmetric,
    "is_symplectic"          : is_symplectic,
    "is_traceless"           : is_traceless,
    "is_tridiagonal"         : is_tridiagonal,
    "is_upper_triangular"    : is_upper_triangular,
}  # fmt: skip
r"""Dictionary of all available matrix tests."""

MATRIX_TESTS_WITH_EXTRA_ARG = {
    "is_masked": is_masked,
    "is_low_rank": is_low_rank,
    "is_banded": is_banded,
}
