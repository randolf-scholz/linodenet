r"""Matrix-specific domain primitives and partial-order labels."""

__all__ = ["MatrixDomain", "MatrixDomains"]

from types import MappingProxyType

from torch import Tensor

from .base import Domain, Intersection, Inverse, PosetEnum, Union


class MatrixDomain(Domain):
    r"""Stub base class for matrix domains."""

    def __contains__(self, item: Tensor, /) -> bool:
        raise NotImplementedError

    def __le__(self, other: MatrixDomain, /) -> bool:
        return NotImplemented

    def __lt__(self, other: MatrixDomain, /) -> bool:
        raise NotImplementedError

    def __gt__(self, other: MatrixDomain, /) -> bool:
        return NotImplemented

    def __ge__(self, other: MatrixDomain, /) -> bool:
        return NotImplemented

    def __invert__(self) -> Inverse[MatrixDomain]:
        return Inverse(self)

    def __or__(self, other: MatrixDomain, /) -> Union[MatrixDomain]:
        return Union({self, other})

    def __and__(self, other: MatrixDomain, /) -> Intersection[MatrixDomain]:
        return Intersection({self, other})


class MatrixDomains(PosetEnum):
    r"""Enumeration of some matrix domains."""

    ANY = "any"  # top node
    NONE = "none"  # bottom node

    BOOLEAN = "boolean"  # only 0 and 1 entries
    SPARSE = "sparse"  # many 0 entries
    MASKED = "masked"  # X⊙M = X for some mask M

    RECTANGULAR = "rectangular"  # m × n matrices
    SQUARE = "square"  # n × n matrices
    EVEN_SQUARE = "even_square"  # 2n × 2n matrices

    # specific matrices
    ZERO = "zero"  # the zero matrix
    IDENTITY = "identity"  # the identity matrix
    STANDARD_SYMPLECTIC = "standard_symplectic"  # [0, 𝕀; -𝕀, 0]

    # rank
    LOW_RANK = "low_rank"  # rank small relative to size
    RANK_ONE = "rank_one"

    # determinant-based
    SINGULAR = "singular"  # det=0
    INVERTIBLE = "invertible"  # GLₙ(R) (det≠0)
    UNIT_DETERMINANT = "unit_determinant"  # SLₙ(R) (det=1)
    GENERAL_LINEAR = "invertible"  # alias
    SPECIAL_LINEAR = "unit_determinant"  # alias
    POSITIVE_DETERMINANT = "positive_determinant"  # GLₙ⁺(R) (det>0)
    NEGATIVE_DETERMINANT = "negative_determinant"  # GLₙ⁻(R) (det<0)

    # symmetry / entry based
    SYMMETRIC = "symmetric"  # 𝕊ₙ(R)
    SKEW_SYMMETRIC = "skew_symmetric"
    DIAGONAL = "diagonal"
    TRIDIAGONAL = "tridiagonal"
    BANDED = "banded"
    TOEPLITZ = "toeplitz"  # constant along diagonals
    HANKEL = "hankel"  # constant along anti-diagonals
    CIRCULANT = "circulant"  # constant along diagonals, wrap around

    # eigenvalues
    POSITIVE_DEFINITE = "positive_definite"  # 𝕊ₙ⁺(ℝ)
    NEGATIVE_DEFINITE = "negative_definite"  # 𝕊ₙ⁻(ℝ)
    POSITIVE_SEMIDEFINITE = "positive_semidefinite"  # 𝕊ₙ⁺(ℝ) ∪ {0}
    NEGATIVE_SEMIDEFINITE = "negative_semidefinite"  # 𝕊ₙ⁻(ℝ) ∪ {0}

    CONTRACTION = "contraction"  # ‖A‖₂ < 1
    SPECTRAL_NORMALIZED = "spectral_normalized"  # ‖A‖₂ = 1
    LIPSCHITZ_BOUNDED = "lipschitz_bounded"  # ‖A‖₂ ≤ C
    DIAGONALLY_DOMINANT = "diagonally_dominant"  # |Aᵢᵢ| ≥ ∑_{j≠i} |Aᵢⱼ| for all i

    NORMAL = "normal"
    ORTHOGONAL = "orthogonal"  # Oₙ(R)
    CAYLEY_ORTHOGONAL = "cayley_orthogonal"  # {Q ∈ SOₙ(n) ∣ -1 ∉ spec(Q)}
    SPECIAL_ORTHOGONAL = "special_orthogonal"  # SOₙ(R)
    PERMUTATION = "permutation"

    TRACELESS = "traceless"
    SYMPLECTIC = "symplectic"  # 2n×2n with AᵀJA = J for J=[0, I;-I, 0]
    HAMILTONIAN = "hamiltonian"  # 2n×2n with (JA)ᵀ = JA for J=[0, I;-I, 0]

    TRIANGULAR = "triangular"  # lower or upper
    UPPER_TRIANGULAR = "upper_triangular"
    LOWER_TRIANGULAR = "lower_triangular"

    ROW_STOCHASTIC = "row_stochastic"
    COLUMN_STOCHASTIC = "column_stochastic"
    DOUBLY_STOCHASTIC = "doubly_stochastic"

    IDEMPOTENT = "idempotent"  # Aᵏ = A for some k≥2
    PROJECTION = "projection"  # A² = A
    NILPOTENT = "nilpotent"  # Aᵏ = 0 for some k≥2

    EFFICIENTLY_INVERTIBLE = "efficient_invertible"

    HADAMARD = "hadamard"  # entries ±1, HHᵀ=n𝕀

    def __contains__(self, item: Tensor, /) -> Tensor:
        raise NotImplementedError


M = MatrixDomains  # temporary alias
MatrixDomains.KNOWN_EDGES = MappingProxyType({
    M.BANDED: frozenset({M.RECTANGULAR}),
    M.CAYLEY_ORTHOGONAL: frozenset({M.SPECIAL_ORTHOGONAL}),
    M.COLUMN_STOCHASTIC: frozenset({M.RECTANGULAR}),
    M.CONTRACTION: frozenset({M.RECTANGULAR}),
    M.DIAGONAL: frozenset({M.SYMMETRIC, M.TRIDIAGONAL, M.UPPER_TRIANGULAR, M.LOWER_TRIANGULAR}),
    M.DIAGONALLY_DOMINANT: frozenset({M.SQUARE}),
    M.DOUBLY_STOCHASTIC: frozenset({M.ROW_STOCHASTIC, M.COLUMN_STOCHASTIC, M.SQUARE}),
    M.EVEN_SQUARE: frozenset({M.SQUARE}),
    M.IDENTITY: frozenset({M.DIAGONAL, M.PERMUTATION, M.SPECIAL_ORTHOGONAL}),
    M.INVERTIBLE: frozenset({M.SQUARE}),
    M.LIPSCHITZ_BOUNDED: frozenset({M.RECTANGULAR}),
    M.LOWER_TRIANGULAR: frozenset({M.SQUARE, M.TRIANGULAR}),
    M.LOW_RANK: frozenset({M.RECTANGULAR}),
    M.NEGATIVE_DEFINITE: frozenset({M.SYMMETRIC, M.INVERTIBLE, M.NEGATIVE_SEMIDEFINITE}),
    M.NEGATIVE_DETERMINANT: frozenset({M.INVERTIBLE}),
    M.NEGATIVE_SEMIDEFINITE: frozenset({M.SYMMETRIC}),
    M.NORMAL: frozenset({M.SQUARE}),
    M.ORTHOGONAL: frozenset({M.SQUARE, M.INVERTIBLE, M.NORMAL, M.SPECTRAL_NORMALIZED}),
    M.PERMUTATION: frozenset({M.SPARSE, M.ORTHOGONAL, M.DOUBLY_STOCHASTIC}),
    M.POSITIVE_DEFINITE: frozenset({M.SYMMETRIC, M.INVERTIBLE, M.POSITIVE_SEMIDEFINITE}),
    M.POSITIVE_DETERMINANT: frozenset({M.INVERTIBLE}),
    M.POSITIVE_SEMIDEFINITE: frozenset({M.SYMMETRIC}),
    M.RANK_ONE: frozenset({M.LOW_RANK}),
    M.ROW_STOCHASTIC: frozenset({M.RECTANGULAR}),
    M.SINGULAR: frozenset({M.SQUARE}),
    M.SKEW_SYMMETRIC: frozenset({M.SQUARE, M.NORMAL}),
    M.SPARSE: frozenset({M.BOOLEAN}),
    M.SPECIAL_ORTHOGONAL: frozenset({M.ORTHOGONAL, M.UNIT_DETERMINANT}),
    M.SPECTRAL_NORMALIZED: frozenset({M.RECTANGULAR, M.LIPSCHITZ_BOUNDED}),
    M.SQUARE: frozenset({M.RECTANGULAR}),
    M.SYMMETRIC: frozenset({M.SQUARE, M.NORMAL}),
    M.SYMPLECTIC: frozenset({M.EVEN_SQUARE, M.UNIT_DETERMINANT}),
    M.TRACELESS: frozenset({M.SQUARE}),
    M.TRIDIAGONAL: frozenset({M.BANDED, M.SQUARE}),
    M.UPPER_TRIANGULAR: frozenset({M.SQUARE, M.TRIANGULAR}),
    M.UNIT_DETERMINANT: frozenset({M.POSITIVE_DETERMINANT}),
    M.ZERO: frozenset({M.SPARSE}),
    M.TOEPLITZ: frozenset({M.RECTANGULAR}),
    M.CIRCULANT: frozenset({M.TOEPLITZ, M.SQUARE}),
    M.HANKEL: frozenset({M.RECTANGULAR}),
    M.STANDARD_SYMPLECTIC: frozenset({M.SYMPLECTIC, M.SKEW_SYMMETRIC}),
    M.HAMILTONIAN: frozenset({M.EVEN_SQUARE, M.TRACELESS}),
})  # fmt: skip
MatrixDomains.KNOWN_TAGS = MappingProxyType({
    M.EFFICIENTLY_INVERTIBLE: frozenset({
        M.SPARSE, M.PERMUTATION, M.ORTHOGONAL,
        M.TRIANGULAR, M.DIAGONAL, M.TRIDIAGONAL,
    }),
})  # fmt: skip
del M  # remove alias
