r"""Projections for the Linear ODE Networks.

Notes:
    Contains projections in both modular and functional form.
      - See `~linodenet.projections.functional` for functional implementations.
      - See `~linodenet.projections.modular` for modular implementations.
"""

__all__ = [
    # Constants
    "PROJECTIONS",
    "FUNCTIONAL_PROJECTIONS",
    "MODULAR_PROJECTIONS",
    "MATRIX_TESTS",
    # Protocols
    "MatrixTest",
    "Projection",
    # Base Classes
    "ProjectionABC",
    # Sub-Modules
    "functional",
    "modular",
    # is_*-checks
    "is_hamiltonian",
    "is_lowrank",
    "is_normal",
    "is_orthogonal",
    "is_skew_symmetric",
    "is_symmetric",
    "is_symplectic",
    "is_traceless",
    # is_*-masked
    "is_banded",
    "is_diagonal",
    "is_lower_triangular",
    "is_masked",
    "is_upper_triangular",
    # Functions
    "hamiltonian",
    "identity",
    "low_rank",
    "normal",
    "orthogonal",
    "skew_symmetric",
    "symmetric",
    "symplectic",
    "traceless",
    # masked
    "banded",
    "diagonal",
    "lower_triangular",
    "upper_triangular",
    "masked",
    # Classes
    "Hamiltonian",
    "Identity",
    "LowRank",
    "Normal",
    "Orthogonal",
    "SkewSymmetric",
    "Symmetric",
    "Symplectic",
    "Traceless",
    # Masked
    "Banded",
    "Diagonal",
    "LowerTriangular",
    "Masked",
    "UpperTriangular",
]


from linodenet.projections import functional, modular
from linodenet.projections.functional import (
    Projection,
    banded,
    diagonal,
    hamiltonian,
    identity,
    low_rank,
    lower_triangular,
    masked,
    normal,
    orthogonal,
    skew_symmetric,
    symmetric,
    symplectic,
    traceless,
    upper_triangular,
)
from linodenet.projections.modular import (
    Banded,
    Diagonal,
    Hamiltonian,
    Identity,
    LowerTriangular,
    LowRank,
    Masked,
    Normal,
    Orthogonal,
    ProjectionABC,
    SkewSymmetric,
    Symmetric,
    Symplectic,
    Traceless,
    UpperTriangular,
)
from linodenet.projections.testing import (
    MatrixTest,
    is_banded,
    is_diagonal,
    is_hamiltonian,
    is_lower_triangular,
    is_lowrank,
    is_masked,
    is_normal,
    is_orthogonal,
    is_skew_symmetric,
    is_symmetric,
    is_symplectic,
    is_traceless,
    is_upper_triangular,
)

MATRIX_TESTS: dict[str, MatrixTest] = {
    "is_hamiltonian": is_hamiltonian,
    "is_lowrank": is_lowrank,
    "is_normal": is_normal,
    "is_orthogonal": is_orthogonal,
    "is_skew_symmetric": is_skew_symmetric,
    "is_symmetric": is_symmetric,
    "is_symplectic": is_symplectic,
    "is_traceless": is_traceless,
    # masked
    "is_banded": is_banded,
    "is_diagonal": is_diagonal,
    "is_lower_triangular": is_lower_triangular,
    "is_masked": is_masked,
    "is_upper_triangular": is_upper_triangular,
}

FUNCTIONAL_PROJECTIONS: dict[str, Projection] = {
    # Projections
    "hamiltonian": hamiltonian,
    "identity": identity,
    "low_rank": low_rank,
    "normal": normal,
    "orthogonal": orthogonal,
    "skew_symmetric": skew_symmetric,
    "symmetric": symmetric,
    "symplectic": symplectic,
    "traceless": traceless,
    # masked
    "banded": banded,
    "diagonal": diagonal,
    "lower_triangular": lower_triangular,
    "upper_triangular": upper_triangular,
    "masked": masked,
}
r"""Dictionary of all available modular metrics."""

MODULAR_PROJECTIONS: dict[str, type[Projection]] = {
    "Hamiltonian": Hamiltonian,
    "Identity": Identity,
    "LowRank": LowRank,
    "Normal": Normal,
    "Orthogonal": Orthogonal,
    "SkewSymmetric": SkewSymmetric,
    "Symmetric": Symmetric,
    "Symplectic": Symplectic,
    "Traceless": Traceless,
    # Masked
    "Banded": Banded,
    "Diagonal": Diagonal,
    "LowerTriangular": LowerTriangular,
    "Masked": Masked,
    "UpperTriangular": UpperTriangular,
}
r"""Dictionary of all available modular metrics."""

PROJECTIONS: dict[str, Projection | type[Projection]] = {
    **FUNCTIONAL_PROJECTIONS,
    **MODULAR_PROJECTIONS,
}
r"""Dictionary containing all available projections."""
