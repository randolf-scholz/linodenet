r"""Projections for the Linear ODE Networks.

Notes:
    - See `linodenet.projections.functional` for functional implementations.
    - See `linodenet.projections.modules` for module-based implementations.
"""
# ruff: noqa: F403, F405

__all__ = [
    # Sub-Modules
    "base",
    "embeddings",
    "projections",
    "surjections",
    "functional",
    # Constants
    "FUNCTIONAL_PROJECTIONS",
    "MODULAR_PROJECTIONS",
    "SPECIAL_PROJECTIONS",
    "PROJECTIONS",
    "EMBEDDINGS",
    "SURJECTIONS",
]

from linodenet.mappings import base, embeddings, functional, projections, surjections
from linodenet.mappings.base import *
from linodenet.mappings.embeddings import *
from linodenet.mappings.functional import *
from linodenet.mappings.projections import *
from linodenet.mappings.surjections import *

__all__ += base.__all__
__all__ += embeddings.__all__
__all__ += surjections.__all__
__all__ += projections.__all__
__all__ += functional.__all__

EMBEDDINGS: dict[str, type[EmbeddingBase]] = {
    "ConcatEmbedding"  : ConcatEmbedding,
    "LinearEmbedding"  : LinearEmbedding,
}  # fmt: skip
r"""Dictionary of available embeddings."""

SURJECTIONS: dict[str, type[SurjectionBase]] = {

}  # fmt: skip
r"""Dictionary containing all available surjections."""


FUNCTIONAL_PROJECTIONS: dict[str, FunctionalProjection] = {
    "diagonal"            : functional.diagonal,
    "diagonally_dominant" : functional.diagonally_dominant,
    "hamiltonian"         : functional.hamiltonian,
    "identity"            : functional.identity,
    "lower_triangular"    : functional.lower_triangular,
    "normal"              : functional.normal,
    "orthogonal"          : functional.orthogonal,
    "rank_one"            : functional.rank_one,
    "skew_symmetric"      : functional.skew_symmetric,
    "symmetric"           : functional.symmetric,
    "symplectic"          : functional.symplectic,
    "traceless"           : functional.traceless,
    "tridiagonal"         : functional.tridiagonal,
    "upper_triangular"    : functional.upper_triangular,
}  # fmt: skip
r"""Dictionary of all available modular metrics."""

SPECIAL_PROJECTIONS = {
    "banded"       : functional.banded,
    "low_rank"     : functional.low_rank,
    "masked"       : functional.masked,
    "spectral_norm": functional.spectral_norm,
}  # fmt: skip
r"""Projections that require additional arguments"""

MODULAR_PROJECTIONS: dict[str, type[ProjectionBase]] = {
    "Banded"             : projections.Banded,
    "Diagonal"           : projections.Diagonal,
    "DiagonallyDominant" : projections.DiagonallyDominant,
    "Hamiltonian"        : projections.Hamiltonian,
    "Identity"           : projections.Identity,
    "LowRank"            : projections.LowRank,
    "LowerTriangular"    : projections.LowerTriangular,
    "Masked"             : projections.Masked,
    "Normal"             : projections.Normal,
    "Orthogonal"         : projections.Orthogonal,
    "RankOne"            : projections.RankOne,
    "SkewSymmetric"      : projections.SkewSymmetric,
    "SpectralNorm"       : projections.SpectralNorm,
    "Symmetric"          : projections.Symmetric,
    "Symplectic"         : projections.Symplectic,
    "Traceless"          : projections.Traceless,
    "Tridiagonal"        : projections.Tridiagonal,
    "UpperTriangular"    : projections.UpperTriangular,
}  # fmt: skip
r"""Dictionary of all available modular metrics."""

PROJECTIONS = {
    **FUNCTIONAL_PROJECTIONS,
    **SPECIAL_PROJECTIONS,
    **MODULAR_PROJECTIONS,
}
r"""Dictionary containing all available projections."""
