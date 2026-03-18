r"""Projections for the Linear ODE Networks.

Notes:
    - See `linodenet.mappings.functional` for functional implementations.
    - See `linodenet.mappings.projections` for module-based implementations.
"""
# ruff: noqa: F403, F405

__all__ = [
    # Sub-Modules
    "base",
    "embeddings",
    "projections",
    "surjections",
    "functional",
    "bijections",
    "linear",
    "transforms",
    # Constants
    "PROJECTION_FNS",
    "PROJECTION_MODULES",
    "SPECIAL_PROJECTIONS",
    "PROJECTIONS",
    "EMBEDDINGS",
    "SURJECTIONS",
    "TRANSFORMS",
    "BIJECTIONS",
]

from linodenet.mappings import (
    base,
    bijections,
    embeddings,
    functional,
    linear,
    projections,
    surjections,
    transforms,
)
from linodenet.mappings.base import *
from linodenet.mappings.bijections import *
from linodenet.mappings.embeddings import *
from linodenet.mappings.functional import *
from linodenet.mappings.projections import *
from linodenet.mappings.surjections import *
from linodenet.mappings.transforms import *

assert len(
    _combined := (
        base.__all__
        + embeddings.__all__
        + surjections.__all__
        + projections.__all__
        + functional.__all__
        + bijections.__all__
        + transforms.__all__
    )
) == len(set(_combined)), "duplicate names in __all__"

__all__ += base.__all__
__all__ += embeddings.__all__
__all__ += surjections.__all__
__all__ += projections.__all__
__all__ += functional.__all__
__all__ += bijections.__all__
__all__ += transforms.__all__


EMBEDDINGS: dict[str, type[EmbeddingBase]] = {
    "ConcatEmbedding"  : embeddings.ConcatEmbedding,
    "LinearEmbedding"  : embeddings.LinearEmbedding,
}  # fmt: skip
r"""Dictionary of available embeddings."""

SURJECTIONS: dict[str, type[SurjectionBase]] = {
    "ConcatProjection": surjections.ConcatProjection,
    "GramMatrix"      : surjections.GramMatrix,
}  # fmt: skip
r"""Dictionary containing all available surjections."""

BIJECTIONS: dict[str, type[BijectionBase]] = {
    "MatrixExponential" : bijections.MatrixExponential,
    "CayleyMap"         : bijections.CayleyMap,
}  # fmt: skip
r"""Dictionary containing all available bijections."""

TRANSFORMS: dict[str, type[Transform]] = {
    "ContractiveTransform" : transforms.ContractiveTransform,
    "SplineTransform"      : transforms.SplineTransform,
    "LowRankTransform"     : transforms.LowRankTransform,
    "TriangularTransform"  : transforms.TriangularTransform,
}  # fmt: skip
r"""Dictionary containing all available bijections."""

PROJECTION_FNS: dict[str, FunctionalProjection] = {
    "diagonal"            : functional.diagonal,
    "diagonally_dominant" : functional.diagonally_dominant,
    "hamiltonian"         : functional.hamiltonian,
    "identity"            : functional.identity,
    "lower_triangular"    : functional.lower_triangular,
    "normal"              : functional.normal,
    "orthogonal"          : functional.orthogonal,
    "rank_one"            : functional.rank_one,
    "skew_symmetric"      : functional.skew_symmetric,
    "spectral_normalized" : functional.spectral_normalized,
    "symmetric"           : functional.symmetric,
    "symplectic"          : functional.symplectic,
    "traceless"           : functional.traceless,
    "tridiagonal"         : functional.tridiagonal,
    "upper_triangular"    : functional.upper_triangular,
}  # fmt: skip
r"""Dictionary of all available modular metrics."""

SPECIAL_PROJECTIONS = {
    "banded"            : functional.banded,
    "low_rank"          : functional.low_rank,
    "masked"            : functional.masked,
    "contraction"       : functional.contraction,
    "lipschitz_bounded" : functional.lipschitz_bounded,
}  # fmt: skip
r"""Projections that require additional arguments"""

PROJECTION_MODULES: dict[str, type[ProjectionBase]] = {
    "Banded"             : projections.Banded,
    "Contraction"        : projections.Contraction,
    "Diagonal"           : projections.Diagonal,
    "DiagonallyDominant" : projections.DiagonallyDominant,
    "Hamiltonian"        : projections.Hamiltonian,
    "Identity"           : projections.Identity,
    "LipschitzBounded"   : projections.LipschitzBounded,
    "LowRank"            : projections.LowRank,
    "LowerTriangular"    : projections.LowerTriangular,
    "Masked"             : projections.Masked,
    "Normal"             : projections.Normal,
    "Orthogonal"         : projections.Orthogonal,
    "RankOne"            : projections.RankOne,
    "SkewSymmetric"      : projections.SkewSymmetric,
    "SpectralNormalized" : projections.SpectralNormalized,
    "Symmetric"          : projections.Symmetric,
    "Symplectic"         : projections.Symplectic,
    "Traceless"          : projections.Traceless,
    "Tridiagonal"        : projections.Tridiagonal,
    "UpperTriangular"    : projections.UpperTriangular,
}  # fmt: skip
r"""Dictionary of all available modular metrics."""

PROJECTIONS = {
    **PROJECTION_FNS,
    **SPECIAL_PROJECTIONS,
    **PROJECTION_MODULES,
}
r"""Dictionary containing all available projections."""
