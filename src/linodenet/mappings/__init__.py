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
    "BIJECTIONS",
    "EMBEDDINGS",
    "PROJECTIONS",
    "SURJECTIONS",
    "TRANSFORMS",
    "MATRIX_PROJECTIONS",
    "VECTOR_PROJECTIONS",
    # functional
    "MATRIX_PROJECTION_FNS",
    "MATRIX_PROJECTIONS_WITH_ARGS",
    "VECTOR_PROJECTION_FNS",
    "PROJECTION_FNS",
    # other
    "LinearContraction",
    "RankOneContraction",
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
from linodenet.mappings.linear import LinearContraction, RankOneContraction
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


BIJECTIONS: dict[str, type[BijectionBase]] = {
    "MatrixExponential" : bijections.MatrixExponential,
    "CayleyMap"         : bijections.CayleyMap,
    "SmoothSoftsign"    : bijections.SmoothSoftsign,
    "TanhMap"           : bijections.TanhMap,
}  # fmt: skip
r"""Dictionary containing all available bijections (nn.Module)."""

EMBEDDINGS: dict[str, type[EmbeddingBase]] = {
    "ConcatEmbedding"  : embeddings.ConcatEmbedding,
    "LinearEmbedding"  : embeddings.LinearEmbedding,
}  # fmt: skip
r"""Dictionary of available embeddings (nn.Module)."""

TRANSFORMS: dict[str, type[Transform]] = {
    "ContractiveFP"        : transforms.ResidualContraction,
    "ReZeroContraction"    : transforms.ReZeroContraction,
    "ContractiveTransform" : transforms.ResidualContractionFallback,
    "SplineTransform"      : transforms.SplineTransform,
    "LowRankTransform"     : transforms.LowRankTransform,
    "TriangularTransform"  : transforms.TriangularTransform,
}  # fmt: skip
r"""Dictionary containing all available transforms (nn.Module)."""

VECTOR_PROJECTIONS: dict[str, type[ProjectionBase]] = {
    "UnitVector" : projections.UnitVector,
}  # fmt: skip
r"""Dictionary containing all available vector projections (nn.Module)."""

MATRIX_PROJECTIONS: dict[str, type[ProjectionBase]] = {
    "Banded"             : projections.Banded,
    "Contraction"        : projections.Contraction,
    "Diagonal"           : projections.Diagonal,
    "DiagonallyDominant" : projections.DiagonallyDominant,
    "Hamiltonian"        : projections.Hamiltonian,
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
r"""Dictionary containing all available matrix projections (nn.Module)."""

PROJECTIONS: dict[str, type[ProjectionBase]] = {
    **MATRIX_PROJECTIONS,
    **VECTOR_PROJECTIONS,
}
r"""Dictionary containing all available projections."""

SURJECTIONS: dict[str, type[SurjectionBase]] = {
    **PROJECTIONS,
    "ConcatProjection"     : surjections.ConcatProjection,
    "NegativeDefinite"     : surjections.NegativeDefinite,
    "OrthogonalCayley"     : surjections.OrthogonalCayley,
    "OrthogonalHouseholder": surjections.OrthogonalHouseholder,
    "PositiveDefinite"     : surjections.PositiveDefinite,
    "PositiveSemiDefinite" : surjections.PositiveSemiDefinite,
    "PositiveVector"       : surjections.PositiveVector,
    "SpecialOrthogonal"    : surjections.SpecialOrthogonal,
    "StochasticVector"     : surjections.StochasticVector,
}  # fmt: skip
r"""Dictionary containing all available surjections (nn.Module)."""

VECTOR_PROJECTION_FNS: dict[str, ProjectionFn] = {
    "unit_vector" : unit_vector
}  # fmt: skip
r"""Dictionary containing all available vector projections (function)."""

MATRIX_PROJECTION_FNS: dict[str, ProjectionFn] = {
    "diagonal"              : functional.diagonal,
    "diagonally_dominant"   : functional.diagonally_dominant,
    "hamiltonian"           : functional.hamiltonian,
    "lower_triangular"      : functional.lower_triangular,
    "normal"                : functional.normal,
    "orthogonal"            : functional.orthogonal,
    "rank_one"              : functional.rank_one,
    "skew_symmetric"        : functional.skew_symmetric,
    "spectral_normalized"   : functional.spectral_normalized,
    "symmetric"             : functional.symmetric,
    "symplectic"            : functional.symplectic,
    "traceless"             : functional.traceless,
    "tridiagonal"           : functional.tridiagonal,
    "upper_triangular"      : functional.upper_triangular,
}  # fmt: skip
r"""Dictionary containing all available matrix projections (function)."""

MATRIX_PROJECTIONS_WITH_ARGS: dict[str, ProjectionFnWithArgs] = {
    "banded"            : functional.banded,
    "low_rank"          : functional.low_rank,
    "masked"            : functional.masked,
    "contraction"       : functional.contraction,
    "lipschitz_bounded" : functional.lipschitz_bounded,
}  # fmt: skip
r"""Matrix projections that require additional arguments"""

PROJECTION_FNS: dict[str, ProjectionFnWithArgs] = {
    **VECTOR_PROJECTION_FNS,
    **MATRIX_PROJECTION_FNS,
    **MATRIX_PROJECTIONS_WITH_ARGS,
}  # fmt: skip
r"""Dictionary containing all available projections (function)."""
