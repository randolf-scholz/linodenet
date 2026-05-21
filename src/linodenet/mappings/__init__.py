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
]

from . import (
    abstract,
    base,
    bijections,
    embeddings,
    functional,
    linear,
    projections,
    surjections,
    transforms,
)
from .abstract import *
from .base import *
from .bijections import *
from .embeddings import *
from .functional import *
from .linear import *
from .projections import *
from .surjections import *
from .transforms import *
from .transforms import linear_rational_spline

assert len(
    _combined := (
        abstract.__all__
        + base.__all__
        + bijections.__all__
        + embeddings.__all__
        + functional.__all__
        + linear.__all__
        + projections.__all__
        + surjections.__all__
        + transforms.__all__
    )
) == len(set(_combined)), "duplicate names in __all__"

__all__ += abstract.__all__
__all__ += base.__all__
__all__ += bijections.__all__
__all__ += embeddings.__all__
__all__ += functional.__all__
__all__ += linear.__all__
__all__ += projections.__all__
__all__ += surjections.__all__
__all__ += transforms.__all__

BIJECTIONS: dict[str, type[BijectionBase]] = {
    "MatrixExponential" : bijections.MatrixExponential,
    "PositiveScalarMatrix": bijections.PositiveScalarMatrix,
    "PositiveDiagonal"  : bijections.PositiveDiagonal,
    "CayleyMap"         : bijections.CayleyMap,
    "SmoothSoftsign"    : transforms.SmoothSoftsign,
    "Tanh"              : transforms.Tanh,
}  # fmt: skip
r"""Dictionary containing all available bijections (nn.Module)."""

EMBEDDINGS: dict[str, type[EmbeddingBase]] = {
    "ConcatEmbedding"  : embeddings.ConcatEmbedding,
    "LinearEmbedding"  : embeddings.LinearEmbedding,
}  # fmt: skip
r"""Dictionary of available embeddings (nn.Module)."""

TRANSFORMS: dict[str, type[TransformBase]] = {
    "BimodalToGaussian"    : transforms.BimodalToGaussian,
    "BottleneckFlow"       : transforms.BottleneckFlow,
    "GaussianToBimodal"    : transforms.GaussianToBimodal,
    "GaussianToMixture"    : transforms.GaussianToMixture,
    "InverseTransform"     : base.InverseTransform,
    "IResNet"              : transforms.IResNet,
    "LowRankTransform"     : transforms.LowRankTransform,
    "MixtureToGaussian"    : transforms.MixtureToGaussian,
    "ResidualBottleneck"   : transforms.ResidualBottleneck,
    "ResidualContraction"  : transforms.ResidualContraction,
    "ResidualContractionFallback" : transforms.ResidualContractionFallback,
    "SplineTransform"      : linear_rational_spline.SplineTransform,
    "TransformSequence"    : base.TransformSequence,
    "TriangularTransform"  : transforms.TriangularTransform,
    # scalar transforms
    "CELU"                 : transforms.scalar.CELU,
    "ELU"                  : transforms.scalar.ELU,
    "EntLU"                : transforms.scalar.EntLU,
    "Sigmoid"              : transforms.scalar.Sigmoid,
    "SmoothSoftsign"       : transforms.scalar.SmoothSoftsign,
    "Softplus"             : transforms.scalar.Softplus,
    "Softsign"             : transforms.scalar.Softsign,
    "Tanh"                 : transforms.scalar.Tanh,
    "Tanhshrink"           : transforms.scalar.Tanhshrink,
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
    "CholeskyFactor"      : surjections.CholeskyFactor,
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
    **MATRIX_PROJECTION_FNS,
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
