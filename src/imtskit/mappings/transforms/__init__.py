r"""Diffeomorphisms: bijections with differentiable inverses.

Transform modules provide `encode` and `decode` methods satisfying
`decode(encode(x)) = x` and `encode(decode(y)) = y`. Both directions are
differentiable, and `encode_and_logabsdet` and `decode_and_logabsdet` return
the transformed values together with their log absolute Jacobian determinants.

Examples:
    - Affine maps: $y=Ax+b$ and $x=A⁻¹(y-b)$ for invertible $A$, including
        diagonal and triangular matrices with nonzero diagonal entries.
    - Elementwise diffeomorphisms with nonzero derivatives.
    - Coupling flows: $yₐ=xₐ+\exp(xᵦ)$ and $yᵦ=xᵦ$.
    - Residual maps: $y=x+F(x)$ with a contractive residual $F$.
    - Low-rank perturbations: $y=(𝕀+AB)x$ when $𝕀+AB$ is invertible.
    - Continuous-time flows: $ẋ=f(t,x)$ under conditions ensuring a
        differentiable flow and inverse.

Note:
    `torch.distributions.Transform.log_abs_det_jacobian` takes both the input
    and output. This can avoid recomputing intermediate values: for $f(x)=xᵃ$
    on $x>0$ with $a≠0$, $\log|f′(x)|=\log|a⋅y/x|$.

    Here, the combined value-and-log-determinant methods compute both results
    in one call. For example:

    ```pycon
    >>> from imtskit.mappings.transforms import TriangularTransform
    >>> import torch
    >>> transform = TriangularTransform(3)
    >>> x = torch.randn(2, 3)
    >>> y, forward_logabsdet = transform.encode_and_logabsdet(x)
    >>> recovered, inverse_logabsdet = transform.decode_and_logabsdet(y)
    >>> assert torch.allclose(x, recovered)

    ```
"""
# ruff: file-ignore[F403]

__all__ = [
    # submodules
    "scalar",
    # ABCs and Protocols
    # classes
    "BimodalToGaussian",
    "BottleneckFlow",
    "GaussianToBimodal",
    "GaussianToMixture",
    "IResNet",
    "LowRankTransform",
    "MixtureToGaussian",
    "ResidualBottleneck",
    "ResidualContraction",
    "ResidualContractionFallback",
    "SplineTransform",
    "SPDLowRankTransform",
    "TriangularTransform",
]

from . import scalar
from .bottleneck_flow import BottleneckFlow
from .gaussian_transport import (
    BimodalToGaussian,
    GaussianToBimodal,
    GaussianToMixture,
    MixtureToGaussian,
)
from .iresnet import IResNet
from .linear_rational_spline import SplineTransform
from .residual_contraction import (
    ResidualBottleneck,
    ResidualContraction,
    ResidualContractionFallback,
)
from .residual_low_rank import LowRankTransform, SPDLowRankTransform
from .scalar import *
from .triangular import TriangularTransform

__all__ += scalar.__all__
