r"""Catalog of non-linear contraction mappings.

This module collects common elementwise maps with global Lipschitz constant at
most $1$. Some are strict contractions, e.g. sigmoid and hardsigmoid; many
others are only non-expansive with optimal constant exactly $1$.

The intent is primarily organizational: it provides a vetted list of
non-linear maps that can be safely composed with contractive linear layers
without increasing the global Lipschitz bound.
"""

__all__ = [
    "NonlinearContractionSpec",
    "LipschitzAttainment",
    "STRONG_CONTRACTIONS",
    "WEAK_CONTRACTIONS",
    "NON_EXPANSIVE_MAPPINGS",
    "NONLINEAR_CONTRACTIONS",
    "STRICT_NONLINEAR_CONTRACTIONS",
    "list_nonlinear_contractions",
    "get_nonlinear_contraction",
    "elu_contraction",
    "celu_contraction",
    "leaky_relu_contraction",
]

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Final

from torch import nn

from .bijections import SmoothSoftsign, TanhMap

type ModuleFactory = Callable[[], nn.Module]


class LipschitzAttainment(StrEnum):
    r"""Qualitative description of how the optimal Lipschitz bound is attained."""

    NEVER = "never"
    FINITE_SET = "finite_set"
    INFINITE_SET = "infinite_set"
    EVERYWHERE = "everywhere"


@dataclass(frozen=True, slots=True)
class NonlinearContractionSpec:
    r"""Metadata for a non-linear contraction map.

    Attributes:
        factory: Zero-argument constructor returning the module instance.
        lipschitz_bound: A certified global Lipschitz bound.
        attainment: Qualitative description of where the optimal bound is attained.
        note: Short qualification about the bound or intended usage.
    """

    factory: ModuleFactory
    lipschitz_bound: float
    attainment: LipschitzAttainment
    note: str


def elu_contraction(*, alpha: float = 1.0) -> nn.ELU:
    r"""Return an ELU activation with certified Lipschitz constant at most $1$."""
    if not 0 < alpha <= 1:
        raise ValueError(f"Expected 0 < alpha <= 1, got {alpha=}.")
    return nn.ELU(alpha=alpha)


def celu_contraction(*, alpha: float = 1.0) -> nn.CELU:
    r"""Return a CELU activation with certified Lipschitz constant at most $1$."""
    if not 0 < alpha <= 1:
        raise ValueError(f"Expected 0 < alpha <= 1, got {alpha=}.")
    return nn.CELU(alpha=alpha)


def leaky_relu_contraction(*, negative_slope: float = 1e-2) -> nn.LeakyReLU:
    r"""Return a LeakyReLU activation with certified Lipschitz constant at most $1$."""
    if not 0 <= negative_slope <= 1:
        raise ValueError(f"Expected 0 <= negative_slope <= 1, got {negative_slope=}.")
    return nn.LeakyReLU(negative_slope=negative_slope)


STRONG_CONTRACTIONS: Final[dict[str, NonlinearContractionSpec]] = {
    "Sigmoid": NonlinearContractionSpec(
        factory=nn.Sigmoid,
        lipschitz_bound=0.25,
        attainment=LipschitzAttainment.FINITE_SET,
        note="Derivative bounded by 1/4.",
    ),
    "Hardsigmoid": NonlinearContractionSpec(
        factory=nn.Hardsigmoid,
        lipschitz_bound=1 / 6,
        attainment=LipschitzAttainment.INFINITE_SET,
        note="Piecewise linear surrogate with maximal slope 1/6.",
    ),
}
r"""Maps with certified global Lipschitz bound strictly below $1$."""

WEAK_CONTRACTIONS: Final[dict[str, NonlinearContractionSpec]] = {
    "Tanh": NonlinearContractionSpec(
        factory=nn.Tanh,
        lipschitz_bound=1.0,
        attainment=LipschitzAttainment.FINITE_SET,
        note="Derivative sech²(x) is at most 1.",
    ),
    "TanhMap": NonlinearContractionSpec(
        factory=TanhMap,
        lipschitz_bound=1.0,
        attainment=LipschitzAttainment.FINITE_SET,
        note="Project-local tanh bijection on (-1, 1).",
    ),
    "Softsign": NonlinearContractionSpec(
        factory=nn.Softsign,
        lipschitz_bound=1.0,
        attainment=LipschitzAttainment.FINITE_SET,
        note="Derivative 1 / (1 + |x|)² is at most 1.",
    ),
    "SmoothSoftsign": NonlinearContractionSpec(
        factory=SmoothSoftsign,
        lipschitz_bound=1.0,
        attainment=LipschitzAttainment.FINITE_SET,
        note="Smooth project-local softsign-style bijection.",
    ),
    "Softplus": NonlinearContractionSpec(
        factory=nn.Softplus,
        lipschitz_bound=1.0,
        attainment=LipschitzAttainment.NEVER,
        note="Derivative is sigmoid, hence bounded by 1.",
    ),
    "LogSigmoid": NonlinearContractionSpec(
        factory=nn.LogSigmoid,
        lipschitz_bound=1.0,
        attainment=LipschitzAttainment.NEVER,
        note="Derivative is sigmoid(-x), bounded by 1.",
    ),
    "Tanhshrink": NonlinearContractionSpec(
        factory=nn.Tanhshrink,
        lipschitz_bound=1.0,
        attainment=LipschitzAttainment.NEVER,
        note="Derivative tanh²(x) is at most 1.",
    ),
}
r"""Non-expansive maps whose unit bound is attained never or only on a finite set."""

NON_EXPANSIVE_MAPPINGS: Final[dict[str, NonlinearContractionSpec]] = {
    "Identity": NonlinearContractionSpec(
        factory=nn.Identity,
        lipschitz_bound=1.0,
        attainment=LipschitzAttainment.EVERYWHERE,
        note="Trivial non-expansive map.",
    ),
    "ReLU": NonlinearContractionSpec(
        factory=nn.ReLU,
        lipschitz_bound=1.0,
        attainment=LipschitzAttainment.INFINITE_SET,
        note="Piecewise linear with slopes in {0, 1}.",
    ),
    "ReLU6": NonlinearContractionSpec(
        factory=nn.ReLU6,
        lipschitz_bound=1.0,
        attainment=LipschitzAttainment.INFINITE_SET,
        note="Clipped ReLU; still piecewise linear with slopes in {0, 1}.",
    ),
    "Hardtanh": NonlinearContractionSpec(
        factory=nn.Hardtanh,
        lipschitz_bound=1.0,
        attainment=LipschitzAttainment.INFINITE_SET,
        note="Saturating clamp with slope 1 on the linear region.",
    ),
    "Hardshrink": NonlinearContractionSpec(
        factory=nn.Hardshrink,
        lipschitz_bound=1.0,
        attainment=LipschitzAttainment.INFINITE_SET,
        note="Piecewise linear shrinkage with slopes in {0, 1}.",
    ),
    "Softshrink": NonlinearContractionSpec(
        factory=nn.Softshrink,
        lipschitz_bound=1.0,
        attainment=LipschitzAttainment.INFINITE_SET,
        note="Piecewise affine shrinkage with slopes in {0, 1}.",
    ),
    "ELU": NonlinearContractionSpec(
        factory=elu_contraction,
        lipschitz_bound=1.0,
        attainment=LipschitzAttainment.INFINITE_SET,
        note="Certified for alpha <= 1 only.",
    ),
    "CELU": NonlinearContractionSpec(
        factory=celu_contraction,
        lipschitz_bound=1.0,
        attainment=LipschitzAttainment.INFINITE_SET,
        note="Certified for alpha <= 1 only.",
    ),
    "LeakyReLU": NonlinearContractionSpec(
        factory=leaky_relu_contraction,
        lipschitz_bound=1.0,
        attainment=LipschitzAttainment.INFINITE_SET,
        note="Certified for 0 <= negative_slope <= 1 only.",
    ),
}
r"""Non-expansive maps whose unit bound is attained on an infinite set."""

assert len(
    _combined := (
        list(STRONG_CONTRACTIONS)
        + list(WEAK_CONTRACTIONS)
        + list(NON_EXPANSIVE_MAPPINGS)
    )
) == len(set(_combined)), "duplicate names across contraction groups"

NONLINEAR_CONTRACTIONS: Final[dict[str, NonlinearContractionSpec]] = {
    **STRONG_CONTRACTIONS,
    **WEAK_CONTRACTIONS,
    **NON_EXPANSIVE_MAPPINGS,
}
r"""Known non-linear contractions and non-expansive activations."""

STRICT_NONLINEAR_CONTRACTIONS: Final[dict[str, NonlinearContractionSpec]] = {
    name: spec
    for name, spec in NONLINEAR_CONTRACTIONS.items()
    if spec.lipschitz_bound < 1.0
}
r"""Subset of maps with certified global Lipschitz bound strictly below $1$."""


def list_nonlinear_contractions(*, strict: bool | None = None) -> tuple[str, ...]:
    r"""Return the available contraction names.

    Args:
        strict: If `True`, only return strict contractions. If `False`, only
            return the non-strict entries. If `None`, return all entries.
    """
    match strict:
        case True:
            source = STRICT_NONLINEAR_CONTRACTIONS
        case False:
            source = {
                name: spec
                for name, spec in NONLINEAR_CONTRACTIONS.items()
                if spec.lipschitz_bound >= 1.0
            }
        case None:
            source = NONLINEAR_CONTRACTIONS
    return tuple(source)


def get_nonlinear_contraction(name: str, /) -> nn.Module:
    r"""Instantiate a named non-linear contraction."""
    try:
        spec = NONLINEAR_CONTRACTIONS[name]
    except KeyError as err:
        msg = f"Unknown non-linear contraction: {name!r}."
        raise KeyError(msg) from err
    return spec.factory()
