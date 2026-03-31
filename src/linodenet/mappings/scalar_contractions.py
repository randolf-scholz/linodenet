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
    # enums
    "ScalarContraction",
    "NonExpansiveMapping",
]

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum

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


_SCALAR_CONTRACTION_SPECS: dict[str, NonlinearContractionSpec] = {
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

_NON_EXPANSIVE_MAPPING_SPECS: dict[str, NonlinearContractionSpec] = {
    **_SCALAR_CONTRACTION_SPECS,
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
        factory=nn.ELU,
        lipschitz_bound=1.0,
        attainment=LipschitzAttainment.INFINITE_SET,
        note="Certified for alpha <= 1 only.",
    ),
    "CELU": NonlinearContractionSpec(
        factory=nn.CELU,
        lipschitz_bound=1.0,
        attainment=LipschitzAttainment.INFINITE_SET,
        note="Certified for alpha <= 1 only.",
    ),
    "LeakyReLU": NonlinearContractionSpec(
        factory=nn.LeakyReLU,
        lipschitz_bound=1.0,
        attainment=LipschitzAttainment.INFINITE_SET,
        note="Certified for 0 <= negative_slope <= 1 only.",
    ),
}


class ScalarContraction(StrEnum):
    r"""Named scalar contractions with certified Lipschitz metadata."""

    SIGMOID = "Sigmoid"
    HARDSIGMOID = "Hardsigmoid"
    TANH = "Tanh"
    TANH_MAP = "TanhMap"
    SOFTSIGN = "Softsign"
    SMOOTH_SOFTSIGN = "SmoothSoftsign"
    SOFTPLUS = "Softplus"
    LOG_SIGMOID = "LogSigmoid"
    TANHSHRINK = "Tanhshrink"

    @classmethod
    def _missing_(cls, value: object) -> ScalarContraction | None:
        if not isinstance(value, str):
            return None
        key = value.casefold()
        for member in cls:
            if member.value.casefold() == key:
                return member
        return None

    @property
    def spec(self) -> NonlinearContractionSpec:
        r"""Return the certified metadata for this scalar contraction."""
        return _SCALAR_CONTRACTION_SPECS[self.value]

    @classmethod
    def new(cls, name: str | nn.Module, /) -> nn.Module:
        r"""Instantiate a named scalar contraction or pass through a module."""
        match name:
            case nn.Module() as module:
                return module
            case str():
                try:
                    return cls(name).spec.factory()
                except ValueError as err:
                    msg = f"Unknown scalar contraction: {name!r}."
                    raise KeyError(msg) from err
            case _:
                raise TypeError(f"Expected str or nn.Module, got {type(name)}")


class NonExpansiveMapping(StrEnum):
    r"""Named non-expansive mappings with certified Lipschitz metadata."""

    SIGMOID = "Sigmoid"
    HARDSIGMOID = "Hardsigmoid"
    TANH = "Tanh"
    TANH_MAP = "TanhMap"
    SOFTSIGN = "Softsign"
    SMOOTH_SOFTSIGN = "SmoothSoftsign"
    SOFTPLUS = "Softplus"
    LOG_SIGMOID = "LogSigmoid"
    TANHSHRINK = "Tanhshrink"
    IDENTITY = "Identity"
    RELU = "ReLU"
    RELU6 = "ReLU6"
    HARDTANH = "Hardtanh"
    HARDSHRINK = "Hardshrink"
    SOFTSHRINK = "Softshrink"
    ELU = "ELU"
    CELU = "CELU"
    LEAKY_RELU = "LeakyReLU"

    @classmethod
    def _missing_(cls, value: object) -> NonExpansiveMapping | None:
        if not isinstance(value, str):
            return None
        key = value.casefold()
        for member in cls:
            if member.value.casefold() == key:
                return member
        return None

    @property
    def spec(self) -> NonlinearContractionSpec:
        r"""Return the certified metadata for this non-expansive mapping."""
        return _NON_EXPANSIVE_MAPPING_SPECS[self.value]

    @classmethod
    def new(cls, name: str | nn.Module, /) -> nn.Module:
        r"""Instantiate a named non-expansive mapping or pass through a module."""
        match name:
            case nn.Module() as module:
                return module
            case str():
                try:
                    return cls(name).spec.factory()
                except ValueError as err:
                    msg = f"Unknown non-expansive mapping: {name!r}."
                    raise KeyError(msg) from err
            case _:
                raise TypeError(f"Expected str or nn.Module, got {type(name)}")


assert all(
    name in NonExpansiveMapping._value2member_map_
    for name in ScalarContraction._value2member_map_
), "NonExpansiveMapping must contain every ScalarContraction value."
