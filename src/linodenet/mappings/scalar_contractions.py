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

import re
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
    "sigmoid": NonlinearContractionSpec(
        factory=nn.Sigmoid,
        lipschitz_bound=0.25,
        attainment=LipschitzAttainment.FINITE_SET,
        note="Derivative bounded by 1/4.",
    ),
    "hardsigmoid": NonlinearContractionSpec(
        factory=nn.Hardsigmoid,
        lipschitz_bound=1 / 6,
        attainment=LipschitzAttainment.INFINITE_SET,
        note="Piecewise linear surrogate with maximal slope 1/6.",
    ),
    "tanh": NonlinearContractionSpec(
        factory=nn.Tanh,
        lipschitz_bound=1.0,
        attainment=LipschitzAttainment.FINITE_SET,
        note="Derivative sech²(x) is at most 1.",
    ),
    "tanh-map": NonlinearContractionSpec(
        factory=TanhMap,
        lipschitz_bound=1.0,
        attainment=LipschitzAttainment.FINITE_SET,
        note="Project-local tanh bijection on (-1, 1).",
    ),
    "softsign": NonlinearContractionSpec(
        factory=nn.Softsign,
        lipschitz_bound=1.0,
        attainment=LipschitzAttainment.FINITE_SET,
        note="Derivative 1 / (1 + |x|)² is at most 1.",
    ),
    "smooth-softsign": NonlinearContractionSpec(
        factory=SmoothSoftsign,
        lipschitz_bound=1.0,
        attainment=LipschitzAttainment.FINITE_SET,
        note="Smooth project-local softsign-style bijection.",
    ),
    "softplus": NonlinearContractionSpec(
        factory=nn.Softplus,
        lipschitz_bound=1.0,
        attainment=LipschitzAttainment.NEVER,
        note="Derivative is sigmoid, hence bounded by 1.",
    ),
    "log-sigmoid": NonlinearContractionSpec(
        factory=nn.LogSigmoid,
        lipschitz_bound=1.0,
        attainment=LipschitzAttainment.NEVER,
        note="Derivative is sigmoid(-x), bounded by 1.",
    ),
    "tanhshrink": NonlinearContractionSpec(
        factory=nn.Tanhshrink,
        lipschitz_bound=1.0,
        attainment=LipschitzAttainment.NEVER,
        note="Derivative tanh²(x) is at most 1.",
    ),
}

_NON_EXPANSIVE_MAPPING_SPECS: dict[str, NonlinearContractionSpec] = {
    **_SCALAR_CONTRACTION_SPECS,
    "identity": NonlinearContractionSpec(
        factory=nn.Identity,
        lipschitz_bound=1.0,
        attainment=LipschitzAttainment.EVERYWHERE,
        note="Trivial non-expansive map.",
    ),
    "relu": NonlinearContractionSpec(
        factory=nn.ReLU,
        lipschitz_bound=1.0,
        attainment=LipschitzAttainment.INFINITE_SET,
        note="Piecewise linear with slopes in {0, 1}.",
    ),
    "relu6": NonlinearContractionSpec(
        factory=nn.ReLU6,
        lipschitz_bound=1.0,
        attainment=LipschitzAttainment.INFINITE_SET,
        note="Clipped ReLU; still piecewise linear with slopes in {0, 1}.",
    ),
    "hardtanh": NonlinearContractionSpec(
        factory=nn.Hardtanh,
        lipschitz_bound=1.0,
        attainment=LipschitzAttainment.INFINITE_SET,
        note="Saturating clamp with slope 1 on the linear region.",
    ),
    "hardshrink": NonlinearContractionSpec(
        factory=nn.Hardshrink,
        lipschitz_bound=1.0,
        attainment=LipschitzAttainment.INFINITE_SET,
        note="Piecewise linear shrinkage with slopes in {0, 1}.",
    ),
    "softshrink": NonlinearContractionSpec(
        factory=nn.Softshrink,
        lipschitz_bound=1.0,
        attainment=LipschitzAttainment.INFINITE_SET,
        note="Piecewise affine shrinkage with slopes in {0, 1}.",
    ),
    "elu": NonlinearContractionSpec(
        factory=nn.ELU,
        lipschitz_bound=1.0,
        attainment=LipschitzAttainment.INFINITE_SET,
        note="Certified for alpha <= 1 only.",
    ),
    "celu": NonlinearContractionSpec(
        factory=nn.CELU,
        lipschitz_bound=1.0,
        attainment=LipschitzAttainment.INFINITE_SET,
        note="Certified for alpha <= 1 only.",
    ),
    "leaky-relu": NonlinearContractionSpec(
        factory=nn.LeakyReLU,
        lipschitz_bound=1.0,
        attainment=LipschitzAttainment.INFINITE_SET,
        note="Certified for 0 <= negative_slope <= 1 only.",
    ),
}


def _to_kebab_case(value: str, /) -> str:
    r"""Normalize a name to lowercase kebab-case."""
    normalized = re.sub(r"[_\s]+", "-", value.strip())
    normalized = re.sub(r"([a-z0-9])([A-Z][a-z])", r"\1-\2", normalized)
    return normalized.lower()


class ScalarContraction(StrEnum):
    r"""Named scalar contractions with certified Lipschitz metadata."""

    SIGMOID = "sigmoid"
    HARDSIGMOID = "hardsigmoid"
    TANH = "tanh"
    TANH_MAP = "tanh-map"
    SOFTSIGN = "softsign"
    SMOOTH_SOFTSIGN = "smooth-softsign"
    SOFTPLUS = "softplus"
    LOG_SIGMOID = "log-sigmoid"
    TANHSHRINK = "tanhshrink"

    @classmethod
    def _missing_(cls, value: object) -> ScalarContraction | None:
        if not isinstance(value, str):
            return None
        key = _to_kebab_case(value)
        for member in cls:
            if member.value == key:
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

    SIGMOID = "sigmoid"
    HARDSIGMOID = "hardsigmoid"
    TANH = "tanh"
    TANH_MAP = "tanh-map"
    SOFTSIGN = "softsign"
    SMOOTH_SOFTSIGN = "smooth-softsign"
    SOFTPLUS = "softplus"
    LOG_SIGMOID = "log-sigmoid"
    TANHSHRINK = "tanhshrink"
    IDENTITY = "identity"
    RELU = "relu"
    RELU6 = "relu6"
    HARDTANH = "hardtanh"
    HARDSHRINK = "hardshrink"
    SOFTSHRINK = "softshrink"
    ELU = "elu"
    CELU = "celu"
    LEAKY_RELU = "leaky-relu"

    @classmethod
    def _missing_(cls, value: object) -> NonExpansiveMapping | None:
        if not isinstance(value, str):
            return None
        key = _to_kebab_case(value)
        for member in cls:
            if member.value == key:
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
