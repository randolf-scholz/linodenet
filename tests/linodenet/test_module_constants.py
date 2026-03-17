r"""Checks the internal consistency of the module constants."""

from collections.abc import Callable, Mapping
from inspect import isabstract
from types import ModuleType
from typing import NamedTuple, is_protocol

import pytest

import linodenet as lib
from linodenet.distributions import DISTRIBUTIONS, Distribution, DistributionBase
from linodenet.flows import FLOWS, Flow, FlowBase
from linodenet.flows.transforms import BIJECTIONS
from linodenet.imputation import IMPUTERS, ImputerProtocol
from linodenet.initializations import INITIALIZATIONS, Initialization
from linodenet.mappings import (
    EMBEDDINGS,
    FUNCTIONAL_PROJECTIONS,
    MODULAR_PROJECTIONS,
    SURJECTIONS,
    Embedding,
    EmbeddingBase,
    Projection,
    ProjectionBase,
    Surjection,
    SurjectionBase,
)
from linodenet.mappings.base import Bijection, BijectionBase
from linodenet.nn.activations import ALL_ACTIVATIONS, Activation
from linodenet.parametrize import PARAMETRIZATIONS, Parametrization, ParametrizationBase
from linodenet.regularizations import (
    FUNCTIONAL_REGULARIZATIONS,
    MODULAR_REGULARIZATIONS,
    Regularization,
    RegularizationBase,
)
from linodenet.testing import MATRIX_TESTS, MatrixTest


class Case(NamedTuple):
    r"""NamedTuple for each case."""

    module: ModuleType
    protocol: type | None
    base_class: type | None
    elements: Mapping[str, type] | Mapping[str, Callable]


CASES: dict[str, Case] = {
    "activations"         : Case(lib.nn.activations , Activation     , None               , ALL_ACTIVATIONS           ),
    # "bijections"          : Case(lib.bijections     , Bijection      , BijectionBase      , BIJECTIONS                ),
    "distributions"       : Case(lib.distributions  , Distribution   , DistributionBase   , DISTRIBUTIONS             ),
    "embeddings"          : Case(lib.nn.embeddings  , Embedding      , EmbeddingBase      , EMBEDDINGS                ),
    "surjections"         : Case(lib.nn.surjections , Surjection     , SurjectionBase     , SURJECTIONS               ),
    "flows"               : Case(lib.flows          , Flow           , FlowBase           , FLOWS                     ),
    "imputation"          : Case(lib.imputation     , ImputerProtocol, None               , IMPUTERS                  ),
    "initializations"     : Case(lib.initializations, Initialization , None               , INITIALIZATIONS           ),
    "matrix_tests"        : Case(lib.testing        , MatrixTest     , None               , MATRIX_TESTS              ),
    "parametrizations"    : Case(lib.parametrize    , Parametrization, ParametrizationBase, PARAMETRIZATIONS          ),
    "projections_cls"     : Case(lib.projections    , Projection     , ProjectionBase     , MODULAR_PROJECTIONS       ),
    "projections_fun"     : Case(lib.projections    , Projection     , None               , FUNCTIONAL_PROJECTIONS    ),
    "regularizations_cls" : Case(lib.regularizations, Regularization , RegularizationBase , MODULAR_REGULARIZATIONS   ),
    "regularizations_fun" : Case(lib.regularizations, Regularization , None               , FUNCTIONAL_REGULARIZATIONS),
}  # fmt: skip
r"""Dictionary of all available cases."""


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    # SEE: https://stackoverflow.com/q/40818146
    # FIXME: https://github.com/pytest-dev/pytest/issues/349
    # FIXME: https://github.com/pytest-dev/pytest/issues/4050
    if "case_name" in metafunc.fixturenames:
        if "item_name" in metafunc.fixturenames:
            metafunc.parametrize(
                ["case_name", "item_name"],
                [
                    (name, element)
                    for name, case in CASES.items()
                    for element in case.elements
                ],
            )
        else:
            metafunc.parametrize("case_name", CASES)


def test_name_casing(case_name: str, item_name: str) -> None:
    r"""Check that the case name is in snake_case."""
    case = CASES[case_name]
    item = case.elements[item_name]
    assert item_name.isidentifier()

    if isinstance(item, type) or case_name == "parametrizations":
        # check if the name is in CamelCase
        assert item_name[0].isupper()
        assert "_" not in item_name
    else:
        # check if the name is in snake_case
        assert item_name.islower()


def test_protocol(case_name: str) -> None:
    r"""Check that the protocol associated with the classes is indeed a protocol."""
    case = CASES[case_name]
    assert case.protocol is None or is_protocol(case.protocol)


def test_base_class(case_name: str) -> None:
    r"""Check that the base class is a class and not a protocol."""
    case = CASES[case_name]
    cls = case.base_class

    if cls is not None:
        assert isinstance(cls, type)
        assert not is_protocol(cls)


def test_issubclass(case_name: str, item_name: str) -> None:
    r"""Check if the class is a subclass of the correct base class."""
    case = CASES[case_name]
    obj = case.elements[item_name]

    if case.base_class is not None:
        assert isinstance(obj, type)
        assert issubclass(obj, case.base_class)


def test_name(case_name: str, item_name: str) -> None:
    r"""Check if the name of the class matches the item name."""
    case = CASES[case_name]
    obj = case.elements[item_name]
    # fallback for jit.ScriptFunction
    name = getattr(obj, "__name__", getattr(obj, "name", None))
    assert name == item_name


def test_dict_complete(case_name: str) -> None:
    r"""Check if all encoders are in the ENCODERS constant."""
    case = CASES[case_name]

    match case.base_class:
        case None:  # skip for function-dicts
            return
        case base_class:
            missing: dict[str, type] = {
                name: cls
                for name, cls in vars(case.module).items()
                if (
                    not is_protocol(cls)
                    and not isabstract(cls)
                    and isinstance(cls, type)
                    and issubclass(cls, base_class)
                    and cls is not base_class
                    # and __init__ is compatible with base_class.__init__
                )
                and cls not in case.elements.values()
            }

            if missing:
                raise AssertionError(f"Missing {case_name}: {sorted(missing)}")
