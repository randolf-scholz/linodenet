r"""Checks the internal consistency of the module constants."""

from collections.abc import Callable, Mapping
from inspect import isabstract
from types import ModuleType
from typing import NamedTuple, is_protocol

import pytest
from torch import nn

import linodenet as lib
from linodenet.distributions import DISTRIBUTIONS, Distribution, DistributionBase
from linodenet.initializations import INITIALIZATION_FNS
from linodenet.initializations.base import Initialization
from linodenet.mappings import (
    BIJECTIONS,
    EMBEDDINGS,
    PROJECTION_FNS,
    PROJECTIONS,
    SURJECTIONS,
    TRANSFORMS,
)
from linodenet.mappings.abstract import (
    Bijection,
    Embedding,
    Projection,
    Surjection,
    Transform,
)
from linodenet.nn.activations import ACTIVATIONS, Activation
from linodenet.regularizations import (
    REGULARIZATION_FNS,
    REGULARIZATIONS,
    Regularization,
    RegularizationBase,
)
from linodenet.state_propagation.flows import FLOWS, Flow, FlowBase
from linodenet.state_update.imputation import IMPUTERS, ImputerProtocol


class Case[StrT: str](NamedTuple):
    r"""NamedTuple for each case."""

    module: ModuleType
    protocol: type | None
    base_class: type | None
    elements: Mapping[StrT, type] | Mapping[StrT, Callable]
    excluded: set = set()


CASES: dict[str, Case] = {
    "activations"         : Case(lib.nn.activations   , Activation     , None               , ACTIVATIONS        ),
    "transforms"          : Case(lib.transforms       , Transform      , nn.Module          , TRANSFORMS         ),
    "bijections"          : Case(lib.bijections       , Bijection      , nn.Module          , BIJECTIONS         ),
    "distributions"       : Case(lib.distributions    , Distribution   , DistributionBase   , DISTRIBUTIONS      ),
    "embeddings"          : Case(lib.embeddings       , Embedding      , nn.Module          , EMBEDDINGS         ),
    "surjections"         : Case(lib.surjections      , Surjection     , nn.Module          , SURJECTIONS        ),
    "flows"               : Case(lib.flows            , Flow           , FlowBase           , FLOWS              ),
    "imputation"          : Case(lib.imputation       , ImputerProtocol, None               , IMPUTERS           ),
    "initializations"     : Case(lib.initializations, Initialization, None, INITIALIZATION_FNS),
    # "matrix_domain_tests" : Case(lib.testing          , None           , None               , MATRIX_TESTS       ),
    # "parametrizations"    : Case(lib.parametrizations , Parametrization, ParametrizationBase, PARAMETRIZATIONS   ),
    "projections_cls"     : Case(lib.projections      , Projection     , nn.Module     , PROJECTIONS        ),
    "projections_fun"     : Case(lib.projections      , Projection     , None               , PROJECTION_FNS     ),
    "regularizations_cls" : Case(lib.regularizations  , Regularization , RegularizationBase , REGULARIZATIONS    ),
    "regularizations_fun" : Case(lib.regularizations  , Regularization , None               , REGULARIZATION_FNS ),
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


def test_name(case_name: str, item_name: object) -> None:
    r"""Check if the name of the class matches the item name."""
    if not isinstance(item_name, str):
        return
    if case_name == "activations":
        return
    case = CASES[case_name]
    obj = case.elements[item_name]
    # fallback for jit.ScriptFunction
    name = getattr(obj, "__name__", getattr(obj, "name", None))
    assert name == item_name


def test_name_casing(case_name: str, item_name: object) -> None:
    r"""Check that the case name is in snake_case."""
    if not isinstance(item_name, str):
        return
    if case_name == "activations":
        assert item_name == item_name.lower()
        assert "_" not in item_name
        return
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


def test_dict_complete(case_name: str) -> None:
    r"""Check if all encoders are in the ENCODERS constant."""
    case = CASES[case_name]

    match case.base_class:
        case None:  # skip for function-dicts
            return
        case base_class:
            missing: dict[type, str] = {
                cls: name
                for name, cls in vars(case.module).items()
                if (
                    not is_protocol(cls)
                    and not isabstract(cls)
                    and isinstance(cls, type)
                    and issubclass(cls, base_class)
                    and cls is not base_class
                    and cls.__name__ in case.module.__all__
                    # and __init__ is compatible with base_class.__init__
                )
                and cls not in case.elements.values()
            }

            if missing.keys() - case.excluded:
                raise AssertionError(f"Missing {case_name}: {missing}")
